#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 13:59:29 2025

@author: root
"""


import os

from comet_ml import Experiment


import torch
import numpy as np
import tqdm
import soundfile as sf
from einops import rearrange
from torch.utils.data import Dataset ,DataLoader
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import dac 
import librosa
from Models.C_AR_Prior import C_AR_Model_Prior as C_AR_Model

"""

TRAINING PARAMETERS

"""
device='cuda'
DAC_Model ="Path/to/DAC_model_16khz.pth"
DATA_PATHS = [
    "Path/to/EARS/train",
    "Path/to/EARS/valid",
    ]


sr=16000
NUM_EPOCHS =200
BATCH_SIZE =256
SAVE_EVERY=50
GEN_EVERY=20
LEARNING_RATE =0.001 * (BATCH_SIZE*2/256)
checkpoint_path=None#"Path/to/ckpt"

"""

MODEL HYPER-PARAMETERS

"""
NAME="Prior_BIG-WSJ"

MAX_LEN =50
Nq=12
Params = {"input_dim":1024,
       "dim":128,
       "max_seq_len":MAX_LEN,
       "N_layers":5,
       "dim_head":32,
       "heads":4 }
    
def load_train_objs( DAC_Model, device,Params,LEARNING_RATE,checkpoint_path,NUM_EPOCHS):

    DAC_Model = dac.DAC.load(DAC_Model)
    DAC_Model.to(device)
    
    if checkpoint_path:
        print("checkpoint found : " , checkpoint_path)
        checkpoint = torch.load(checkpoint_path,map_location=device,weights_only=False)
        Prior_Model_params = checkpoint['Prior_Model_params']
        EPOCH=checkpoint ['epoch']
    else : 
        Prior_Model_params = Params
    Prior_Model= C_AR_Model(**Prior_Model_params).to(device)
    optimizer = torch.optim.AdamW(Prior_Model.parameters(),
                                  lr= LEARNING_RATE,
                                  betas=(0.9, 0.95),
                                  weight_decay=0.05)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                     lr_lambda=lambda epoch: cosine_decay(epoch,10, NUM_EPOCHS))

    return DAC_Model, Prior_Model, optimizer, lr_scheduler


def par_count(Model):
    parcount=0
    for p in Model.parameters():
        nn=1
        for s in list(p.size()):
            nn = nn*s
        parcount += nn
    return parcount



def ddp_setup(rank: int, world_size: int):
  """
  Args:
      rank: Unique identifier of each process
     world_size: Total number of processes
  """
  os.environ["MASTER_ADDR"] = "localhost"
  os.environ["MASTER_PORT"] = "12355"
  torch.cuda.set_device(rank)
  init_process_group(backend="nccl", rank=rank, world_size=world_size)



def cosine_decay(epoch, warmup_epochs, total_epochs):
    if epoch < warmup_epochs:
        return min((epoch + 1) / warmup_epochs, 1.0)  # Warmup phase (Linear)
    else:
        cosine_epoch = epoch - warmup_epochs
        return 0.5 * (1 + np.cos(np.pi * cosine_epoch / (total_epochs - warmup_epochs))) 


class labled_AudioDataset(Dataset):
    def __init__(self, clean_list, max_len, random_start=True):
        self.clean_list = clean_list
        self.max_len = max_len
        self.random_start = random_start
    def __len__(self):
        return len(self.clean_list)

    def __getitem__(self, idx):
        wav_file = self.clean_list[idx]
        with sf.SoundFile(wav_file) as f:
            total_frames = len(f)
        required_frames = self.max_len
        
        if total_frames < required_frames:
            speech, sr = sf.read(wav_file)
            pad_size = required_frames - total_frames
            speech=speech/np.max(np.abs(speech))
            speech = np.pad(speech, (0, pad_size), 'constant')
        else:
            if self.random_start:
                rand_start = torch.randint(0, total_frames - required_frames + 1, (1,)).item()
            else:
                rand_start=16000
            rand_stop = rand_start + required_frames
            speech, sr = sf.read(wav_file, start=rand_start, stop=rand_stop)
            speech=speech/np.max(np.abs(speech))
        speech = speech[np.newaxis, np.newaxis, :]
        return torch.from_numpy(speech).float().squeeze(0)


class Trainer:
    def __init__(self, DAC_Model, Prior_Model, gpu_id, dataset,Val_dataset,optimizer, scheduler, save_every,gen_every, Nq,sr,max_len,NAME):
        
        """
        Trainer class.

        Args:
            DAC_Model: The DAC model used for training.
            Prior_Model: The RQ-Transformer model.
            gpu_id: The GPU device ID to use (rank).
            dataset:     Training dataset.
            Val_dataset: Validation dataset.
            optimizer: Optimizer for model training.
            scheduler: Learning rate scheduler.
            save_every: Interval (in epochs) to save model checkpoints.
            gen_every: Interval (in epochs) to generate samples.
            Nq: Number of quantization levels.
            sr: Sample rate for audio processing.
            NAME: Name of the training experiment.

        Initializes distributed data parallel (DDP) models
        """
        self.DAC_Model = DAC_Model.to(gpu_id)
        self.Prior_Model = Prior_Model.to(gpu_id)
        self.gpu_id = gpu_id
        self.dataset = dataset
        self.val_dataset = Val_dataset
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_every = save_every
        self.gen_every = gen_every
        self.Nq = Nq
        self.sr=sr
        self.max_len=max_len
        self.total_steps = 0
        self.loss_values = []
        self.loss_avg = []
        self.val = []
        self.lr = []
        self.DAC_Model = DDP(DAC_Model, device_ids=[gpu_id])
        self.Prior_Model = DDP(Prior_Model, device_ids=[gpu_id],find_unused_parameters=True)
        self.NAME=NAME
        


    @torch.no_grad()
    def Encode(self,x):
        x =self.DAC_Model.module.preprocess(x,self.sr)
        return self.DAC_Model.module.encoder(x) # : (B x D x T)
    
    
    def sample_full_gaussian(self,mean, log_var, W):
        """
        Sample from N(mean, W diag(exp(log_var)) W^T)
    
        Args:
            mean:    (B, D)
            log_var: (B, D)
            W:       (D, D) orthogonal matrix
    
        Returns:
            sample:  (B, D)
        """
        B, D = mean.shape
    
        # Standard normal noise
        eps = torch.randn(B, D, device=mean.device)
    
        # Diagonal scaling
        std = torch.exp(0.5 * log_var)
        z = eps * std
    
        # Rotate
        z_rot = torch.einsum('ij,bj->bi', W, z)
    
        return mean + z_rot



    @torch.no_grad()
    def Generate(self,epoch,T):
        """
       Generate a sequence of clean speech embeddings from learned prior distribution
        """
        self.Prior_Model.module.eval()
        with torch.no_grad():
            full_seq = torch.empty((1, 0, 1024), dtype=torch.float, device=self.gpu_id)
            W, _ = torch.linalg.qr(self.Prior_Model.module.W_raw)
            for start in range(0, T, self.max_len): 
                x_in= torch.empty((1,0,1024),device=self.gpu_id)
                for i in tqdm.tqdm(range( self.max_len),desc="Generating from prior "):
                    mean,log_var=self.Prior_Model(clean_embeds=x_in)
                    pred = self.sample_full_gaussian(
                        mean=mean[:, -1],
                        log_var=log_var[:, -1],
                        W=W
                    )
                    pred=rearrange(pred, 'b d -> b 1 d')
                    x_in=torch.cat((x_in,pred),dim=1)
                full_seq=torch.cat((full_seq,x_in),dim=1)
            x_out=rearrange(full_seq ,'b t d-> b d t')
            x_out,_,_,_,_=self.DAC_Model.module.quantizer(x_out,self.Nq)
            x_audio = self.DAC_Model.module.decoder(x_out)
            x_audio=x_audio.detach().cpu().numpy().squeeze()
            sf.write(f"{self.NAME}_Audio/Generated_{epoch}.wav", x_audio , 16000)
            
############# Comet is used to log  loss values and audio examples

            # experiment.log_audio(
            #     audio_data= x_audio,
            #     sample_rate=16000,
            #     file_name=f"Example Audio epoch : {epoch}",
            #     step=epoch,)
            
    def process_batch_train_audio(self, batch):
        x=self.Encode(batch.to(self.gpu_id))
        x=rearrange(x, "b d t -> b t d")
        loss = self.Prior_Model(clean_embeds=x,return_loss=True)
        return loss.mean()

    def _run_batch(self, batch,epoch):
        self.optimizer.zero_grad()
        loss = self.process_batch_train_audio(batch)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    @torch.no_grad()
    def _validate(self ):
        self.Prior_Model.eval()
        with torch.no_grad():
            loss_values = []
            for i , data in enumerate(tqdm.tqdm(self.val_dataset, desc="Validation :")):
                    loss = self.process_batch_train_audio(data)
                    loss_values.append(loss.item())
        return np.mean(loss_values)


    def _run_epoch_(self, epoch ):
        print(f"[GPU {self.gpu_id}] Epoch {epoch} | Steps:{len(self.dataset)}")
        self.Prior_Model.module.train()
        self.loss_values = []
        for i, data in enumerate(tqdm.tqdm(self.dataset, desc=f"Training Epoch {epoch + 1}")):
            loss = self._run_batch(data,epoch )
            self.loss_values.append(loss)
        self.loss_avg.append(np.mean(self.loss_values))
        Validation_loss=self._validate()
        self.val.append(Validation_loss)
        
        self.scheduler.step()
        
################ Comet is used to log  loss values and audio examples

        # experiment.log_metric("Validation_loss", Validation_loss, epoch=epoch)
        # experiment.log_metric("train_loss", np.mean(self.loss_values), epoch=epoch)
        # experiment.log_metric("Learning_Rate", self.scheduler.get_last_lr(), epoch=epoch)


    def _save_checkpoint(self, epoch):
        Prior_Model_params = {
            'input_dim':self.Prior_Model.module.input_dim,
            'dim': self.Prior_Model.module.dim,
            'max_seq_len': self.Prior_Model.module.max_seq_len,
            'N_layers': len(self.Prior_Model.module.Conformer_Prior.layers),
            'dim_head': int(self.Prior_Model.module.dim/self.Prior_Model.module.Conformer_Prior.layers[0].heads),
            'heads':self.Prior_Model.module.Conformer_Prior.layers[0].heads
        }
        ckp = self.Prior_Model.module.state_dict()
        opt = self.optimizer.state_dict()
        PATH = f"Checkpoints/{self.NAME}_ckpt_{epoch}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict':ckp,
            'optimizer_state_dict': opt,
            'Prior_Model_params': Prior_Model_params
            }, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")


    def load_from_checkpoint(self, path):
        checkpoint = torch.load(path,weights_only=False)
        self.Prior_Model.module.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        self.scheduler.last_epoch=start_epoch
        return start_epoch


    def train(self, max_epochs, start_epoch ):
        os.makedirs(f"{self.NAME}_Audio", exist_ok=True)

        for epoch in range(start_epoch, max_epochs):
            self._run_epoch_(epoch  )
            if (epoch % self.gen_every == 0) :
                self.Generate(epoch, self.max_len)
            if (self.gpu_id == 0) and (epoch % self.save_every == 0):
                self._save_checkpoint(epoch)
        if (self.gpu_id == 0):
            self._save_checkpoint(max_epochs)
            self.Generate(epoch, self.max_len)





def main(rank: int,
         world_size: int,
         DAC_Model,
         device,
         DATA_PATHS,
         SAVE_EVERY: int,
         GEN_EVERY:int,
         NUM_EPOCHS: int,
         BATCH_SIZE: int,
         LEARNING_RATE,
         Nq,
         MAX_LEN ,
         Params,
         checkpoint_path,
         NAME,
         sr
         ):
    
    ddp_setup(rank, world_size)
    max_len=int(sr*(MAX_LEN/50))
    
    clean_training= librosa.util.find_files( DATA_PATHS[0], ext='wav') 
    dataset=labled_AudioDataset(clean_training , max_len,random_start=True)
    train_dataset=DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, sampler=DistributedSampler(dataset))
    
    
    clean_validation= librosa.util.find_files( DATA_PATHS[1], ext='wav') 
    dataset=labled_AudioDataset(clean_validation, max_len,random_start=True)
    val_dataset=DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,sampler=DistributedSampler(dataset))
    
    DAC_Model, Prior_Model, optimizer, scheduler = load_train_objs(DAC_Model,device,Params,LEARNING_RATE,checkpoint_path,NUM_EPOCHS)
    
    
    parameter_count=par_count(Prior_Model)
    print("Number of Model Parameters :",parameter_count)

    trainer =Trainer(  DAC_Model
                      , Prior_Model
                      , rank
                      , train_dataset
                      , val_dataset
                      , optimizer
                      , scheduler
                      , SAVE_EVERY
                      , GEN_EVERY
                      , Nq=Nq
                      , sr=sr
                      , max_len=MAX_LEN
                      , NAME=NAME)
    
    if checkpoint_path:
       start_epoch = trainer.load_from_checkpoint(checkpoint_path)
    else:
       start_epoch = 0
    trainer.train(NUM_EPOCHS, start_epoch )
    destroy_process_group()

if __name__ == '__main__':

    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size,
                         DAC_Model,
                         device,
                         DATA_PATHS,
                         SAVE_EVERY,
                         GEN_EVERY,
                         NUM_EPOCHS,
                         BATCH_SIZE,
                         LEARNING_RATE,
                         Nq,
                         MAX_LEN,
                         Params,
                         checkpoint_path,
                         NAME,
                         sr ), nprocs=world_size)
    # experiment.end()
