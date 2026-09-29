#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 11:13:06 2026

@author: root
"""


from comet_ml import Experiment

import os

import torch
import librosa
from torch.utils.data import  DataLoader
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import  destroy_process_group
import dac 
from Models.C_AR_Prior import C_AR_Model_Prior
from Models.C_NAR import C_NAR_Model
from torch.utils.data import Dataset 
from Trainer import Trainer,  ddp_setup
import tqdm
from einops import rearrange
import numpy as np
import soundfile as sf


"""

TRAINING PARAMETERS

"""
device='cuda'
DAC_Model ="Path/to/DAC_model_16khz.pth"
DATA_PATH="path/to/validation/Datasets"
sr=16000
NUM_EPOCHS =20
BATCH_SIZE =2
SAVE_EVERY=30
GEN_EVERY=2
LEARNING_RATE =0.00002
checkpoint_path="Checkpoints/C-NAR_ckpt.pt"
prior_path="Checkpoints/Prior_ckpt.pt"

"""

MODEL HYPER-PARAMETERS

"""
NAME="TTA_Data_set_name"

MAX_LEN =50

class AudioDataset(Dataset):
    def __init__(self, Noisy_list, max_len, random_start=True):
        self.Noisy_list = Noisy_list
        self.max_len = max_len
        self.random_start = random_start
    def __len__(self):
        return len(self.Noisy_list)

    def __getitem__(self, idx):
        wav_file = self.Noisy_list[idx]
        with sf.SoundFile(wav_file) as f:
            total_frames = len(f)
        required_frames = self.max_len
        if total_frames < required_frames:
            speech, sr = sf.read(wav_file)
            pad_size = required_frames - total_frames
            speech = np.pad(speech, (0, pad_size), 'constant')
        else:
            if self.random_start:
                rand_start = torch.randint(0, total_frames - required_frames + 1, (1,)).item()
            else:
                rand_start=0
            rand_stop = rand_start + required_frames
            speech, sr = sf.read(wav_file, start=rand_start, stop=rand_stop)
            speech=speech/np.max(np.abs(speech))
        speech = speech[np.newaxis, np.newaxis,:]
        
        return torch.from_numpy(speech).float().squeeze(0)

def load_train_objs( DAC_Model, device, prior_path):

    DAC_Model = dac.DAC.load(DAC_Model)
    DAC_Model.encoder.to(device)
    DAC_Model.quantizer.to(device)
    

    if not prior_path:
        raise ValueError("Prior checkpoint path must be provided.")

    print("Loading prior checkpoint:", prior_path)
    checkpoint = torch.load(
        prior_path,
        map_location=device,
        weights_only=False
    )

    Prior_Model_params = checkpoint['Prior_Model_params']
    prior_model = C_AR_Model_Prior(**Prior_Model_params).to(device)
    prior_model.load_state_dict(checkpoint['model_state_dict'])

    prior_model.eval()

    return DAC_Model,prior_model 

def kl_full_gaussian(mu_q, mu_p, log_var_p, W):
    """
    KL( N(mu_q, I) || N(mu_p, W diag(exp(log_var_p)) W^T) )

    Args:
        mu_q:      (B,T,D)  mean from inference model q(x|y)
        mu_p:      (B,T,D)  mean from prior p(x)
        log_var_p: (B,T,D)  log variances from prior
        W:         (D,D)    orthogonal matrix

    Returns:
        kl: scalar
    """

    B, T, D = mu_q.shape

    v = torch.exp(log_var_p)                 # (B,T,D)
    delta = mu_q - mu_p                      # (B,T,D)

    # rotate
    delta_tilde = torch.einsum('ij,btj->bti', W.T, delta)

    kl  = 0.5* (delta_tilde**2 / v).sum(dim=-1)  # (B,T)

    return kl.mean()


class C_NAR_TTA(Trainer):
    def __init__(self,
                 DAC_Model,
                 SE_Model,
                 Prior,
                 gpu_id,
                 dataset,
                 file,
                 optimizer,
                 gen_every,
                 NAME):
    

        self.DAC_Model = DAC_Model.to(gpu_id)
        self.SE_Model = SE_Model.to(gpu_id)
        self.Prior = Prior.to(gpu_id)
        self.gpu_id = gpu_id
        self.DAC_Model = DDP(DAC_Model, device_ids=[gpu_id])
        self.Prior = DDP(Prior, device_ids=[gpu_id])
        self.SE_Model = DDP(SE_Model, device_ids=[gpu_id],find_unused_parameters=True)
        self.dataset = dataset
        self.file=file
        self.optimizer = optimizer
        self.gen_every = gen_every
        self.sr=16000
        self.total_steps = 0
        self.loss_values = []
        self.loss_avg = []
        self.val = []
        self.lr = []
        self.NAME=NAME

    def process_batch_train_audio(self, batch):
        
        y = self.Encode(batch.to(self.gpu_id))
        y = rearrange(y, "b d t -> b t d")
    
        # Inference model output
        y_ = self.SE_Model(embeds=y, clean_embeds=None, return_loss=False)
    
        # Prior evaluation (frozen)
        with torch.no_grad():
            mu_p, log_var = self.Prior(clean_embeds=y_, return_loss=False)
            W, _ = torch.linalg.qr(self.Prior.module.W_raw)
    
        # KL term
        kl = kl_full_gaussian(
            mu_q=y_,
            mu_p=mu_p[:, :-1],
            log_var_p=log_var[:, :-1],
            W=W
        )
        loss = 0.1* kl  

        return loss
    def Denoise_full_seq(self , file_path ,epoch,step):
        self.Prior.module.eval()
        with torch.no_grad():
            speech, sr = sf.read(file_path)
            speech=speech/np.max(np.abs(speech))
            speech = speech[np.newaxis, np.newaxis, :]
            speech = torch.from_numpy(speech).float()
            z=self.Encode(speech.to(self.gpu_id) )
            x = torch.empty((1, z.shape[1],  0), device=device)
            for t in range( 0,z.shape[-1],100):
                zc=z[:,:,t:t+100]
                zc = rearrange(zc, 'b d t -> b t d')
            
                x_ = self.SE_Model(embeds=zc,clean_embeds=None,return_loss=False)
                
                x_=rearrange(x_ ,'b t d-> b d t')
                x=torch.cat((x,x_),dim=-1)
            x=self.DAC_Model.module.quantizer(x,12)[0]
            x = self.DAC_Model.module.decoder(x)
    
            x = x.detach().cpu().numpy().squeeze()
        name=os.path.splitext(os.path.basename(file_path))[0]
        sf.write(f"TTA_Audio/{self.NAME}/{name}/_Enhanced_{epoch:0=2}.wav",x , 16000)
            
    

    def _run_epoch_(self, epoch ,step):
        print(f"[GPU {self.gpu_id}] Epoch {epoch} | Steps:{len(self.dataset)}")
        self.SE_Model.module.train()
        self.Prior.module.eval()
        self.loss_values = []
        for i, data in enumerate(tqdm.tqdm(self.dataset, desc=f"Training Epoch {epoch + 1}")):
            loss = self._run_batch(data,epoch )
            self.loss_values.append(loss)
        self.loss_avg.append(np.mean(self.loss_values))

        
    
    def train(self, max_epochs, start_epoch ,step):
        name=os.path.splitext(os.path.basename(self.file))[0]

        os.makedirs(f"TTA_Audio/{self.NAME}/{name}", exist_ok=True)
        
        speech, sr = sf.read(self.file)
        speech = speech[np.newaxis, np.newaxis, :]
        speech = torch.from_numpy(speech).float()
        
        
        for epoch in range(start_epoch, max_epochs):
            if (self.gpu_id == 0) and (epoch % self.gen_every == 0):
                self.Denoise_full_seq(self.file , epoch,step)
            self._run_epoch_(epoch ,step=step )
               
        self.Denoise_full_seq(self.file , max_epochs,step=step)


def main(rank: int,
         world_size: int,
         DAC_Model,
         device,
         DATA_PATH,
         GEN_EVERY:int,
         NUM_EPOCHS: int,
         BATCH_SIZE: int,
         LEARNING_RATE,
         MAX_LEN ,
         checkpoint_path,
         prior_path,
         NAME,
         sr
         ):
    
    ddp_setup(rank, world_size)
    max_len=int(sr*(MAX_LEN/50))
    TRAINING= librosa.util.find_files(DATA_PATH, ext='wav')
    
    DAC_Model, prior_model= load_train_objs(
        DAC_Model,
        device,
        prior_path)
    
    
    for step, file in enumerate(TRAINING):
        print("Test-time Adaptaion on ", os.path.basename(file))
        DATA=[file, file]
        dataset=AudioDataset(DATA, max_len,random_start=False)
        train_dataset=DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, sampler=DistributedSampler(dataset))
        if not checkpoint_path:
            raise ValueError("Prior checkpoint path must be provided.")
        print("checkpoint found : " , checkpoint_path)
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False
        )
        SE_Model_params = checkpoint['Cont_Model_params']
        SE_Model= C_NAR_Model(**SE_Model_params).to(device)
        SE_Model.load_state_dict(checkpoint['model_state_dict'])
        optimizer = torch.optim.SGD(SE_Model.parameters(),
                                      lr= LEARNING_RATE,
                                      momentum=0.9)
        trainer = C_NAR_TTA(DAC_Model
                              , SE_Model
                              , prior_model
                              , rank
                              , train_dataset
                              , file
                              , optimizer
                              , GEN_EVERY
                              , NAME)
        
        start_epoch = 0
        try:
            trainer.train(NUM_EPOCHS, start_epoch ,step)
        except Exception as e  :
            print("FAILED : ",file ,'\n', e)
    destroy_process_group()

if __name__ == '__main__':

    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size,
                         DAC_Model,
                         device,
                         DATA_PATH,
                         GEN_EVERY,
                         NUM_EPOCHS,
                         BATCH_SIZE,
                         LEARNING_RATE,
                         MAX_LEN,
                         checkpoint_path,
                         prior_path,
                         NAME,
                         sr ), nprocs=world_size)
