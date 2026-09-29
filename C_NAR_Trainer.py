#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 31 11:14:23 2025

@author: root
"""


import os

import torch
import librosa
from torch.utils.data import  DataLoader
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import  destroy_process_group
import dac 
from Models.C_NAR import C_NAR_Model
from Trainer import Trainer, labled_AudioDataset, cosine_decay, ddp_setup, par_count , compute_sisdr
import tqdm
from einops import rearrange
import numpy as np
from pystoi import stoi
import soundfile as sf
import scipy as sp


"""

TRAINING PARAMETERS

"""
device='cuda'
DAC_Model ="Path/to/DAC_model_16khz.pth"
DATA_PATHS = [
    "Libri2Mix/train/clean",
    "Libri2Mix/train/noisy",
    "Libri2Mix/valid/clean",
    "Libri2Mix/valid/noisy",
    ]
sr=16000
NUM_EPOCHS =300
BATCH_SIZE =256
SAVE_EVERY=50
GEN_EVERY=10
LEARNING_RATE =0.0005 * (BATCH_SIZE/256)
checkpoint_path=None #"Checkpoints/Model_Chechpoint.pt"

"""

MODEL HYPER-PARAMETERS

"""
NAME="SE_Model_Name"

MAX_LEN =50
Nq=12
Params = {"input_dim":1024,
       "dim":384,
       "max_seq_len":MAX_LEN,
       "N_layers":10,
       "dim_head":32,
       "heads":12 }
    

def load_train_objs( DAC_Model, device,Params,LEARNING_RATE,checkpoint_path,NUM_EPOCHS):

    DAC_Model = dac.DAC.load(DAC_Model)
    DAC_Model.encoder.to(device)
    DAC_Model.quantizer.to(device)
    
    if checkpoint_path:
        print("checkpoint found : " , checkpoint_path)
        checkpoint = torch.load(checkpoint_path,map_location=device,weights_only=False)
        SE_Model_params = checkpoint['SE_Model_params']
    else : 
        SE_Model_params = Params
    SE_Model= C_NAR_Model(**SE_Model_params).to(device)
    optimizer = torch.optim.AdamW(SE_Model.parameters(),
                                  lr= LEARNING_RATE,
                                  betas=(0.9, 0.95),
                                  weight_decay=0.05)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                     lr_lambda=lambda epoch: cosine_decay(epoch,10, NUM_EPOCHS))

    return DAC_Model, SE_Model, optimizer, lr_scheduler

def align_and_crop_signals(signal1, signal2):
    """
    Aligns signal2 to signal1 using cross-correlation and crops both signals to their overlapping portion.
    
    Parameters:
    signal1 (numpy.ndarray): The reference signal.
    signal2 (numpy.ndarray): The signal to be aligned.
    
    Returns:
    numpy.ndarray: The aligned and cropped version of signal1.
    numpy.ndarray: The aligned and cropped version of signal2.
    int: The shift amount to align signal2 to signal1.
    """
    # Perform cross-correlation
    correlation = sp.signal.correlate(signal1, signal2, mode='full')
    
    # Find the index of the maximum correlation
    max_corr_index = np.argmax(correlation)
    
    # Calculate the shift amount
    shift_amount = max_corr_index - (len(signal2) - 1)
    
    # Determine the overlap region
    if shift_amount > 0:
        start1 = shift_amount
        end1 = len(signal1)
        start2 = 0
        end2 = len(signal1) - shift_amount
    else:
        start1 = 0
        end1 = len(signal1) + shift_amount
        start2 = -shift_amount
        end2 = len(signal2)
    
    aligned_signal1 = signal1[start1:end1]
    aligned_signal2 = signal2[start2:end2]
    
    return aligned_signal1, aligned_signal2, shift_amount




class C_NAR_Trainer(Trainer):
    
    def process_batch_train_audio(self, batch):
        
        x=self.Encode(batch[0].to(self.gpu_id))
        y=self.Encode(batch[1].to(self.gpu_id))

        x=rearrange(x, "b d t -> b t d")
        y=rearrange(y, "b d t -> b t d")
        
        loss = self.SE_Model(embeds=y,clean_embeds=x,return_loss=True)
        return loss
    @torch.no_grad()
    def _denoise_validation(self,epoch):
        """
        Performs validation specifically for the denoising phase.

        Args:
            epoch: Current epoch number.

        """
        self.SE_Model.module.eval()
        with torch.no_grad():
            SI_SDR = []
            ESTOI =[]
            
            k=np.random.randint(self.val_dataset.__len__())# sampling random int for audio generation ***
                        
            for i , data in enumerate(tqdm.tqdm(self.val_dataset, desc="Validation : Denoising")):
                
                x=self.Encode(data[0].to(self.gpu_id))
                y=self.Encode(data[1].to(self.gpu_id))

                y=rearrange(y, "b d t -> b t d")
                x=rearrange(x, "b d t -> b t d")
                
                y_ = self.SE_Model(embeds=y,clean_embeds=x,return_loss=False)
                y_=rearrange(y_ ,'b t d-> b d t')
                y_=self.DAC_Model.module.quantizer(y_,12)[0]
                y = self.DAC_Model.module.decoder(y_)

                y = y.detach().cpu().numpy().squeeze()
                xc = data[0].detach().cpu().numpy().squeeze()
                x_n = data[1].detach().cpu().numpy().squeeze()
                for j, x_i in enumerate(xc):
                    try:
                        x_i,y_j,_=align_and_crop_signals(x_i, y[j])
                        x_i=x_i[:y_j.shape[0]]
                        
                        SI_SDR.append(compute_sisdr(y_j, x_i))
                        ESTOI.append(stoi(x_i, y_j, 16000, extended=True))
                    except Exception as e:
                        print(f"STOI computation failed for batch {i} : {j}: {e}")
                        break
                    if (j ==1 ) and (i==k) :
                       sf.write(f"{self.NAME}_Audio/Reconstructed_Denoised_{epoch}.wav", y_j , 16000)
                       sf.write(f"{self.NAME}_Audio/Noisy_Signal{epoch}.wav",            x_n[j], 16000)
                       sf.write(f"{self.NAME}_Audio/Clean_Signal_{epoch}.wav",           x_i, 16000)
                       
                       
######### Comet is used to log  loss values, SE metric, and audio examples
        #                experiment.log_audio(
        #                     audio_data= y_j,
        #                     sample_rate=16000,
        #                     file_name=f" Reconstructed Signal After Denoising: {epoch}",
        #                     step=epoch,)
        #                experiment.log_audio(
        #                     audio_data= x_n[j],
        #                     sample_rate=16000,
        #                     file_name=f" Original Noisy Signal: {epoch}",
        #                     step=epoch,)
        #                experiment.log_audio(
        #                     audio_data= x_i,
        #                     sample_rate=16000,
        #                     file_name=f" Original Clean Signal: {epoch}",
        #                     step=epoch,)
        # experiment.log_metric("SI_SDR", np.mean(SI_SDR), epoch=epoch)
        # experiment.log_metric("ESTOI", np.mean(ESTOI), epoch=epoch)
            
            
    def _save_checkpoint(self, epoch):
        SE_Model_params = {
            'input_dim':self.SE_Model.module.input_dim,
            'dim': self.SE_Model.module.dim,
            'max_seq_len': self.SE_Model.module.max_seq_len,
            'N_layers': len(self.SE_Model.module.noise_transformer.layers),
            'dim_head': int(self.SE_Model.module.dim/self.SE_Model.module.noise_transformer.layers[0].heads),
            'heads':self.SE_Model.module.noise_transformer.layers[0].heads#,
        }
        ckp = self.SE_Model.module.state_dict()
        opt = self.optimizer.state_dict()
        PATH = f"Checkpoints/{self.NAME}_ckpt_{epoch}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict':ckp,
            'optimizer_state_dict': opt,
            'SE_Model_params': SE_Model_params
            }, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")


    def _run_epoch_(self, epoch ):
        print(f"[GPU {self.gpu_id}] Epoch {epoch} | Steps:{len(self.dataset)}")
        self.SE_Model.module.train()
        self.loss_values = []
        for i, data in enumerate(tqdm.tqdm(self.dataset, desc=f"Training Epoch {epoch + 1}")):
            loss = self._run_batch(data,epoch )
            self.loss_values.append(loss)
        self.loss_avg.append(np.mean(self.loss_values))
        Validation_loss=self._validate()
        self.val.append(Validation_loss)

        # experiment.log_metric("Validation_loss", Validation_loss, epoch=epoch)
        # experiment.log_metric("train_loss", np.mean(self.loss_values), epoch=epoch)
        # experiment.log_metric("Learning_Rate", self.scheduler.get_last_lr(), epoch=epoch)

        self.scheduler.step()
        

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
    noisy_training= librosa.util.find_files(  DATA_PATHS[1], ext='wav') 
    dataset=labled_AudioDataset(clean_training, noisy_training, max_len,random_start=True)
    train_dataset=DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, sampler=DistributedSampler(dataset))
    
    
    clean_validation= librosa.util.find_files( DATA_PATHS[2], ext='wav')
    noisy_validation= librosa.util.find_files( DATA_PATHS[3], ext='wav')
    dataset=labled_AudioDataset(clean_validation, noisy_validation, max_len ,random_start=True)
    val_dataset=DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,sampler=DistributedSampler(dataset))
    
    DAC_Model, SE_Model, optimizer, scheduler = load_train_objs(DAC_Model,device,Params,LEARNING_RATE,checkpoint_path,NUM_EPOCHS)
    
    
    parameter_count=par_count(SE_Model)
    print("Number of Model Parameters :",parameter_count)

    trainer = C_NAR_Trainer(  DAC_Model
                      , SE_Model
                      , rank
                      , train_dataset
                      , val_dataset
                      , optimizer
                      , scheduler
                      , SAVE_EVERY
                      , GEN_EVERY
                      , Nq=Nq
                      , sr=sr
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
