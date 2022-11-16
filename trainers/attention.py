import os
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
import torch.optim as optim
import numpy as np
import tqdm
from dassl.metrics import compute_accuracy
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.optim import build_optimizer, build_lr_scheduler
import time
from clip import clip
from clip.model import convert_weights
import pandas as pd
import tqdm
import random

from .coop import load_clip_to_cpu
from .imagenet_templates import IMAGENET_TEMPLATES, IMAGENET_TEMPLATES_SELECT


@TRAINER_REGISTRY.register()
class ZeroshotCLIP(TrainerX):
    def build_model(self, df):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        self.classnames = classnames
        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg) 
        self.clip_model = clip_model.float().to("cuda")    
        self.conceptnet_sentences = torch.load("conceptnet_features.pkl")
        
        class LowDimer(nn.Module):
            def __init__(self):
                super().__init__()

                self.img_lowdim_trf = nn.Linear(512,512)
                self.txt_lowdim_trf = nn.Linear(512,512)

            def forward(self,x):
                return x #This is dummy forward.

        low_dimer = LowDimer()       
    
        low_dimer.to(self.device)
        clip_model.to(self.device)

        self.img_lowdim_trf = low_dimer.img_lowdim_trf
        self.txt_lowdim_trf = low_dimer.txt_lowdim_trf
        self.optim = build_optimizer(low_dimer, cfg.OPTIM )
        #import pdb; pdb.set_trace()
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None


        self.register_model("low_dimer", low_dimer, self.optim, self.sched)
        
        # added
        self.df = df
        self.mode = cfg.MODE

        if self.mode == 'attention':
            conceptnet_sentences = self.conceptnet_sentences
            original_length = []
            max_num = 0
            for cs in conceptnet_sentences :
                original_length.append(cs.shape[0])
                if cs.shape[0] > max_num : max_num = cs.shape[0]
            
            self.conceptnet_sentences = torch.stack([ 
                    F.pad(class_prompts,(0, 0, 0, max_num - class_prompts.shape[0]), mode='constant', value=0) \
                            for class_prompts in conceptnet_sentences ])
            self.max_num = max(original_length)
            mask_sequence = []
            for leng in original_length :
                mask = torch.zeros(max_num)
                mask[leng:] = -99999999
                mask_sequence.append(mask)
            
            self.mask_sequence = torch.stack(mask_sequence).to(self.device)       

    def model_inference(self, image_features):

        if self.mode == 'attention' :
            # nn.linear에 각 prompt들을 다 forward시켜서, lower dimension embedding을 구해야함. 
            # 근데 이거 offline으로 해두면, 학습이 되는게 아니라서 무조건 여기서 해야함.
            x_low = self.img_lowdim_trf(image_features)
            txt_low = self.txt_lowdim_trf(self.conceptnet_sentences.view(-1,512))  # 838000x128
            M = x_low@txt_low.T                                             # Nx838000
            M = M.view(-1,1000,838)                                        # Nx1000x838
            M += self.mask_sequence.unsqueeze(0)
            #M = F.gumbel_softmax(M,tau=1.0, dim=-1, hard=True)              # Nx1000x838
            M = F.softmax(M, dim=-1)              # Nx1000x838
            M = torch.bmm(M.permute((1,0,2)),self.conceptnet_sentences) # Nx1000x512
            M = M.permute((1,2,0)) # Nx512x1000
            x = image_features/image_features.norm(dim=-1, keepdim=True)
            M = M / M.norm(dim=-1, keepdim=True)
            sims = x@M

            #extract diagonal
            ret = []
            for i in range(x.shape[0]) : #for .. N
                ret.append(sims[i,i,:]) #N_C
            ret  = torch.stack(ret).to("cuda") #NxN_C
            logit_scale = self.clip_model.logit_scale.exp()   
            logits = logit_scale*ret
              
            return logits
    
    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        label = label.type(torch.int64)
        prec = self.cfg.TRAINER.COOP.PREC
        if prec == "amp":
            with autocast():
                output = self.model_inference(image)
                loss = F.cross_entropy(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output = self.model_inference(image)
            loss = F.cross_entropy(output, label.squeeze(dim=-1))
            self.model_backward_and_update(loss) #Make sure that CLIP encoder is not trained,, and attention nnLinear is only triained ..
        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
        }
        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch[0]
        label = batch[1]
        #input = batch["img"]
        #label = batch["label"]

        input = input.to(self.device)
        label = label.to(self.device)
        return input, label





