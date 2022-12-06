import os
import time
import torch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from feature_extractor import set_data_loader, get_conceptnet_feature
from torch.cuda.amp import GradScaler, autocast
import torch.optim as optim
from dassl.metrics import compute_accuracy
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.optim import build_optimizer, build_lr_scheduler
from clip import clip
import random
import wandb
import math

from dassl.utils import (
    MetricMeter, AverageMeter, mkdir_if_missing)
    
import pdb
import os.path as osp
import datetime
from tqdm import tqdm
from dassl.optim import build_optimizer, build_lr_scheduler
from .coop import load_clip_to_cpu
from .imagenet_templates import IMAGENET_TEMPLATES, IMAGENET_TEMPLATES_SELECT





CUSTOM_TEMPLATES = {
    "OxfordPets": "a photo of a {}, a type of pet.",
    "OxfordFlowers": "a photo of a {}, a type of flower.",
    "FGVCAircraft": "a photo of a {}, a type of aircraft.",
    "DescribableTextures": "{} texture.",
    "EuroSAT": "a centered satellite photo of {}.",
    "StanfordCars": "a photo of a {}.",
    "Food101": "a photo of {}, a type of food.",
    "SUN397": "a photo of a {}.",
    "Caltech101": "a photo of a {}.",
    "UCF101": "a photo of a person doing {}.",
    "ImageNet": "a photo of a {}.",
    "ImageNetSketch": "a photo of a {}.",
    "ImageNetV2": "a photo of a {}.",
    "ImageNetA": "a photo of a {}.",
    "ImageNetR": "a photo of a {}.",
}

Tensor = torch.Tensor

def gumbel_softmax(logits: Tensor, tau: float = 1, hard: bool = False, eps: float = 1e-10, dim: int = -1) -> Tensor:

    gumbels = (
        -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
    )  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)

    if hard:
        # Straight through.
        index = y_soft.topk(2, dim)[1]
        value = y_soft.max(dim, keepdim=True)[0]
        y_hard = (y_soft > 0).float()
        
        #y_hard = (y_soft > value*0.8).float()
        #y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret

#for clip weight 

def clip_classifier(prompt):
    with torch.no_grad():
        clip_weights = []

        for cp in prompt:
            # Tokenize the prompts
            class_embeddings = cp
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            clip_weights.append(class_embedding)

        clip_weights = torch.stack(clip_weights, dim=1).cuda()
    return clip_weights


@TRAINER_REGISTRY.register()
class ZeroshotCLIP(TrainerX):
    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        self.classnames = classnames
        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)        
        self.clip_model = clip_model.float().to("cuda")
       
        self.train_dataloader, self.valid_dataloader, self.test_dataloader = set_data_loader(cfg, self.dm, self.device, clip_model)      
        self.input_dim = 1024 if cfg.MODEL.BACKBONE.NAME.lower() == 'rn50' else 512
        # sentence embedding
        self.subsample_class = cfg.DATASET.SUBSAMPLE_CLASSES
        self.search_level = cfg.DATASET.SEARCH_LEVEL
        self.emb_root = cfg.DATASET.EMB_ROOT
        self.dataset_name = cfg.DATASET.NAME.lower()

        self.logit_scale = cfg.TRAINER.MY_MODEL.SCALE
        self.dropout = cfg.TRAINER.MY_MODEL.DROPOUT
        self.wd = cfg.OPTIM.WEIGHT_DECAY
        self.mode = cfg.MODE
        self.alpha = cfg.TRAINER.MY_MODEL.ALPHA
        self.entity = cfg.ENTITY
        self.name = f'dropout={self.dropout}_wd={self.wd}_logit_scale{self.logit_scale}_{cfg.MODE}_alpha{self.alpha}'
        
        # pdb.set_trace()
        if 'LOAD_EPOCH' in cfg.keys():
            self.t = cfg.LOAD_EPOCH
        else:
            self.t = 100
        a = self.cfg.TRAINER.MY_MODEL.max_temp
        b = math.log(self.cfg.TRAINER.MY_MODEL.min_temp/self.cfg.TRAINER.MY_MODEL.max_temp)/self.cfg.TRAINER.MY_MODEL.ANNEAL_EPOCH 
        self.temp = a* math.exp(b*self.t)
        class LowDimer(nn.Module):
            def __init__(self):
                super().__init__()
                self.dropout = cfg.TRAINER.MY_MODEL.DROPOUT
                self.input_dim = 1024 if cfg.MODEL.BACKBONE.NAME.lower() == 'rn50' else 512
                #self.img_lowdim_trf = nn.Linear(512,512)
                self.img_lowdim_trf = nn.Sequential(
                        nn.Dropout(self.dropout), 
                        nn.Linear(self.input_dim, self.input_dim) , 
                        )

                #self.txt_lowdim_trf = nn.Linear(512,512)
                self.txt_lowdim_trf = nn.Sequential(
                        nn.Dropout(self.dropout), 
                        nn.Linear(self.input_dim, self.input_dim) , 
                        )

                #self.prompt_dim = nn.Linear(512,512)
                self.prompt_dim = nn.Sequential(
                        nn.Dropout(self.dropout), 
                        nn.Linear(self.input_dim,self.input_dim) , 
                        )


            def forward(self,x):
        
                return x #This is dummy forward.

        self.low_dimer = LowDimer()

        self.low_dimer.to(self.device)
        clip_model.to(self.device)

        # self.img_lowdim_trf = low_dimer.img_lowdim_trf
        # self.txt_lowdim_trf = low_dimer.txt_lowdim_trf
        # self.prompt_lowdim = low_dimer.prompt_dim 
        

        # automated conceptnet_feature extracting
        backbone_name = cfg.MODEL.BACKBONE.NAME.lower().replace('/', '')
        conceptnet_sentences_path = f"{self.emb_root}/{self.dataset_name}/conceptnet_features_{backbone_name}_{self.subsample_class}_level_{self.search_level}.pkl"
        if not os.path.exists(conceptnet_sentences_path):
            self.conceptnet_sentences = get_conceptnet_feature(emb_root=self.emb_root,
                                                             dataset=self.dataset_name,
                                                             backbone=backbone_name,
                                                             subsample_class=self.subsample_class,
                                                             level=self.search_level,
                                                             classnames=self.classnames,
                                                             clip_model=clip_model,
                                                             device=self.device)
        else:
            self.conceptnet_sentences = torch.load(conceptnet_sentences_path)
        
        #for clip weight 
        self.conceptnet_prompt = self.conceptnet_sentences
        
        self.optim = build_optimizer(self.low_dimer, cfg.OPTIM )
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None
 
        temp = CUSTOM_TEMPLATES[cfg.DATASET.NAME]

        self.register_model("low_dimer", self.low_dimer, self.optim, self.sched)
    
        # pdb.set_trace()
        # added
        
        if self.mode == 'ZS':
            prompts = [temp.format(c.replace("_", " ")) for c in classnames]
            print(f"Prompts: {prompts}")
            prompts = torch.cat([clip.tokenize(p) for p in prompts])
            prompts = prompts.to(self.device) 
            with torch.no_grad():
                text_features = clip_model.encode_text(prompts)

        elif self.mode == 'random':
            prompts = [c[random.randint(0,c.shape[0]-1)] for c in self.conceptnet_sentences]
            text_features = torch.stack(prompts).to(self.device)
            
        elif self.mode == 'average':
            text_features = torch.stack([cf.mean(dim=0) for cf in self.conceptnet_sentences] )

        else:
            conceptnet_sentences = self.conceptnet_sentences
            self.or_sent  = conceptnet_sentences
            original_length = []
            max_num = 0
            for cs in conceptnet_sentences :
                original_length.append(cs.shape[0])
                if cs.shape[0] > max_num : max_num = cs.shape[0]
            self.no_pad = conceptnet_sentences
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
            return     

        self.text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    def attention_parallel(self, image, text, mask, max_len:int, mode:str):
        if mode == 'weight_sum':
            image_features =  image
            text_features = text
        else:
            image_features =  self.low_dimer.img_lowdim_trf(image)
            text_features = self.low_dimer.txt_lowdim_trf(text)
            text = self.low_dimer.prompt_dim(text)


        clip_weights = clip_classifier(self.conceptnet_prompt)
        img_w = image @ clip_weights

        M = image_features @ text_features.view(-1,self.input_dim).T #830000x1000
        M = M.view(-1,len(self.classnames), max_len) #Nx1000x838
        M += mask.unsqueeze(0)

        if 'EVAL_ONLY' in self.cfg.keys():
            if self.cfg.EVAL_ONLY != 1:
                self.t = self.epoch
        else:
            self.t = self.epoch
        a = self.cfg.TRAINER.MY_MODEL.max_temp
        b = math.log(self.cfg.TRAINER.MY_MODEL.min_temp/self.cfg.TRAINER.MY_MODEL.max_temp)/self.cfg.TRAINER.MY_MODEL.ANNEAL_EPOCH 
        self.temp = a* math.exp(b*self.t)
        if mode == 'gumbel':
            M1 = F.gumbel_softmax(M, tau=self.temp, hard=True)
            M2 = M - M1.detach()
            M2 = F.gumbel_softmax(M2, tau=self.temp, hard=True)
            M = M1 + M2
        else:
            M = F.softmax(M, dim=-1)  # Nx1000x838    

        M = torch.bmm(M.permute((1,0,2)), text) # Nx1000x512
        M = M.permute((1,2,0)) # Nx512x1000
        M = M / M.norm(dim=1, keepdim=True)
        alpha = self.alpha
        sims = alpha * torch.einsum('ij,ijk->ik', image, M) + img_w #Nx1000
        ##dual softmax

        return sims


    def model_inference(self, image_features, split):
        if self.logit_scale == 0:
            logit_scale = self.clip_model.logit_scale.exp()
        else:
            logit_scale = self.logit_scale
        
        if split == None:
            image_features = image_features + 0.2 * torch.randn_like(image_features) 
        
        image_features_norm = image_features / image_features.norm(dim=-1, keepdim=True)
            

        if self.mode == 'gumbel' or self.mode == 'attention' or  self.mode == 'weight_sum' :      
            mask = self.mask_sequence
            max = self.max_num
            m = self.mode   
            text_features_norm = self.conceptnet_sentences / self.conceptnet_sentences.norm(dim=-1, keepdim=True) #normalize? or not

            sims = self.attention_parallel(image_features_norm, self.conceptnet_sentences, mask, max, m )
            logits = logit_scale * sims 
            return logits       
        
        else:
            logits = image_features_norm @ self.text_features.T
        
        logits = logit_scale * logits  
        return logits
    
    def before_train(self):
        directory = self.cfg.OUTPUT_DIR
        if self.cfg.RESUME:
            directory = self.cfg.RESUME
        self.start_epoch = self.resume_model_if_exist(directory)

        # Initialize summary writer
        writer_dir = osp.join(self.output_dir, "tensorboard")
        mkdir_if_missing(writer_dir)
        self.init_writer(writer_dir)

        # Remember the starting time (for computing the elapsed time)
        self.time_start = time.time()


    def run_epoch(self):
        self.set_model_mode("train")
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        #edited
        self.num_batches = len(self.train_dataloader) #원래 self.train_loader_x

        end = time.time()
        for self.batch_idx, batch in enumerate(self.train_dataloader): #edited self.train_loader_x
            data_time.update(time.time() - end)
            
            t = self.epoch
            a = self.cfg.TRAINER.MY_MODEL.max_temp
            b = math.log(self.cfg.TRAINER.MY_MODEL.min_temp/self.cfg.TRAINER.MY_MODEL.max_temp)/self.cfg.TRAINER.MY_MODEL.ANNEAL_EPOCH 
            self.temp = a* math.exp(b*t)

            loss_summary = self.forward_backward(batch)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            meet_freq = (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0
            only_few_batches = self.num_batches < self.cfg.TRAIN.PRINT_FREQ
            if meet_freq or only_few_batches:
                nb_remain = 0
                nb_remain += self.num_batches - self.batch_idx - 1
                nb_remain += (
                    self.max_epoch - self.epoch - 1
                ) * self.num_batches
                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                info = []
                info += [f"epoch [{self.epoch + 1}/{self.max_epoch}]"]
                info += [f"batch [{self.batch_idx + 1}/{self.num_batches}]"]
                info += [f"time {batch_time.val:.3f} ({batch_time.avg:.3f})"]
                info += [f"data {data_time.val:.3f} ({data_time.avg:.3f})"]
                info += [f"{losses}"]
                info += [f"lr {self.get_current_lr():.4e}"]
                info += [f"eta {eta}"]
                print(" ".join(info))

            n_iter = self.epoch * self.num_batches + self.batch_idx
            for name, meter in losses.meters.items():
                self.write_scalar("train/" + name, meter.avg, n_iter)
            self.write_scalar("train/lr", self.get_current_lr(), n_iter)


            end = time.time()

        wandb.log({
            "train_loss " : losses.meters["loss"].avg,
            "train_acc" : losses.meters["acc"].avg
            }, step=self.epoch)


    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
        losses = AverageMeter()
        acces = AverageMeter()
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT
        # added
        if split == "val" and self.valid_dataloader is not None:#val_loader
            data_loader = self.valid_dataloader #val_loader
        else:
            split == "test"  # in case val_loader is None
            data_loader = self.test_dataloader   #val_loader

        print(f"Evaluate on the *{split}* set")
        
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)
            label = label.type(torch.int64)
            output = self.model_inference(input, split = "val")
            self.evaluator.process(output, torch.squeeze(label))
            #added
            loss = F.cross_entropy(output, torch.squeeze(label))

            losses.update(loss.item(), input.shape[0])
            acc = compute_accuracy(output, label)[0].item()
            acces.update(acc, input.shape[0])

        loss = losses.avg
        acc = acces.avg

        wandb.log({
            "test_loss" : loss,
            "test_acc" : acc
            }, step=self.epoch)

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        label = label.type(torch.int64)
        prec = self.cfg.TRAINER.COOP.PREC
        optim_name = self.cfg.OPTIM.NAME
        if prec == "amp":
            with autocast():
                output = self.model_inference(image, split = None)
                loss = F.cross_entropy(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            if optim_name == 'sam':
                output = self.model_inference(image, split = None)
                loss = F.cross_entropy(output, label.squeeze(dim=-1))
                loss.backward(retain_graph=True )
                #self.model_backward_and_update(loss) #Make sure that CLIP encoder is not trained,, and attention nnLinear is only triained ..
                self.optim.first_step(zero_grad=True)
                F.cross_entropy(output, label.squeeze(dim=-1)).backward()
                self.optim.second_step(zero_grad=True)
            else:
                output = self.model_inference(image, split = None)
                loss = F.cross_entropy(output, label.squeeze(dim=-1))
                self.model_backward_and_update(loss) #Make sure that CLIP encoder is not trained,, and attention nnLinear is only triained ..

            
        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
        }
        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def after_epoch(self):
        last_epoch = (self.epoch + 1) == self.max_epoch
        do_test = not self.cfg.TEST.NO_TEST
        meet_checkpoint_freq = (
            (self.epoch + 1) % self.cfg.TRAIN.CHECKPOINT_FREQ == 0
            if self.cfg.TRAIN.CHECKPOINT_FREQ > 0 else False
        )

        if do_test and self.cfg.TEST.FINAL_MODEL == "best_val":
            curr_result = self.test(split="val")
            is_best = curr_result > self.best_result
            if is_best:
                self.best_result = curr_result
                self.save_model(
                    self.epoch,
                    self.output_dir,
                    val_result=curr_result,
                    model_name="model-best.pth.tar"
                )

        if meet_checkpoint_freq or last_epoch:
            self.save_model(self.epoch, self.output_dir)

        self.test()

    def parse_batch_train(self, batch):
        input = batch[0]
        label = batch[1]
        #input = batch["img"]
        #label = batch["label"]

        input = input.to(self.device)
        label = label.to(self.device)
        return input, label


@TRAINER_REGISTRY.register()
class ZeroshotCLIP2(ZeroshotCLIP):
    """Prompt ensembling."""

    # templates = IMAGENET_TEMPLATES
    templates = IMAGENET_TEMPLATES_SELECT

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        clip_model.to(self.device)
        
        for params in clip_model.parameters():
            params.requires_grad_(False)

        self.clip_model = clip_model
        # add custom-made prompt
        if cfg.DATASET.NAME != "ImageNet":
            self.templates += [CUSTOM_TEMPLATES[cfg.DATASET.NAME]]

        num_temp = len(self.templates)
        print(f"Prompt ensembling (n={num_temp})")

        mean_text_features = 0
        for i, temp in enumerate(self.templates):
            prompts = [temp.format(c.replace("_", " ")) for c in classnames]
            prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(self.device)
            text_features = clip_model.encode_text(prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            mean_text_features = mean_text_features + text_features
        mean_text_features = mean_text_features / num_temp
        mean_text_features = mean_text_features / mean_text_features.norm(dim=-1, keepdim=True)

        self.text_features = mean_text_features
        self.clip_model = clip_model



