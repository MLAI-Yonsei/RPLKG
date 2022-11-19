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
import wandb
import pickle

from dassl.utils import (
    MetricMeter, AverageMeter, tolist_if_not, count_num_param, load_checkpoint,
    save_checkpoint, mkdir_if_missing, resume_from_checkpoint,
    load_pretrained_weights
)
import pdb
import time
import numpy as np
import os.path as osp
import datetime
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from dassl.data import DataManager
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.utils import (
    MetricMeter, AverageMeter, tolist_if_not, count_num_param, load_checkpoint,
    save_checkpoint, mkdir_if_missing, resume_from_checkpoint,
    load_pretrained_weights
)
from dassl.modeling import build_head, build_backbone
from dassl.evaluation import build_evaluator

from data_load import CustomImageDataset
from torch.utils.data import DataLoader
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



@TRAINER_REGISTRY.register()
class ZeroshotCLIP(TrainerX):
    def build_model(self, df):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        self.classnames = classnames
        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)        
        self.clip_model = clip_model.float().to("cuda")    
        self.emb_root = '/mlainas/KGPrompt_data/imagenet'
       

        class LowDimer(nn.Module):
            def __init__(self):
                super().__init__()

                #self.img_lowdim_trf = nn.Linear(512,512)
                self.img_lowdim_trf = nn.Sequential( nn.Dropout(0.4) , nn.Linear(512,512) )

                #self.txt_lowdim_trf = nn.Linear(512,512)
                self.txt_lowdim_trf = nn.Sequential( nn.Dropout(0.4) , nn.Linear(512,512) )

                #self.prompt_dim = nn.Linear(512,512)
                self.prompt_dim = nn.Sequential( nn.Dropout(0.4) , nn.Linear(512,512) )

                self.img_lowdim_trf2 = nn.Linear(512,512)
                self.img_lowdim_trf2 = nn.Linear(512,512)


            def forward(self,x):
        
                return x #This is dummy forward.

        low_dimer = LowDimer()


        '''
        if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()
            low_dimer.float()
        else :
            low_dimer.half()
        '''
               
        # added, 해당 seed, dataset, shot에 해당하는 embedding이 뽑혀있지 않다면 뽑고, 아니면 불러온다.
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, cfg.DATASET.NAME.lower())
        preprocessed = os.path.join(self.dataset_dir, "preprocessed.pkl")
        preprocessed_emb = os.path.join(self.dataset_dir, "preprocessed_emb.pkl")
        if os.path.exists(preprocessed_emb):
            # 만약 seed, dataset, shot에 해당하는 임베딩 파일이 존재한다면
            # 이걸 load하고
            pass       
        else:
        # 아니면 파일을 뽑아 pkl형태로 저장한다.
        # 그런데 해당 seed, dataset, shot에 해당하는 raw_data의 피클 파일이 존재한다면
            if os.path.exists(preprocessed):
                print(f"Loading preprocessed few-shot data from {preprocessed}")
                with open(preprocessed, "rb") as file:
                    data = pickle.load(file)
                    train = data["train"]
                # pdb.set_trace()

        low_dimer.to(self.device)
        clip_model.to(self.device)

        self.img_lowdim_trf = low_dimer.img_lowdim_trf
        self.txt_lowdim_trf = low_dimer.txt_lowdim_trf
        self.prompt_lowdim = low_dimer.prompt_dim
        self.img_lowdim_trf2 = low_dimer.img_lowdim_trf2
        
        self.conceptnet_sentences = torch.load("/mlainas/yewon/KGPrompt_data/imagenet/conceptnet_features.pkl")
        self.optim = build_optimizer(low_dimer, cfg.OPTIM )
        #import pdb; pdb.set_trace()
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None
 
        temp = CUSTOM_TEMPLATES[cfg.DATASET.NAME]

        self.register_model("low_dimer", low_dimer, self.optim, self.sched)
        
        # added
        self.df = df
        self.mode = cfg.MODE
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
            image_features =  self.img_lowdim_trf(image)
            text_features = self.txt_lowdim_trf(text)
            prompt_fc = self.prompt_lowdim(text)


        M = image_features @ text_features.view(-1,512).T #830000x1000
        M = M.view(-1,1000, max_len) #Nx1000x838
        M += mask.unsqueeze(0)
        
        if mode == 'gumbel':
            M = F.gumbel_softmax(M, tau=1.0, hard=True)
        else:
            M = F.softmax(M, dim=-1)  # Nx1000x838    

        M = torch.bmm(M.permute((1,0,2)), prompt_fc) # Nx1000x512
        M = M.permute((1,2,0)) # Nx512x1000
        M = M / M.norm(dim=1, keepdim=True)
        sims = torch.einsum('ij,ijk->ik', image, M)
        return sims


    def model_inference(self, image_features):
        
        logit_scale = self.clip_model.logit_scale.exp() 
        image_features_norm = image_features / image_features.norm(dim=-1, keepdim=True)

        if self.mode == 'weight_sum':         
            mask = self.mask_sequence
            max = self.max_num
            m = self.mode   
            text_features_norm = self.conceptnet_sentences / self.conceptnet_sentences.norm(dim=-1, keepdim=True) #normalize? or not
            sims = self.attention_parallel(image_features_norm,  self.conceptnet_sentences, mask, max, m)
            logits = logit_scale*sims
            return logits

        elif self.mode == 'attention' :
            
            mask = self.mask_sequence
            max = self.max_num
            m = self.mode   
            sims = self.attention_parallel(image_features_norm, self.conceptnet_sentences, mask, max, m )
            logits = logit_scale*sims
            
            return logits
        
        elif self.mode == 'gumbel' :
            # nn.linear에 각 prompt들을 다 forward시켜서, lower dimension embedding을 구해야함. 
            # 근데 이거 offline으로 해두면, 학습이 되는게 아니라서 무조건 여기서 해야함.        
            mask = self.mask_sequence
            max = self.max_num
            m = self.mode   
            text_features_norm = self.conceptnet_sentences / self.conceptnet_sentences.norm(dim=-1, keepdim=True) #normalize? or not

            sims = self.attention_parallel(image_features_norm, self.conceptnet_sentences, mask, max, m )
            logits = logit_scale*sims
            
            return logits

        
        else:
            logits = image_features_norm @ self.text_features.T
        # added
        
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
        wandb.init(project="KGPrompt-221118")



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
        if split == "val" and self.val_dataloader is not None:#val_loader
            data_loader = self.val_dataloader #val_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.val_dataloader   #val_loader

        print(f"Evaluate on the *{split}* set")

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)
            label = label.type(torch.int64)
            output = self.model_inference(input)
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



