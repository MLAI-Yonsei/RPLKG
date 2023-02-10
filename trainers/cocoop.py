import os.path as osp
from collections import OrderedDict
import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

from .zsclip import ZeroshotCLIP, clip_classifier
from feature_extractor import set_data_loader, get_conceptnet_feature
import wandb
import os

_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.COCOOP.N_CTX
        ctx_init = cfg.TRAINER.COCOOP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        vis_dim = clip_model.visual.output_dim
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]

        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)

        self.meta_net = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(vis_dim, vis_dim // 16)),
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(vis_dim // 16, ctx_dim))
        ]))
        
        #if cfg.TRAINER.COCOOP.PREC == "fp16":
        #    self.meta_net.half()

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
    
    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,     # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self, im_features):
        prefix = self.token_prefix
        suffix = self.token_suffix
        ctx = self.ctx                     # (n_ctx, ctx_dim)
        bias = self.meta_net(im_features)  # (batch, ctx_dim)
        bias = bias.unsqueeze(1)           # (batch, 1, ctx_dim)
        ctx = ctx.unsqueeze(0)             # (1, n_ctx, ctx_dim)
        ctx_shifted = ctx + bias           # (batch, n_ctx, ctx_dim)
        
        # Use instance-conditioned context tokens for all classes
        prompts = []
        for ctx_shifted_i in ctx_shifted:
            ctx_i = ctx_shifted_i.unsqueeze(0).expand(self.n_cls, -1, -1)
            pts_i = self.construct_prompts(ctx_i, prefix, suffix)  # (n_cls, n_tkn, ctx_dim)
            prompts.append(pts_i)
        prompts = torch.stack(prompts)
        
        return prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.low_dimer = LowDimer(cfg, classnames, clip_model)
        #self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.prompt_learner = self.low_dimer.prompt_learner 
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype


    def forward(self, image, label=None):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        #image_features = self.image_encoder(image)
        #image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        image_features = image / image.norm(dim=-1, keepdim=True)
        prompts = self.prompt_learner(image_features)
        
        logits = []
        for pts_i, imf_i in zip(prompts, image_features):
            text_features = self.text_encoder(pts_i, tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            l_i = logit_scale * imf_i @ text_features.t()
            logits.append(l_i)
        logits = torch.stack(logits)
        #if self.prompt_learner.training:
         #   return F.cross_entropy(logits, label)
        
        return logits

class LowDimer(nn.Module):
        def __init__(self,cfg,classnames,clip_model):
            super().__init__()
            self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
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

@TRAINER_REGISTRY.register()
class CoCoOp(ZeroshotCLIP):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.COCOOP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
      
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        #if cfg.TRAINER.COCOOP.PREC == "fp32" or cfg.TRAINER.COCOOP.PREC == "amp":
            # CLIP's default precision is fp16
        clip_model.float()
        
        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        
        self.train_dataloader, self.valid_dataloader, self.test_dataloader = set_data_loader(cfg, self.dm, self.device, clip_model)      
        self.input_dim = 1024 if cfg.MODEL.BACKBONE.NAME.lower() == 'rn50' else 512
        # sentence embedding
        self.subsample_class = cfg.DATASET.SUBSAMPLE_CLASSES
        self.search_level = cfg.DATASET.SEARCH_LEVEL
        self.emb_root = cfg.DATASET.EMB_ROOT
        self.dataset_name = cfg.DATASET.NAME.lower()
        bk=cfg.MODEL.BACKBONE.NAME
        self.logit_scale = cfg.TRAINER.MY_MODEL.SCALE
        self.dropout = cfg.TRAINER.MY_MODEL.DROPOUT
        self.wd = cfg.OPTIM.WEIGHT_DECAY
        self.mode = cfg.MODE
        self.alpha = cfg.TRAINER.MY_MODEL.ALPHA
        num_shot = cfg.DATASET.NUM_SHOTS
        self.name = f'{self.dataset_name}_dropout={self.dropout}_wd={self.wd}_shot{num_shot}_{cfg.MODE}_alpha{self.alpha}_level{self.search_level}_seed{cfg.SEED}_{bk}'
        self.classnames = classnames
        
        #wandb.init(project="KGPrompt_230210",
        #    name = self.name,
        #    entity="ingdoo") 


        self.low_dimer = self.model.low_dimer

        self.low_dimer.to(self.device)
        clip_model.to(self.device)

        self.clip_model = clip_model

        self.img_lowdim_trf = self.low_dimer.img_lowdim_trf
        self.txt_lowdim_trf = self.low_dimer.txt_lowdim_trf
        self.prompt_lowdim = self.low_dimer.prompt_dim 
       
        # automated conceptnet_feature extracting
        conceptnet_sentences_path = f"{self.emb_root}/{self.dataset_name}/conceptnet_features_{cfg.MODEL.BACKBONE.NAME.lower().replace('/', '')}_{self.subsample_class}_level_{self.search_level}.pkl"
        if not os.path.exists(conceptnet_sentences_path):
            self.conceptnet_sentences = get_conceptnet_feature(emb_root=self.emb_root,
                                                             dataset=self.dataset_name,
                                                             backbone=cfg.MODEL.BACKBONE.NAME,
                                                             subsample_class=self.subsample_class,
                                                             level=self.search_level,
                                                             classnames=classnames,
                                                             clip_model=clip_model,
                                                             device=self.device)
        else:
            self.conceptnet_sentences = torch.load(conceptnet_sentences_path)
        
        #for clip weight        
        # added
        conceptnet_sentences = self.conceptnet_sentences
        self.conceptnet_prompt = self.conceptnet_sentences 
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

        print("Turning off gradients in both the image and the text encoder")
        name_to_update = "low_dimer"
        
        for name, param in self.model.named_parameters():
            if name_to_update not in name:
                param.requires_grad_(False)
        
        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.low_dimer, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.low_dimer, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("low_dimer", self.model.low_dimer, self.optim, self.sched)
        self.scaler = GradScaler() if cfg.TRAINER.COCOOP.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        #device_count = torch.cuda.device_count()
        #if device_count > 1:
        #    print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
        #    self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        model = self.model
        optim = self.optim
        scaler = self.scaler
        
        prec = self.cfg.TRAINER.COCOOP.PREC
        if prec == "amp":
            with autocast():
                output = self.model_inference(image) + model(image, label.squeeze(dim=-1))
                loss = F.cross_entropy(output, label)
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            output = self.model_inference(image) +  model(image, label.squeeze(dim=-1))
            loss = F.cross_entropy(output, label.squeeze(dim=-1).long())
            self.model_backward_and_update(loss)
            #loss = model(image, label)
            #optim.zero_grad()
            #loss.backward()
            #optim.step()

        loss_summary = {"loss": loss.item()}

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary


    def attention_parallel(self, image, text, mask, max_len:int):
        image_features =  self.img_lowdim_trf(image)
        text_features = self.txt_lowdim_trf(text)
        text = self.prompt_lowdim(text)
        clip_weights = clip_classifier(self.conceptnet_prompt)
        clip_weights = clip_weights.float()
        img_w = image @ clip_weights

        M = image_features @ text_features.view(-1,self.input_dim).T #830000x1000
        M = M.view(-1,len(self.classnames), max_len) #Nx1000x838
        M += mask.unsqueeze(0)
        
        M1 = F.gumbel_softmax(M, tau=self.temp, hard=True)
        M2 = M - M1.detach()
        M2 = F.gumbel_softmax(M2, tau=self.temp, hard=True)
        M = M1 + M2


        M = torch.bmm(M.permute((1,0,2)), text) # Nx1000x512
        M = M.permute((1,2,0)) # Nx512x1000
        M = M / M.norm(dim=1, keepdim=True)
        alpha = self.alpha
        sims = alpha * torch.einsum('ij,ijk->ik', image, M) + img_w #Nx1000
        ##dual softmax

        return sims


    def model_inference(self, image_features): #, split):
        if self.logit_scale == 0:
            logit_scale = self.clip_model.logit_scale.exp()
        else:
            logit_scale = self.logit_scale
        
        #if split == None:
        #    image_features = image_features + 0.2*torch.randn_like(image_features) 
        
        image_features_norm = image_features / image_features.norm(dim=-1, keepdim=True)   
    
        mask = self.mask_sequence
        max = self.max_num
        self.conceptnet_sentences = self.conceptnet_sentences.float()
        sims = self.attention_parallel(image_features_norm, self.conceptnet_sentences, mask, max)
        logits = logit_scale*sims 

        return logits       


    def model_inference_(self, image_features):
        return super().model_inference(image_features)

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)