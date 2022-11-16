import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
import pandas as pd

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

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
        n_ctx = cfg.TRAINER.COOP.N_CTX # ctx length
        ctx_init = cfg.TRAINER.COOP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        self.clip_model = clip_model
        df = pd.read_csv('sents.csv', encoding='utf-8')
        if torch.cuda.is_available() and cfg.USE_CUDA:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.df = df
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"
        self.fc1 = nn.Linear(512, 128)


        prompts = []
        for c in classnames:
            class_prompts = []
            class_df = df[df['word'] == c]
            for sent in class_df['sent']:
                class_prompts.append(sent)
            prompts += class_prompts       
        prompts = torch.cat([clip.tokenize(p) for p in prompts])    
        
        text_features = self.clip_model.encode_text(prompts)
        text_features  = self.fc1(text_features)

    def forward(self):
        prompts = self.prompts
        prompts.to(self.device)
        text_features = self.clip_model.encode_text(prompts)
        text_features  = self.fc1(text_features)
        return text_features


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()

        df = pd.read_csv('sents.csv', encoding='utf-8')
        if torch.cuda.is_available() and cfg.USE_CUDA:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.image_encoder = clip_model.encode_image()
        self.image_encoder = clip_model.encode_text()
       
        z_t_all = []
        for c in self.classnames:
            class_prompts = []
            class_df = self.df[self.df['word'] == c]
            for sent in class_df['sent']:
                class_prompts.append(clip.tokenize(sent))
            class_prompts = torch.cat(class_prompts).to(self.device)
            class_text_features = self.clip_model.encode_text(class_prompts)
            z_t = self.fc1(class_text_features)
            z_t_norm = z_t / z_t.norm(dim=-1, keepdim=True)
            idx = torch.argmax(torch.norm(z_i_norm @ z_t_norm.t(), dim=0))
            z_t_all.append(class_text_features[idx])
        
        text_features = torch.stack(text_features, 0)
        logit_scale = self.clip_model.logit_scale.exp()
        logits = logit_scale * z_i @ z_t.t()   
        logits = F.gumbel_softmax(logits, tau=1, hard=False)

        prompts = []
        for c in classnames:
            class_prompts = []
            class_df = df[df['word'] == c]
            for sent in class_df['sent']:
                class_prompts.append(sent)
            prompts += class_prompts       
        self.prompts = torch.cat([clip.tokenize(p) for p in prompts])      
        
        #self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.clip_model = clip_model
        self.image_encoder = self.clip_model.encode_image()
        self.text_encoder = self.clip_model.encode_text()

        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.classnames = classnames

        self.fc1 = nn.Linear(512, 128)

    def forward(self, image):

        image_features = self.image_encoder(image)
        z_i = self.fc1(image_features)        
        z_t = self.prompt_learner()   
        image_features = z_i / z_i.norm(dim=-1, keepdim=True)
        text_features = z_t / z_t.norm(dim=-1, keepdim=True)
        scores = image_features @ text_features.t()
        max_idx = torch.argmax(scores, dim = 0)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * scores[max_idx]
        F.gumbel_softmax(logits, tau = 1, hard=False)

        return logits



@TRAINER_REGISTRY.register()
class CoOp(TrainerX):
    """Context Optimization (CoOp).

    Learning to Prompt for Vision-Language Models
    https://arxiv.org/abs/2109.01134
    """

    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self, df):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames


        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = clip_model

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        #device_count = torch.cuda.device_count()
        #if device_count > 1:
        #    print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
        #    self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        
        prec = self.cfg.TRAINER.COOP.PREC
        if prec == "amp":
            with autocast():
                output = self.model(image)
                loss = F.cross_entropy(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output = self.model(image)
            loss = F.cross_entropy(output, label)
            self.model_backward_and_update(loss)

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

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
