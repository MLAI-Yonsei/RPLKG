import pdb
import torch
import time
import pickle
from PIL import Image
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from torch.utils.data import Dataset

from clip import clip

class CustomImageDataset(Dataset):
    def __init__(self, image_feat , label):
        self.label= torch.Tensor(label)
        self.img_feat = torch.Tensor(image_feat)
        self.len = self.label.shape[0]

    def __len__(self):
        return self.len
        
    def __getitem__(self,idx):        
        return self.img_feat[idx], self.label[idx]


def dataset_to_npy(dm, backbone, stage, clip_model, device):
    dataset = dm.dataset
    print(stage)
    if stage == 'train':
        data_x = dataset.train_x
        tfm = dm.tfm_train
    elif stage == 'val':
        data_x = dataset.val
        tfm = dm.tfm_test
    else:
        data_x = dataset.test
        tfm = dm.tfm_test

    with torch.no_grad():
        start = time.time()
        col_len = 1025 if backbone == 'rn50' else 513
        npy_list = torch.zeros(len(data_x), col_len)
        for i, data in enumerate(tqdm(data_x)):
            img_path = data.impath
            label = [float(data.label)]
            label = torch.Tensor(label).to(device).unsqueeze(0)
            img = Image.open(img_path).convert("RGB")
            img_tensor = tfm(img).to(device)
            img_emb = clip_model.encode_image(torch.unsqueeze(img_tensor, 0))
            row = torch.cat([img_emb, label], dim=1)
            npy_list[i] = row
        npy_list = npy_list.cpu().detach().numpy()
    return npy_list

def get_dataset(dir, dm, backbone, stage, clip_model, device):
    print(dir)
    # pdb.set_trace()
    if os.path.exists(dir):
        data = np.load(dir)
        image_feat = data[:, :-1]
        label = data[:,-1:]

    else:
        # 해당 데이터셋의 shot, seed에 피쳐가 없다면
        # TODO: 어떻게 임베딩 뽑는지 파악 후, 코드 작성
        dataset = dm.dataset    
        data = dataset_to_npy(dm=dm,
                              backbone=backbone,
                              stage=stage, 
                              clip_model=clip_model,
                              device=device)
        np.save(dir, data)

    image_feat = data[:, :-1]
    label = data[:,-1:]
    
    dataset = CustomImageDataset(image_feat, label)
    return dataset 

def set_data_loader(cfg, dm, device, clip_model, test=None):
    # 만약 npy 파일이 없다면
    backbone = cfg.MODEL.BACKBONE.NAME.lower().replace('/', '')
    dataset_name = cfg.DATASET.NAME.lower()
    # pdb.set_trace()
    # if test:
    # dataset_name = 'imagenetsketch'
    # dataset_name = 'imagenetv2'
    dataset_name = 'imageneta'
    # dataset_name = 'imagenetr'
    num_shot = cfg.DATASET.NUM_SHOTS
    seed = cfg.SEED
    subsample_class = cfg.DATASET.SUBSAMPLE_CLASSES
    emb_root = f'/mlainas/KGPrompt_data/{dataset_name}'
    if not os.path.exists(emb_root):
        os.mkdir(emb_root)
    # pdb.set_trace()
    train_dir = f'{emb_root}/{backbone}_shot_{num_shot}_seed_{seed}_{subsample_class}_train.npy'
    valid_dir = f'{emb_root}/{backbone}_shot_{num_shot}_seed_{seed}_{subsample_class}_valid.npy'
    test_dir = f'{emb_root}/{backbone}_shot_{num_shot}_seed_{seed}_{subsample_class}_test.npy'

    class_names_ = None
    if dataset_name in ['imageneta', 'imagenetr']:
        classnames_path = f'{emb_root}/classnames.pkl'
        if not os.path.exists(classnames_path):
            classnames = dm.dataset.classnames
            class_names_ = classnames
            with open(classnames_path, 'wb') as fp:
                pickle.dump(class_names_, fp)
    
    # TODO: implement dataloder for imagenet vairants testsets  
    if dataset_name in ['imageneta', 'imagenetr', 'imagenetsketch', 'imagenetv2']:
        # pdb.set_trace()
        test_data  = get_dataset(test_dir, dm, backbone, 'test', clip_model, device)
        # pdb.set_trace()
        test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=100, shuffle=True)
        return test_dataloader, test_dataloader, test_dataloader, class_names_, dataset_name
    # test_dataloader = torch.utils.data.DataLoader(train_data, batch_size=100, shuffle=True)
    # return test_dataloader, test_dataloader, test_dataloader

    print(train_dir)
    print(valid_dir)
    print(test_dir)

    train_data = get_dataset(train_dir, dm, backbone, 'train', clip_model, device)
    if dataset_name in ['imageneta', 'imagenetr', 'imagenetsketch', 'imagenetv2']:
        valid_data = train_data
        test_data = train_data
    else:
        valid_data = get_dataset(valid_dir, dm, backbone, 'val', clip_model, device)
        if dataset_name == 'imagenet':
            test_data = valid_data
            # dm.impath 빼오기
            valid_dm_dataset = dm.dataset.val
            impath_list_path = f'{emb_root}/imagenet_impath_list.pkl'
            if not os.path.exists(impath_list_path):
                impath_list = []
                for data in valid_dm_dataset:
                    im_path = data.impath
                    impath_list.append(im_path)
                with open(impath_list_path, mode='wb') as f:
                    pickle.dump(impath_list, f)
        else:
            test_data  = get_dataset(test_dir, dm, backbone, 'test', clip_model, device)


    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    valid_dataloader = torch.utils.data.DataLoader(valid_data, batch_size=100, shuffle=False)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=100, shuffle=False)
    
    return train_dataloader, valid_dataloader, test_dataloader, class_names_, dataset_name

def get_conceptnet_feature(dm, emb_root, dataset, backbone, subsample_class, level, classnames, clip_model, device):
    """
    emb_root: root path of embedding
    dataset: name of dataset (ImageNet, EuroSAT, ...)
    backbone: name of backbone (vit_b16, vit_b32, rn50, rn101)
    subsample_class: subsample class policy - in base, new, all
    level: search level for conceptNet. in [0, 1, 2, 3, 4]
    classnames: classnames for dataset
    clip_model: clip model for feature extracting
    device: device 
    """
    emb_root = f'/mlainas/KGPrompt_data/{dataset}'
    if dataset in ['imageneta', 'imagenetr']:
        classnames_path = f'{emb_root}/classnames.pkl'
        with open(classnames_path, 'rb') as fp:
            classnames = pickle.load(fp)
    # search level split
    df_path = f'{emb_root}/sents.csv'
    df = pd.read_csv(df_path)
    df = df[df['level'] <= level]
    # pdb.set_trace()
    emb_list = []
    for c in classnames:
        class_df = df[df['classname'] == c]
        sents = class_df['sent']
        sent_list = []
        for sent in sents:
            sent_list.append(sent)
        if len(sent_list) == 0:
            print(c)
            continue
        tok_list = torch.stack([clip.tokenize(sent) for sent in sent_list])
        tok_list = tok_list.to(device)
        with torch.no_grad():
            if len(tok_list.shape) > 2:
                tok_list = tok_list.squeeze(1)
            class_feature = clip_model.encode_text(tok_list) 
            # pdb.set_trace()
            # class_feature = class_feature @ (clip_model.text_projection.T
        emb_list.append(class_feature)
    
    # TODO: complete save path
    # conceptnet_sentences_path = f"{self.emb_root}/{self.dataset_name}/conceptnet_features_{cfg.MODEL.BACKBONE.NAME.lower().replace('/', '')}_{self.subsample_class}_level_{self.search_level}.pkl"
    save_path = f'{emb_root}/conceptnet_features_{backbone}_{subsample_class}_level_{level}.pkl'
    torch.save(emb_list, save_path)

    return emb_list