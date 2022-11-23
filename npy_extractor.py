import torch
import time
from PIL import Image
import numpy as np
import os
from tqdm import tqdm
from torch.utils.data import Dataset


class CustomImageDataset(Dataset):
    def __init__(self, image_feat , label):
        self.label= torch.Tensor(label)
        self.img_feat = torch.Tensor(image_feat)
        self.len = self.label.shape[0]

    def __len__(self):
        return self.len
        
    def __getitem__(self,idx):        
        return self.img_feat[idx], self.label[idx]


def dataset_to_npy(dm, stage, clip_model, device):
    dataset = dm.dataset
    if stage == 'train':
        data_x = dataset.train_x
        tfm = dm.tfm_train
    else:
        data_x = dataset.val
        tfm = dm.tfm_test

    with torch.no_grad():
        start = time.time()
        npy_list = torch.zeros(len(data_x), 513)
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

def set_data_loader(cfg, dm, device, clip_model):
    # 만약 npy 파일이 없다면
    dataset_name = cfg.DATASET.NAME.lower()
    num_shot = cfg.DATASET.NUM_SHOTS
    seed = cfg.SEED
    emb_root = f'/mlainas/KGPrompt_data/{dataset_name}'

    train_dir = f'{emb_root}/shot_{num_shot}_seed_{seed}_train.npy'
    valid_dir = f'{emb_root}/shot_{num_shot}_seed_{seed}_valid.npy'

    print(train_dir)
    print(valid_dir)

    if os.path.exists(train_dir):
        data_train = np.load(train_dir)
        image_feat_train = data_train[:, :-1]
        label_train = data_train[:,-1:]

    else:
        # 해당 데이터셋의 shot, seed에 피쳐가 없다면
        # TODO: 어떻게 임베딩 뽑는지 파악 후, 코드 작성
        dataset = dm.dataset
        stage = 'train'    
        data_train = dataset_to_npy(dm=dm,
                                    stage=stage, 
                                    clip_model=clip_model,
                                    device=device)
        np.save(train_dir, data_train)

    image_feat_train = data_train[:, :-1]
    label_train = data_train[:,-1:]
    
    train_data = CustomImageDataset(image_feat_train, label_train)

    if os.path.exists(valid_dir):
        data_val = np.load(valid_dir)

    else:
        # 해당 데이터셋의 shot, seed에 피쳐가 없다면
        # TODO: 어떻게 임베딩 뽑는지 파악 후, 코드 작성
        pass
        dataset = dm.dataset
        stage = 'valid'    
        data_val = dataset_to_npy(dm=dm,
                                    stage=stage, 
                                    clip_model=clip_model,
                                    device=device)
        np.save(valid_dir, data_val)

    image_feat_val = data_val[:, :-1]
    label_valid = data_val[:,-1:]

    valid_data = CustomImageDataset(image_feat_val, label_valid)
  
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    valid_dataloader = torch.utils.data.DataLoader(valid_data, batch_size=100, shuffle=True)

    return train_dataloader, valid_dataloader