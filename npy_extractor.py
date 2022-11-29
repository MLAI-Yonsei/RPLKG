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
    subsample_class = cfg.DATASET.SUBSAMPLE_CLASSES
    emb_root = f'/mlainas/KGPrompt_data/{dataset_name}'
    if not os.path.exists(emb_root):
        os.mkdir(emb_root)

    train_dir = f'{emb_root}/shot_{num_shot}_seed_{seed}_{subsample_class}_train.npy'
    valid_dir = f'{emb_root}/shot_{num_shot}_seed_{seed}_{subsample_class}_valid.npy'
    test_dir = f'{emb_root}/shot_{num_shot}_seed_{seed}_{subsample_class}_test.npy'

    print(train_dir)
    print(valid_dir)
    print(test_dir)

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
        stage = 'val'    
        data_val = dataset_to_npy(dm=dm,
                                    stage=stage, 
                                    clip_model=clip_model,
                                    device=device)
        np.save(valid_dir, data_val)

    image_feat_val = data_val[:, :-1]
    label_valid = data_val[:,-1:]

    valid_data = CustomImageDataset(image_feat_val, label_valid)
    
    if dataset_name == 'imagenet':
        # if datasetname == 'imagenet', testloader == valloader
        image_feat_test = image_feat_val
        label_tes = label_valid
    else:
        if os.path.exists(test_dir):
            data_test = np.load(test_dir)
            image_feat_test = data_test[:, :-1]
            label_test = data_test[:,-1:]

        else:
            # 해당 데이터셋의 shot, seed에 피쳐가 없다면
            # TODO: 어떻게 임베딩 뽑는지 파악 후, 코드 작성
            dataset = dm.dataset
            stage = 'test'    
            data_test = dataset_to_npy(dm=dm,
                                        stage=stage, 
                                        clip_model=clip_model,
                                        device=device)
            np.save(test_dir, data_test)

        image_feat_test = data_test[:, :-1]
        label_test = data_test[:,-1:]
        
    test_data = CustomImageDataset(image_feat_test, label_test)
    import pdb;pdb.set_trace()

    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    valid_dataloader = torch.utils.data.DataLoader(valid_data, batch_size=100, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=100, shuffle=True)
    
    return train_dataloader, valid_dataloader, test_dataloader 