import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
import torchvision.transforms.functional as TF

import json
import os


class AnimeFaceDataset(Dataset):
    def __init__(self, set_type='train', resize=128, dataset_root='data/animefacedataset/'):
        self.dataset_root = dataset_root
        self.resize = resize
        if set_type == 'train':
            with open('data/train-index.json', 'r') as f:
                self.index_data = json.load(f)
        elif set_type == 'val':
            with open('data/val-index.json', 'r') as f:
                self.index_data = json.load(f)
        else:
            raise ValueError('set_type must be either "train" or "val"')

    def __getitem__(self, index):
        file_path = os.path.join(self.dataset_root, 'images', self.index_data[index]['file'])
        img = read_image(file_path)
        img = (img - 127.5) / 127.5
        img = TF.resize(img, [self.resize, self.resize], antialias=True)
        return img

    def __len__(self):
        return len(self.index_data)
