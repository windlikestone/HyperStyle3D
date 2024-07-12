"""Datasets"""

import os

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
import torchvision.transforms as transforms
import torchvision
import glob
import PIL
import random
import math
import pickle
import numpy as np


class Styles(Dataset):
    """Styles Dataset"""

    def __init__(self, dataset_path, img_size):
        super().__init__()

        self.data = glob.glob(dataset_path)
        assert len(self.data) > 0, "Can't find data; make sure you specify the path to your dataset"
        # self.transform = transforms.Compose(
        #             [transforms.ToTensor(), transforms.Resize((img_size, img_size), interpolation=0)])

        self.transform = transforms.Compose(
                    [transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
                     transforms.CenterCrop((img_size, img_size)),
                     transforms.ToTensor(),
                     transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
                    ])


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        X = PIL.Image.open(self.data[index])
        X = X.convert("RGB")
        X = self.transform(X)

        return X, 0


def get_dataset(name='Styles', subsample=None, batch_size=1, **kwargs):
    dataset = globals()[name](**kwargs)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=False,
        num_workers=8
    )
    return dataloader, 3
