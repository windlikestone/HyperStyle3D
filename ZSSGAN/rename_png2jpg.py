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
from torchvision import utils


class Styles(Dataset):
    """Styles Dataset"""

    def __init__(self, dataset_path, img_size):
        super().__init__()

        self.data = glob.glob(dataset_path)
        assert len(self.data) > 0, "Can't find data; make sure you specify the path to your dataset"
        self.transform = transforms.Compose(
                    [transforms.ToTensor(), transforms.Resize((img_size, img_size), interpolation=0)])

    def len(self):
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

if __name__ == "__main__":

	datasettings = {'dataset_path': '/home/chenzhuo/workspace/3DAnimationGAN/ZSSGAN/output/video/styles_dataset/*.png',
                    'img_size': 256}

	png_path = "/home/chenzhuo/workspace/3DAnimationGAN/ZSSGAN/output/video/styles_dataset"

	StylesDataset = Styles(datasettings['dataset_path'], datasettings['img_size'])
	for i in range(StylesDataset.len()):
		print(i)
		style_png, _ = StylesDataset.__getitem__(i)
		utils.save_image(style_png, os.path.join(png_path, f"style_img_{str(i).zfill(6)}.jpg"))