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


class CelebA(Dataset):
    def __init__(self, img_size, **kwargs):
        super().__init__()

        self.data = glob.glob('/mnt/lustre/chenzhuo.vendor/workspace/data/1000_images/*.jpg')
        assert len(self.data) > 0, "Can't find data; make sure you specify the path to your dataset"
        self.transform = transforms.Compose(
                    [transforms.ToTensor(), transforms.Resize((img_size, img_size), interpolation=0)])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        X = PIL.Image.open(self.data[index])
        X = self.transform(X)

        return X

Images_set = CelebA(1024)
# numbers = [num for num in range(len(Images_set))]
# list_of_sample = random.sample(numbers, 5000)

for index in range(len(list_of_sample)):
    a = Images_set.__getitem__(index)
    torchvision.utils.save_image(a, '/mnt/lustre/chenzhuo.vendor/workspace/data/1000_images_rename/%06d.jpg'%(index))
    print("%06d.jpg"%(index))