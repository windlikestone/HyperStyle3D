import numpy as np
import torch
import torch.nn as nn

import math

from torchvision import models 
import torch.nn.functional as F

import torch.optim as optim

import PIL.Image

from collections import OrderedDict

from torchvision.utils import save_image
from torch_utils import misc
from torch_utils.ops import conv2d_gradfix

from torchvision import transforms

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


class IDLoss(nn.Module):
    def __init__(self, ckpt_path):
        super(IDLoss, self).__init__()
        from ZSSGAN.criteria.id_loss.model_irse import Backbone

        print('Loading ResNet ArcFace')
        self.facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')
        self.facenet.load_state_dict(torch.load(ckpt_path))
        self.pool = torch.nn.AdaptiveAvgPool2d((256, 256))
        self.face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))
        self.facenet.eval()
        self.facenet.cuda()

    def extract_feats(self, x):
        if x.shape[2] != 256:
            x = self.pool(x)
        x = x[:, :, 35:223, 32:220]  # Crop interesting region
        x = self.face_pool(x)
        x_feats = self.facenet(x)
        return x_feats

    def resize(self, x):
        resize_func = transforms.Resize([256, 256])
        x_resize = resize_func(x)
        return x_resize

    def forward(self, y_hat, y):
        # print("before resize",y_hat.shape, y.shape)
        # y_hat = self.resize(y_hat)
        # y = self.resize(y)
        # print("after resize", y_hat.shape, y.shape)
        n_samples = y.shape[0]
        y_feats = self.extract_feats(y)  # Otherwise use the feature from there
        y_hat_feats = self.extract_feats(y_hat)
        y_feats = y_feats.detach()
        loss = 0
        sim_improvement = 0
        count = 0
        for i in range(n_samples):
            diff_target = y_hat_feats[i].dot(y_feats[i])
            loss += 1 - diff_target
            count += 1

        return loss / count, sim_improvement / count