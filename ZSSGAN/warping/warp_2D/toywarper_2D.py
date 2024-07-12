import sys
import os
sys.path.insert(0, os.path.abspath('../'))


import torch
import torchvision.transforms as transforms
import torchvision

import numpy as np
import copy
from PIL import Image

from functools import partial
from warp_2D.controller_warp_2D import WarpController 

# class ZSSGAN(torch.nn.Module):
#     def __init__(self):
#         super(ZSSGAN, self).__init__()

#         self.device = 'cuda:0'
#         # geometry warping controller
#         self.controller_warp = WarpController(68, self.device)

#     def forward(self, img, ldks_src, ldks_tgt):

#     	trainable_img = self.controller_warp(trainable_img, ldks_src, ldks_tgt)


if __name__ == "__main__":
    device = "cuda"

    # save snapshot of code / args before training.
    initial_image_path = "/home/chenzhuo/workspace/3DAnimationGAN/ZSSGAN/results/result_small_3d_10/000150/final_model/random_angles/images/0000007_thumb.png"
    initial_ldks_path = "/home/chenzhuo/workspace/3DAnimationGAN/ZSSGAN/warping/C00001_000_43.txt"
    target_ldks_path = "/home/chenzhuo/workspace/3DAnimationGAN/ZSSGAN/warping/C00001_000_43_deformation.txt"
    save_path = "/home/chenzhuo/workspace/3DAnimationGAN/ZSSGAN/warping/text/ldks_warp/warped_img.jpg"
    ori_path = "/home/chenzhuo/workspace/3DAnimationGAN/ZSSGAN/warping/text/ldks_warp/ori_img.jpg"

    preprocess = transforms.Compose([transforms.ToTensor(), transforms.Resize((64, 64), interpolation=0)])
    trainable_img = preprocess(Image.open(initial_image_path))

    torchvision.utils.save_image(trainable_img, ori_path)
    trainable_img = trainable_img.permute(1, 2, 0).unsqueeze(0)

    ldks_src = torch.from_numpy(np.loadtxt(initial_ldks_path)).unsqueeze(0).float()
    ldks_tgt = torch.from_numpy(np.loadtxt(target_ldks_path)).unsqueeze(0).float()


    os.makedirs(os.path.join("text", "ldks_warp"), exist_ok=True)

    warper = WarpController(5, device)

    warped_image = warper(trainable_img, ldks_src, ldks_tgt)
    
    warped_image = warped_image.permute(0, 3, 1, 2)
    torchvision.utils.save_image(warped_image, save_path)
