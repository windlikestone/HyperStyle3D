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
from warping.controller_warp import WarpController 

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
    initail_image_path = "/home/chenzhuo/Downloads/point/trainB/18/C00001_000_43.jpg"
    initail_ldks_path = "/home/chenzhuo/workspace/3DAnimationGAN/ZSSGAN/warping/3D_ldks.txt"
    target_ldks_path = "/home/chenzhuo/workspace/3DAnimationGAN/ZSSGAN/warping/3D_deformed_ldks.txt"
    save_path = "text/ldks_warp/warped_img_3D.jpg"
    ori_path = "text/ldks_warp/ori_img_3D.jpg"

    preprocess = transforms.Compose([transforms.ToTensor(), transforms.Resize((512, 512), interpolation=0)])
    trainable_img = preprocess(Image.open(initail_image_path)).unsqueeze(0)
    torchvision.utils.save_image(trainable_img, ori_path)

    ldks_src_3d = torch.from_numpy(np.loadtxt(initail_ldks_path)).unsqueeze(0).repeat(2, 1, 1).float()
    ldks_tgt_3d = torch.from_numpy(np.loadtxt(target_ldks_path)).unsqueeze(0).repeat(2, 1, 1).float()
    print("ldks_tgt_3d", ldks_src_3d.shape)

    # 3D ldks warping
    trainable_img = torch.load("/home/chenzhuo/workspace/3DAnimationGAN/ZSSGAN/warping/rgb.pt")
    print("trainable_img", trainable_img.shape)
    trainable_pts = torch.load("/home/chenzhuo/workspace/3DAnimationGAN/ZSSGAN/warping/input_pts.pt")

    # ldks_src_2d = torch.from_numpy(np.loadtxt(initail_ldks_path)).long()
    # ldks_tgt_2d = torch.from_numpy(np.loadtxt(target_ldks_path)).long()

    # ldks_src_3d = trainable_pts[:, ldks_src_2d[:, 1], ldks_src_2d[:, 0], 6, :] # 2D ldks is oppsite to 3D ldks in X and Y dimension
    # ldks_tgt_3d = trainable_pts[:, ldks_tgt_2d[:, 1], ldks_tgt_2d[:, 0], 6, :] # 6 represents the middle point of a camera ray.


    os.makedirs(os.path.join("text", "ldks_warp"), exist_ok=True)

    warper = WarpController(5, device)
    warped_image = warper(trainable_img, ldks_src_3d, ldks_tgt_3d)
    

    torchvision.utils.save_image(warped_image, save_path)
