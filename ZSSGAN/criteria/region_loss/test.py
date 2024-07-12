#!/usr/bin/python
# -*- encoding: utf-8 -*-

from logger import setup_logger
from model import BiSeNet

import torch

import os
import os.path as osp
import torchvision.transforms as transforms
import cv2

def region_mask(im, parsing_anno, stride, index=17):

    # im = np.array(im)
    # vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(torch.uint8)
    print("vis_parsing_anno", vis_parsing_anno)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)

    zeros = torch.zeros_like(im)
    ones = torch.ones_like(im)
    hair_mask = torch.where(vis_parsing_anno == index, zeros, ones) # 17 -> hair

    return hair_mask

def face_segmentation(frozen_img, trainable_img):

    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    save_pth = osp.join()
    net.load_state_dict(torch.load(save_pth))
    net.eval()

    to_tensor = transforms.Compose([
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    f_img = to_tensor(frozen_img)
    t_img = to_tensor(trainable_img)

    print("f_img.shape", f_img.shape)
    f_out = net(f_img)[0]
    t_out = net(f_img)[0]
    f_parsing = f_out.argmax(1)
    t_parsing = t_out.argmax(1)
    # print(parsing)

    f_mask = region_mask(f_img, f_parsing, stride=1)
    t_mask = region_mask(t_img, t_parsing, stride=1)

    f_img = f_img * f_mask
    t_img = t_img * t_mask

    return f_img, f_img


def region_loss(self, src_img: torch.Tensor, target_img: torch.Tensor) -> torch.Tensor:
    return self.l2_loss(src_img, target_img)


if __name__ == "__main__":
    evaluate(dspth='/mnt/lustre/chenzhuo.vendor/workspace/StyleGAN-nada/face-parsing.PyTorch/test_img/pixar', cp='79999_iter.pth')


