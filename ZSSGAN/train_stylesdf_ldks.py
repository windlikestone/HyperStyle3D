'''
Train a zero-shot GAN using CLIP-based supervision.

Example commands:
    CUDA_VISIBLE_DEVICES=1 python train.py --size 1024 
                                           --batch 2 
                                           --n_sample 4 
                                           --output_dir /path/to/output/dir 
                                           --lr 0.002 
                                           --frozen_gen_ckpt /path/to/stylegan2-ffhq-config-f.pt 
                                           --iter 301 
                                           --source_class "photo" 
                                           --target_class "sketch" 
                                           --lambda_direction 1.0 
                                           --lambda_patch 0.0 
                                           --lambda_global 0.0 
                                           --lambda_texture 0.0 
                                           --lambda_manifold 0.0 
                                           --phase None 
                                           --auto_layer_k 0 
                                           --auto_layer_iters 0 
                                           --auto_layer_batch 8 
                                           --output_interval 50 
                                           --clip_models "ViT-B/32" "ViT-B/16" 
                                           --clip_model_weights 1.0 1.0 
                                           --mixing 0.0
                                           --save_interval 50
'''

import argparse
import os
import numpy as np

import torch
from torchvision import utils
import random

from tqdm import tqdm

from model_stylesdf.ZSSGAN_ldks import ZSSGAN

import shutil
import json

from utils.file_utils import copytree, save_images, save_paper_image_grid
# from utils.training_utils import mixing_noise

from options.train_options import TrainOptions
from options.stylesdf_options import BaseOptions
from model_stylesdf.utils import (
    generate_camera_params,
    align_volume,
    extract_mesh_with_marching_cubes,
    xyz2mesh,
    make_noise,
    mixing_noise
)

#TODO convert these to proper args
SAVE_SRC = True
SAVE_DST = True

def load_init_params(filename):
    with open(f"/mnt/lustre/chenzhuo.vendor/workspace/StyleGAN-nada/ZSSGAN/result_photo_toyuhao/{filename}.json",'r',encoding='utf-8') as f:
        initial_dict = json.load(f)
    return initial_dict

def train(args, opt):

    
    # style_list = ["pixar", "disney", "sketch", "painting", "Anime", "caricature"," Chibi Drawings", \
    #                "Minimalist Cartoon Art Styles", "Classic Disney", "Dragon ball", "Hayao Miyazaki Art Style", \
    #                "3D Animation", "Typography Animation", " Flipbook Animation", "Renaissance", \
    #                "Romanticism", "Neoclassicism", "Academic art", "Ancient art", "Japonism", "Baroque", \
    #                "impressionism", "Expressionism"]
    # style_list = ["pixar", "disney", "sketch", "painting", "Anime", "caricature"]
    style_list = ["fat", "Fat", "Longer_Face", "Shorter_Face", "Bigger_Mouth", "Bigger_Chin", "Bigger_Forehead"]
    print("style_list:", style_list)

    print("Initializing networks...")
    
    net = ZSSGAN(args, opt)

    initail_ldks_path = "/home/chenzhuo/workspace/3DAnimationGAN/ZSSGAN/warping/3D_ldks.txt"
        # self.target_ldks_path  = "/home/chenzhuo/workspace/3DAnimationGAN/ZSSGAN/warping/3D_deformed_ldks.txt"
    source_ldks = torch.from_numpy(np.loadtxt(initail_ldks_path)).to(device)
    # self.target_ldks = torch.from_numpy(np.loadtxt(self.target_ldks_path))

    initial_target_ldks = source_ldks

    target_ldks = torch.Tensor(initial_target_ldks.cpu().detach().numpy()).to(device)

    target_ldks.requires_grad = True

    w_optim = torch.optim.SGD([target_ldks], lr=0.01)
   
    # Set up output directories.
    sample_dir = os.path.join(args.output_dir, "sample")
    ckpt_dir   = os.path.join(args.output_dir, "checkpoint")

    os.makedirs(sample_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    # set seed after all networks have been initialized. Avoids change of outputs due to model changes.
    torch.manual_seed(20)
    np.random.seed(20)

    # Training loop
    # fixed_z = torch.randn(args.n_sample, 256, device=device)
    # init_params = load_init_params('00002_params')
    # fixed_z = [torch.tensor(init_params["latent_z"], device=device).repeat(args.n_sample, 1)]
    input_is_latent = False
    fixed_z = mixing_noise(args.n_sample, args.style_dim, args.mixing, device)
    fixed_cam_extrinsics, fixed_focal, fixed_near, fixed_far, fixed_gt_viewpoints = generate_camera_params(args.renderer_output_size, device, batch=args.n_sample,
                                                                                            uniform=args.camera.uniform, azim_range=args.camera.azim,
                                                                                            elev_range=args.camera.elev, fov_ang=args.camera.fov,
                                                                                            dist_radius=args.camera.dist_radius)

    # print("fixed_z", fixed_z.shape)

    for i in tqdm(range(args.iter)):

        if i % args.output_interval == 0:
            net.eval()

            with torch.no_grad():
                # fixed_style_idx = random.randint(0,22)
                for fixed_target_class in style_list:
                    # fixed_target_class = style_list[fixed_style_idx]
                    [sampled_src, sampled_dst], loss = net(fixed_z, fixed_cam_extrinsics, fixed_focal, fixed_near, fixed_far, fixed_target_class, input_is_latent=input_is_latent, target_ldks=target_ldks)

                    # if args.crop_for_cars:
                    #     sampled_dst = sampled_dst[:, :, 64:448, :]

                    grid_rows = int(args.n_sample ** 0.5)

                    if SAVE_SRC:
                        save_images(sampled_src, sample_dir, "src", grid_rows, i)

                    if SAVE_DST:
                        save_images(sampled_dst, sample_dir, "dst", grid_rows, i)
                        utils.save_image(   sampled_dst,
                                            os.path.join(sample_dir, f"{fixed_target_class}_{str(i).zfill(6)}.jpg"),
                                            nrow=grid_rows,
                                            normalize=True,
                                            range=(-1, 1)
                                        )

        if (args.save_interval is not None) and (i > 0) and (i % args.save_interval == 0):
            torch.save(
                {
                    "g_ema": net.generator_trainable.generator.state_dict(),
                    "g_optim": g_optim.state_dict(),
                },
                f"{ckpt_dir}/{str(i).zfill(6)}.pt",
            )

        net.train()

        # sample_z = mixing_noise(args.batch, 256, args.mixing, device)
        # sample_z = torch.randn(args.batch, 256, device=device)
        sample_z = mixing_noise(args.batch, args.style_dim, args.mixing, device)
        
        # hyper_style
        random_style_idx = random.randint(0,1)
        sample_target_class = style_list[0]
        # print("target class", sample_target_class)
        # hyper_style_end

        # sample_z = [torch.tensor(init_params["latent_z"], device=device).repeat(args.batch, 1)]
        # input_is_latent = True

        cam_extrinsics, focal, near, far, gt_viewpoints = generate_camera_params(args.renderer_output_size, device, batch=args.batch,
                                                                            uniform=args.camera.uniform, azim_range=args.camera.azim,
                                                                            elev_range=args.camera.elev, fov_ang=args.camera.fov,
                                                                            dist_radius=args.camera.dist_radius)



        # [sampled_src, sampled_dst], loss = net(sample_z, **metadata)
        [sampled_src, sampled_dst], loss = net(sample_z, cam_extrinsics, focal, near, far, sample_target_class, input_is_latent=input_is_latent, target_ldks=target_ldks)

        w_optim.zero_grad()
        loss.backward()
        w_optim.step()

        tqdm.write(f"Clip loss: {loss}")
        tqdm.write(f"target_ldk: {target_ldks}")

    for i in range(args.num_grid_outputs):
        net.eval()

        with torch.no_grad():
            sample_z = mixing_noise(16, 512, 0, device)
            [sampled_src, sampled_dst], _ = net(sample_z, truncation=args.sample_truncation)

            if args.crop_for_cars:
                sampled_dst = sampled_dst[:, :, 64:448, :]

        save_paper_image_grid(sampled_dst, sample_dir, f"sampled_grid_{i}.png")
            

if __name__ == "__main__":
    device = "cuda"

    # args = TrainOptions().parse()
    args = BaseOptions().parse()
    args.training.camera = args.camera
    # args.training.size = args.model.size
    args.training.renderer_output_size = args.model.renderer_spatial_output_dim
    args.training.style_dim = args.model.style_dim
    args.model.freeze_renderer = False
    args.training.channel_multiplier = args.model.channel_multiplier
    print(args)

    # save snapshot of code / args before training.
    os.makedirs(os.path.join(args.training.output_dir, "code"), exist_ok=True)
    copytree("criteria/", os.path.join(args.training.output_dir, "code", "criteria"), )
    shutil.copy2("model_stylesdf/ZSSGAN.py", os.path.join(args.training.output_dir, "code", "ZSSGAN.py"))
    
    with open(os.path.join(args.training.output_dir, "args.json"), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    train(args.training, args)
    