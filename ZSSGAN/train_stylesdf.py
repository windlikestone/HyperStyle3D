import argparse
import os
import numpy as np

import torch
from torchvision import utils
import random

from tqdm import tqdm

from model_stylesdf.ZSSGAN import ZSSGAN

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

SAVE_SRC = True
SAVE_DST = True

def train(args, opt):

    
    # style_list = ["pixar", "disney", "sketch", "painting", "Anime", "caricature"," Chibi Drawings", \
    #                "Minimalist Cartoon Art Styles", "Classic Disney", "Dragon ball", "Hayao Miyazaki Art Style", \
    #                "3D Animation", "Typography Animation", " Flipbook Animation", "Renaissance", \
    #                "Romanticism", "Neoclassicism", "Academic art", "Ancient art", "Japonism", "Baroque", \
    #                "impressionism", "Expressionism"]
    style_list = ["pixar", "disney", "sketch", "painting", "Anime", "caricature"]
    # style_list = ["young"]
    print("style_list:", style_list)

    print("Initializing networks...")
    # opt = args
    # generator = Generator(opt.model, opt.rendering, full_pipeline=False).to(device)
    net = ZSSGAN(args, opt)
    # print("net.generator_trainable", net.generator_trainable)
    # print("net.generator_trainable", net.generator_trainable.generator)

    g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1) # using original SG2 params. Not currently using r1 regularization, may need to change.

    g_optim = torch.optim.Adam(
        net.generator_trainable.parameters(),
        lr=args.lr * g_reg_ratio,
        betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
    )

    # g_optim = torch.optim.Adam([
    #     {'params':net.generator_trainable.generator.renderer.parameters(),'lr':args.lr,'betas':(0, 0.9)},
    #     {'params':net.generator_trainable.generator.decoder.parameters(),'lr':args.lr * g_reg_ratio,'betas':(0 ** g_reg_ratio, 0.99 ** g_reg_ratio)}
    # ])

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


    for i in tqdm(range(args.iter)):

        if i % args.output_interval == 0:
            net.eval()

            with torch.no_grad():
                # fixed_style_idx = random.randint(0,22)
                for fixed_target_class in style_list:
                    # fixed_target_class = style_list[fixed_style_idx]
                    [sampled_src, sampled_dst], loss = net(fixed_z, fixed_cam_extrinsics, fixed_focal, fixed_near, fixed_far, fixed_target_class, input_is_latent)

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
        random_style_idx = random.randint(0,2)
        random_style_idx = i % 2
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
        [sampled_src, sampled_dst], loss = net(sample_z, cam_extrinsics, focal, near, far, sample_target_class, input_is_latent)

        net.zero_grad()
        loss.backward()

        g_optim.step()

        tqdm.write(f"Clip loss: {loss}")

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
    shutil.copy2("model_stylesdf/hypernetworks/hyperlayers.py", os.path.join(args.training.output_dir, "code", "hyperlayers.py"))
    shutil.copy2("model_stylesdf/volume_renderer.py", os.path.join(args.training.output_dir, "code", "volume_renderer.py"))

    with open(os.path.join(args.training.output_dir, "args.json"), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    train(args.training, args)
    