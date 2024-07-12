import os
import torch
import trimesh
import numpy as np
from munch import *
from PIL import Image
from tqdm import tqdm
from torch.nn import functional as F
from torch.utils import data

from torchvision import transforms
from skimage.measure import marching_cubes
from scipy.spatial import Delaunay
from options.stylesdf_options import BaseOptions
from model_stylesdf.model import Generator
from model_stylesdf.utils import (
    generate_camera_params,
    align_volume,
    extract_mesh_with_marching_cubes,
    xyz2mesh,
)
from torchvision import utils as utils_a
from torch import optim
import json
# from criteria.id_loss import IDLoss


torch.random.manual_seed(1234)

def load_init_params(filename):
    with open(f"/mnt/lustre/chenzhuo.vendor/workspace/StyleGAN-nada/ZSSGAN/result_photo_toyuhao/{filename}.json",'r',encoding='utf-8') as f:
        initial_dict = json.load(f)
    return initial_dict

def inverse(opt, g_ema, surface_g_ema, device, mean_latent, surface_mean_latent, orig_img):
    init_params = load_init_params('00002_params')
    g_ema.eval()
    if not opt.no_surface_renderings:
        surface_g_ema.eval()

    # init_w_code_to_3D_architecture
    # latent_code_init = mean_latent.detach().clone().repeat(1, 8, 1)
    # latent_code_init = mean_latent[0].detach().clone()
    latent_code_init = torch.tensor(init_params["latent_z"], device=device)
    print(latent_code_init.shape)
    # set camera angles
    front_viewdir = True
    if front_viewdir:
        # These can be changed to any other specific viewpoints.
        # You can add or remove viewpoints as you wish
        locations = torch.tensor([init_params["location"]], device=device)
        # For zooming in/out change the values of fov
        # (This can be defined for each view separately via a custom tensor
        # like the locations tensor above. Tensor shape should be [locations.shape[0],1])
        # reasonable values are [0.75 * opt.camera.fov, 1.25 * opt.camera.fov]
        fov = opt.camera.fov * torch.ones((locations.shape[0],1), device=device)
        num_viewdirs = locations.shape[0]
    else:
        locations = None
        # fov = None
        fov = opt.camera.fov
        num_viewdirs = opt.num_views_per_id

    # set_grad
    latent = latent_code_init.detach().clone()
    latent.requires_grad = True

    mse_loss = torch.nn.MSELoss()
    # id_loss = IDLoss(args)

    optimizer = optim.Adam([latent], lr=0.01, eps=1e-8, betas=(0.9, 0.999))


    # generate images
    for i in tqdm(range(opt.identities)):

        # sample_cam_extrinsics, sample_focals, sample_near, sample_far, sample_locations = \
        # generate_camera_params(opt.renderer_output_size, device, batch=num_viewdirs,locations=locations,
        #                        uniform=opt.camera.uniform, azim_range=opt.camera.azim,
        #                        elev_range=opt.camera.elev, fov_ang=fov, dist_radius=opt.camera.dist_radius)
       
        # print("sample_cam_extrinsics1", sample_cam_extrinsics.shape)
        # print("sample_focals1", sample_focals.shape)
        # print("sample_near1", sample_near.shape)
        # print("sample_far1", sample_far.shape)

        sample_cam_extrinsics = torch.tensor(init_params["extrinsics"], device=device)
        sample_focals = torch.tensor(init_params["focal"], device=device)
        sample_near = torch.tensor(init_params["near"], device=device)
        sample_far = torch.tensor(init_params["far"], device=device)
        print("sample_cam_extrinsics", sample_cam_extrinsics.shape)
        print("sample_focals", sample_focals.shape)
        print("sample_near", sample_near.shape)
        print("sample_far", sample_far.shape)

        chunk = 8

        img_gen = g_ema([latent], 
                        sample_cam_extrinsics,
                        sample_focals,
                        sample_near,
                        sample_far,
                        input_is_latent=True, 
                        randomize_noise=False,
                        truncation=0.5,
                        truncation_latent=mean_latent)[0]

        # i_loss = id_loss(img_gen, img_orig)[0]
        l2_loss = mse_loss(img_gen, orig_img)
        print(img_gen.shape)
        print(orig_img.shape)
        # print(g_ema.renderer.network.sigma_layer.weight.grad)
        loss = l2_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        tqdm.write(f"Clip loss: {loss}")
        if opt.output_interval > 0 and i % opt.output_interval == 0:
            with torch.no_grad():
                rgb_images = torch.Tensor(0, 3, opt.size, opt.size)
                rgb_images = torch.cat([rgb_images, img_gen.cpu(), orig_img.cpu()], 0)
     
            utils_a.save_image(rgb_images,
                os.path.join(opt.results_dst_dir, 'images','{}.png'.format(str(i).zfill(7))),
                nrow=num_viewdirs,
                normalize=True,
                padding=0,
                range=(-1, 1),)

            # utils_a.save_image(rgb_images,
            #     os.path.join(opt.results_dst_dir, 'images','{}.png'.format(str(i).zfill(7))),
            #     nrow=num_viewdirs,
            #     normalize=True,
            #     padding=0,
            #     range=(-1, 1),)

if __name__ == "__main__":
    device = "cuda"
    opt = BaseOptions().parse()
    opt.model.is_test = True
    opt.model.freeze_renderer = False
    opt.rendering.offset_sampling = True
    opt.rendering.static_viewdirs = True
    opt.rendering.force_background = True
    opt.rendering.perturb = 0
    opt.inference.size = opt.model.model_size
    opt.inference.camera = opt.camera
    opt.inference.renderer_output_size = opt.model.renderer_spatial_output_dim
    opt.inference.style_dim = opt.model.style_dim
    opt.inference.project_noise = opt.model.project_noise
    opt.inference.return_xyz = opt.rendering.return_xyz
    opt.inference.output_interval = opt.training.output_interval
    opt.inference.lr = opt.training.lr

    
    # checkpoints_dir = '/mnt/lustre/chenzhuo.vendor/workspace/StyleGAN-nada/ZSSGAN/stylesdf_pixar_model_2_red_110/checkpoint'
    checkpoints_dir = '/mnt/lustre/chenzhuo.vendor/workspace/StyleGAN-nada/ZSSGAN/frozen_model'
    checkpoint_path = os.path.join(checkpoints_dir, opt.experiment.expname + '.pt')
    
    print("checkpoint_path", checkpoint_path)
    if os.path.isfile(checkpoint_path):
        # define results directory name
        result_model_dir = 'final_model'
    else:
        print("None ckpt!")
    # else:
    results_dir_basename = os.path.join(opt.inference.results_dir, opt.experiment.expname)
    opt.inference.results_dst_dir = os.path.join(results_dir_basename, result_model_dir)
    if opt.inference.fixed_camera_angles:
        opt.inference.results_dst_dir = os.path.join(opt.inference.results_dst_dir, 'fixed_angles')
    else:
        opt.inference.results_dst_dir = os.path.join(opt.inference.results_dst_dir, 'random_angles')
    os.makedirs(opt.inference.results_dst_dir, exist_ok=True)
    os.makedirs(os.path.join(opt.inference.results_dst_dir, 'images'), exist_ok=True)
    if not opt.inference.no_surface_renderings:
        os.makedirs(os.path.join(opt.inference.results_dst_dir, 'depth_map_meshes'), exist_ok=True)
        os.makedirs(os.path.join(opt.inference.results_dst_dir, 'marching_cubes_meshes'), exist_ok=True)

    # load saved model
    checkpoint = torch.load(checkpoint_path)

    # load image generation model
    g_ema = Generator(opt.model, opt.rendering).to(device)
    pretrained_weights_dict = checkpoint["g_ema"]
    model_dict = g_ema.state_dict()
    for k, v in pretrained_weights_dict.items():
        if v.size() == model_dict[k].size():
            model_dict[k] = v

    g_ema.load_state_dict(model_dict)

    # load a second volume renderer that extracts surfaces at 128x128x128 (or higher) for better surface resolution
    if not opt.inference.no_surface_renderings:
        opt['surf_extraction'] = Munch()
        print("opt.surf_extraction", opt.surf_extraction)

        opt.surf_extraction.rendering = opt.rendering
        opt.surf_extraction.model = opt.model
        print("opt.model", opt.model)
        print("opt.surf_extraction.model", opt.surf_extraction.model)
        opt.surf_extraction.model.renderer_spatial_output_dim = 128
        opt.surf_extraction.rendering.N_samples = opt.surf_extraction.model.renderer_spatial_output_dim
        opt.surf_extraction.rendering.return_xyz = True
        opt.surf_extraction.rendering.return_sdf = True
        surface_g_ema = Generator(opt.surf_extraction.model, opt.surf_extraction.rendering, full_pipeline=False).to(device)

        # Load weights to surface extractor
        surface_extractor_dict = surface_g_ema.state_dict()
        for k, v in pretrained_weights_dict.items():
            if k in surface_extractor_dict.keys() and v.size() == surface_extractor_dict[k].size():
                surface_extractor_dict[k] = v

        surface_g_ema.load_state_dict(surface_extractor_dict)
    else:
        surface_g_ema = None

    # get the mean latent vector for g_ema
    if opt.inference.truncation_ratio < 1:
        with torch.no_grad():
            mean_latent = g_ema.mean_latent(opt.inference.truncation_mean, device)
    else:
        surface_mean_latent = None

    # get the mean latent vector for surface_g_ema
    if not opt.inference.no_surface_renderings:
        surface_mean_latent = mean_latent[0]
    else:
        surface_mean_latent = None

    # load_original_image
    target_fname = "/mnt/lustre/chenzhuo.vendor/workspace/StyleGAN-nada/ZSSGAN/result_photo_toyuhao/cz.jpg"
    target_pil = Image.open(target_fname).convert('RGB')
    w, h = target_pil.size
    s = min(w, h)
    target_pil = target_pil.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    target_pil = target_pil.resize((opt.inference.size, opt.inference.size), Image.LANCZOS)
    target_uint8 = np.array(target_pil, dtype=np.uint8)
    target_image = torch.tensor(target_uint8.transpose([2, 0, 1]), device=device)
    orig_img = target_image.clone().unsqueeze(0).to(torch.float32) / 255.

    inverse(opt.inference, g_ema, surface_g_ema, device, mean_latent, surface_mean_latent, orig_img)
