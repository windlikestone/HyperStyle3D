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
import json
import re
# from utils.file_utils import copytree, save_images, save_paper_image_grid


torch.random.manual_seed(520)

def load_init_params(filename):
    with open(f"/mnt/lustre/chenzhuo.vendor/workspace/StyleGAN-nada/ZSSGAN/result_photo_toyuhao/{filename}.json",'r',encoding='utf-8') as f:
        initial_dict = json.load(f)
    return initial_dict

def generate(opt, g_ema, surface_g_ema, device, mean_latent, surface_mean_latent):

    # init_params = load_init_params('00000_params')
    g_ema.eval()
    if not opt.no_surface_renderings:
        surface_g_ema.eval()

    # set camera angles
    if opt.fixed_camera_angles:
        # These can be changed to any other specific viewpoints.
        # You can add or remove viewpoints as you wish
        locations = torch.tensor([[0, 0],
                                  [-1.5 * opt.camera.azim, 0],
                                  [-1 * opt.camera.azim, 0],
                                  [-0.5 * opt.camera.azim, 0],
                                  [0.5 * opt.camera.azim, 0],
                                  [1 * opt.camera.azim, 0],
                                  [1.5 * opt.camera.azim, 0],
                                  [0, -1.5 * opt.camera.elev],
                                  [0, -1 * opt.camera.elev],
                                  [0, -0.5 * opt.camera.elev],
                                  [0, 0.5 * opt.camera.elev],
                                  [0, 1 * opt.camera.elev],
                                  [0, 1.5 * opt.camera.elev]], device=device)
        # For zooming in/out change the values of fov
        # (This can be defined for each view separately via a custom tensor
        # like the locations tensor above. Tensor shape should be [locations.shape[0],1])
        # reasonable values are [0.75 * opt.camera.fov, 1.25 * opt.camera.fov]
        fov = opt.camera.fov * torch.ones((locations.shape[0],1), device=device)
        num_viewdirs = locations.shape[0]
    else: # draw random camera angles
        locations = None
        # fov = None
        fov = opt.camera.fov
        num_viewdirs = opt.num_views_per_id

    # generate images
    for i in tqdm(range(opt.identities)):
        with torch.no_grad():
            chunk = 8
            print("num_viewdirs", num_viewdirs)
            style_mixing = False

            sample_z = torch.randn(1, opt.style_dim, device=device).repeat(num_viewdirs,1)
            # style mixing
            
            mixed_z = torch.randn(1, opt.style_dim, device=device).repeat(num_viewdirs,1)
            # latent_code_init = torch.tensor(init_params["latent_z"], device=device).repeat(num_viewdirs,1)
            
            
            sample_cam_extrinsics, sample_focals, sample_near, sample_far, sample_locations = \
            generate_camera_params(opt.renderer_output_size, device, batch=num_viewdirs,
                                   locations=locations, #input_fov=fov,
                                   uniform=opt.camera.uniform, azim_range=opt.camera.azim,
                                   elev_range=opt.camera.elev, fov_ang=fov,
                                   dist_radius=opt.camera.dist_radius)
            rgb_images = torch.Tensor(0, 3, opt.size, opt.size)
            
            rgb_images_thumbs = torch.Tensor(0, 3, opt.renderer_output_size, opt.renderer_output_size)
            
            normal_maps = torch.Tensor(0, 3, opt.renderer_output_size, opt.renderer_output_size)
            for j in range(0, num_viewdirs, chunk):
                # print(j)
                out = g_ema([sample_z[j:j+chunk]],
                            sample_cam_extrinsics[j:j+chunk],
                            sample_focals[j:j+chunk],
                            sample_near[j:j+chunk],
                            sample_far[j:j+chunk],
                            truncation=opt.truncation_ratio,
                            truncation_latent=mean_latent,
                            return_depth=True,
                            return_normal=True)
                print("!!!!!!!!!!!!!!!!!1")



                # out_ref = g_ema([mixed_z[j:j+chunk]],
                #             sample_cam_extrinsics[j:j+chunk],
                #             sample_focals[j:j+chunk],
                #             sample_near[j:j+chunk],
                #             sample_far[j:j+chunk],
                #             truncation=opt.truncation_ratio,
                #             truncation_latent=mean_latent)
                # for inject_index in range(8):
                #     out_mix = g_ema([sample_z[j:j+chunk],mixed_z[j:j+chunk]],
                #                 sample_cam_extrinsics[j:j+chunk],
                #                 sample_focals[j:j+chunk],
                #                 sample_near[j:j+chunk],
                #                 sample_far[j:j+chunk],
                #                 truncation=opt.truncation_ratio,
                #                 truncation_latent=mean_latent,
                #                 inject_index=inject_index)
                #     rgb_images_mix = torch.cat([rgb_images_mix, out_mix[0].cpu()], 0)

                rgb_images = torch.cat([rgb_images, out[0].cpu()], 0)
                # rgb_images_ref = torch.cat([rgb_images_ref, out_ref[0].cpu()], 0)
                
                # rgb_images_diff = torch.cat([rgb_images_diff, (out_mix[0]-out[0]).cpu()], 0)

                rgb_images_thumbs = torch.cat([rgb_images_thumbs, out[1].cpu()], 0)
                # depth_maps = torch.cat([depth_maps, out[-2].cpu()], 0)
                # normal_maps = torch.cat([normal_maps, out[-1].cpu()], 0)

                del out
                torch.cuda.empty_cache()

            utils_a.save_image(rgb_images,
                os.path.join(opt.results_dst_dir, 'images','{}.png'.format(str(i).zfill(7))),
                nrow=num_viewdirs,
                normalize=True,
                padding=0,
                range=(-1, 1),)

            # utils_a.save_image((depth_maps-0.88)/0.24, 
            #     os.path.join(opt.results_dst_dir, 'images','{}_depth.png'.format(str(i).zfill(7))), 
            #     normalize=True)

            # utils_a.save_image(normal_maps/2+0.5, 
            #     os.path.join(opt.results_dst_dir, 'images','{}_normal.png'.format(str(i).zfill(7))), 
            #     normalize=True)

            utils_a.save_image(rgb_images_thumbs,
                os.path.join(opt.results_dst_dir, 'images','{}_thumb.png'.format(str(i).zfill(7))),
                nrow=num_viewdirs,
                normalize=True,
                padding=0,
                range=(-1, 1),)

            # this is done to fit to RTX2080 RAM size (11GB)
            # del out
            torch.cuda.empty_cache()
            opt.no_surface_renderings = False

            if not opt.no_surface_renderings:
                surface_chunk = 1
                scale = surface_g_ema.renderer.out_im_res / g_ema.renderer.out_im_res
                surface_sample_focals = sample_focals * scale
                for j in range(0, num_viewdirs, surface_chunk):
                    surface_out = surface_g_ema([sample_z[j:j+surface_chunk]],
                                                sample_cam_extrinsics[j:j+surface_chunk],
                                                surface_sample_focals[j:j+surface_chunk],
                                                sample_near[j:j+surface_chunk],
                                                sample_far[j:j+surface_chunk],
                                                truncation=opt.truncation_ratio,
                                                truncation_latent=surface_mean_latent,
                                                return_sdf=True,
                                                return_xyz=True)

                    xyz = surface_out[2].cpu()
                    # print("xyz", xyz)
                    sdf = surface_out[3].cpu()

                    # this is done to fit to RTX2080 RAM size (11GB)
                    del surface_out
                    torch.cuda.empty_cache()

                    # mesh extractions are done one at a time
                    for k in range(surface_chunk):
                        curr_locations = sample_locations[j:j+surface_chunk]
                        loc_str = '_azim{}_elev{}'.format(int(curr_locations[k,0] * 180 / np.pi),
                                                          int(curr_locations[k,1] * 180 / np.pi))

                        # Save depth outputs as meshes
                        depth_mesh_filename = os.path.join(opt.results_dst_dir,'depth_map_meshes','sample_{}_depth_mesh{}.obj'.format(i, loc_str))
                        depth_mesh = xyz2mesh(xyz[k:k+surface_chunk])
                        if depth_mesh != None:
                            with open(depth_mesh_filename, 'w') as f:
                                depth_mesh.export(f,file_type='obj')

                        # extract full geometry with marching cubes
                        if j == 0:
                            try:
                                frostum_aligned_sdf = align_volume(sdf)
                                marching_cubes_mesh = extract_mesh_with_marching_cubes(frostum_aligned_sdf[k:k+surface_chunk])
                            except ValueError:
                                marching_cubes_mesh = None
                                print('Marching cubes extraction failed.')
                                print('Please check whether the SDF values are all larger (or all smaller) than 0.')

                            if marching_cubes_mesh != None:
                                marching_cubes_mesh_filename = os.path.join(opt.results_dst_dir,'marching_cubes_meshes','sample_{}_marching_cubes_mesh{}.obj'.format(i, loc_str))
                                with open(marching_cubes_mesh_filename, 'w') as f:
                                    marching_cubes_mesh.export(f,file_type='obj')


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

    # find checkpoint directory
    # check if there's a fully trained model
    checkpoints_dir = '/home/chenzhuo/workspace/3DAnimationGAN/ZSSGAN/model/stylesdf_small_eyes_model/checkpoint'
    # checkpoints_dir = '/home/chenzhuo/workspace/3DAnimationGAN/model_zoo'

    # checkpoints_dir = '/mnt/lustre/chenzhuo.vendor/workspace/StyleGAN-nada/ZSSGAN/frozen_model'
    # # checkpoint_path = os.path.join(checkpoints_dir, opt.experiment.expname + '.pt')
    checkpoint_path = os.path.join(checkpoints_dir, opt.experiment.expname + '.pt')
    checkpoint_path_i = '/home/chenzhuo/workspace/3DAnimationGAN/ZSSGAN/model/stylesdf_red_eyes_model/checkpoint/000300.pt'
    
    print("checkpoint_path", checkpoint_path)
    if os.path.isfile(checkpoint_path):
        # define results directory name
        result_model_dir = 'final_model'
    else:
        print("None ckpt!")
    # else:
    #     checkpoints_dir = os.path.join('checkpoint', opt.experiment.expname)
    #     checkpoint_path = os.path.join(checkpoints_dir,
    #                                    'models_{}.pt'.format(opt.experiment.ckpt.zfill(7)))
    #     # define results directory name
    #     result_model_dir = 'iter_{}'.format(opt.experiment.ckpt.zfill(7))

    # create results directory
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
    checkpoint_i = torch.load(checkpoint_path_i)

    # load image generation model
    g_ema = Generator(opt.model, opt.rendering).to(device)
    pretrained_weights_dict = checkpoint["g_ema"]
    pretrained_weights_dict_i = checkpoint_i["g_ema"]
    interpolate_flag = 0.8
    model_dict = g_ema.state_dict()
    for k, v in pretrained_weights_dict.items():
        if v.size() == model_dict[k].size():
            ptsObj = re.search( r'pts_linears', k, re.M|re.I)
            if ptsObj != None:
                # interpolate_flag -= 0.01
                print(interpolate_flag, k)
            # decoderObj = re.search( r'decoder', k, re.M|re.I)
            # if decoderObj != None:
            #     interpolate_flag -= 0.016
            #     print(interpolate_flag, k)
            
            model_dict[k] = (1 - interpolate_flag) * v + interpolate_flag * pretrained_weights_dict_i[k]
    # for k, v in pretrained_weights_dict.items():
    #         if k in model_dict.keys() and v.size() == model_dict[k].size():
    #             model_dict[k] = v

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
        # opt.surf_extraction.model.renderer_spatial_output_dim = 64
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

    generate(opt.inference, g_ema, surface_g_ema, device, mean_latent, surface_mean_latent)
