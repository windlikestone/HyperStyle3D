import os
import torch
import trimesh
import numpy as np
import skvideo.io
from munch import *
from PIL import Image
from tqdm import tqdm
from torch.nn import functional as F
from torch.utils import data
# from torchvision import utils
from torchvision import transforms
from options.stylesdf_options import BaseOptions
from model_stylesdf.model import Generator
from model_stylesdf.utils  import (
    generate_camera_params, align_volume, extract_mesh_with_marching_cubes,
    xyz2mesh, create_cameras, create_mesh_renderer, add_textures,
    )
from pytorch3d.structures import Meshes

from pdb import set_trace as st

from torchvision import utils as utils_a
from criteria.clip_loss import CLIPLoss
import re
import json

torch.random.manual_seed(126)
# torch.random.manual_seed(1234)
# torch.random.manual_seed(10)
# torch.random.manual_seed(180)
# torch.random.manual_seed(445)
# torch.random.manual_seed(152)

def render_video(opt, g_ema, surface_g_ema, device, mean_latent, surface_mean_latent, text_shape_direction, text_style_direction):
    g_ema.eval()
    if not opt.no_surface_renderings or opt.project_noise:
        surface_g_ema.eval()

    images = torch.Tensor(0, 3, opt.size, opt.size)
    num_frames = 80
    # Generate video trajectory
    trajectory = np.zeros((num_frames,3), dtype=np.float32)

    # set camera trajectory
    # sweep azimuth angles (4 seconds)
    if opt.azim_video:
        t = np.linspace(0, 1, num_frames) 
        elev = 0
        fov = opt.camera.fov
        if opt.camera.uniform:
            azim = opt.camera.azim * np.cos(t * 2 * np.pi)
        else:
            azim = 1.5 * opt.camera.azim * np.cos(t * 2 * np.pi)

        trajectory[:num_frames,0] = 0.75 * azim
        trajectory[:num_frames,1] = 0.75 * elev
        trajectory[:num_frames,2] = fov

    # elipsoid sweep (4 seconds)
    else:
        t = np.linspace(0, 1, num_frames) 
        fov = opt.camera.fov #+ 1 * np.sin(t * 2 * np.pi)
        if opt.camera.uniform:
            elev = opt.camera.elev / 2 + opt.camera.elev / 2  * np.sin(t * 2 * np.pi)
            azim = opt.camera.azim  * np.cos(t * 2 * np.pi)
        else:
            elev = 1.5 * opt.camera.elev * np.sin(t * 2 * np.pi)
            azim = 1.5 * opt.camera.azim * np.cos(t * 2 * np.pi)

        trajectory[:num_frames,0] = 0.75 * azim
        trajectory[:num_frames,1] = 0.75 * elev
        trajectory[:num_frames,2] = fov
    # else:
    #     num = 3
    #     t = np.linspace(0, 1, num)
    #     t_hat = np.linspace(0, 1, num)
    #     fov = opt.camera.fov #+ 1 * np.sin(t * 2 * np.pi)
    #     elev = 1.5 * opt.camera.elev * np.cos(t * np.pi)
    #     azim = 1.5 * opt.camera.azim * np.cos(t * np.pi)
    #     for i in range(num):
    #         for j in range(num):
    #             num_frame = num * i + j
    #             trajectory[num_frame,0] = 0.75 * azim[i]
    #             trajectory[num_frame,1] = elev[j]
    #             trajectory[num_frame,2] = fov


    trajectory = torch.from_numpy(trajectory).to(device)

    # generate input parameters for the camera trajectory
    # sample_cam_poses, sample_focals, sample_near, sample_far = \
    # generate_camera_params(trajectory, opt.renderer_output_size, device, dist_radius=opt.camera.dist_radius)
    sample_cam_extrinsics, sample_focals, sample_near, sample_far, _ = \
    generate_camera_params(opt.renderer_output_size, device, locations=trajectory[:,:2],
                           fov_ang=trajectory[:,2:], dist_radius=opt.camera.dist_radius)

    # In case of noise projection, generate input parameters for the frontal position.
    # The reference mesh for the noise projection is extracted from the frontal position.
    # For more details see section C.1 in the supplementary material.
    if opt.project_noise:
        frontal_pose = torch.tensor([[0.0,0.0,opt.camera.fov]]).to(device)
        # frontal_cam_pose, frontal_focals, frontal_near, frontal_far = \
        # generate_camera_params(frontal_pose, opt.surf_extraction_output_size, device, dist_radius=opt.camera.dist_radius)
        frontal_cam_pose, frontal_focals, frontal_near, frontal_far, _ = \
        generate_camera_params(opt.surf_extraction_output_size, device, location=frontal_pose[:,:2],
                               fov_ang=frontal_pose[:,2:], dist_radius=opt.camera.dist_radius)

    # create geometry renderer (renders the depth maps)
    cameras = create_cameras(azim=np.rad2deg(trajectory[0,0].cpu().numpy()),
                             elev=np.rad2deg(trajectory[0,1].cpu().numpy()),
                             dist=1, device=device)
    renderer = create_mesh_renderer(cameras, image_size=512, specular_color=((0,0,0),),
                    ambient_color=((0.1,.1,.1),), diffuse_color=((0.75,.75,.75),),
                    device=device)

    suffix = '_azim' if opt.azim_video else '_elipsoid'

    # generate videos
    skip_num = 38
    for i in range(skip_num):
        chunk = 1
        sample_z = torch.randn(1, opt.style_dim, device=device).repeat(chunk,1)
        mixed_z  = torch.randn(1, opt.style_dim, device=device).repeat(chunk,1)

    for i in range(opt.identities):
        print('Processing identity {}/{}...'.format(i+1, opt.identities))
        chunk = 1
        sample_z = torch.randn(1, opt.style_dim, device=device).repeat(chunk,1)
        mixed_z  = torch.randn(1, opt.style_dim, device=device).repeat(chunk,1)
        video_filename = 'sample_video_{}{}.mp4'.format(i,suffix)
        writer = skvideo.io.FFmpegWriter(os.path.join(opt.results_dst_dir, video_filename),
                                         outputdict={'-pix_fmt': 'yuv420p', '-crf': '10'})
        if not opt.no_surface_renderings:
            depth_video_filename = 'sample_depth_video_{}{}.mp4'.format(i,suffix)
            depth_writer = skvideo.io.FFmpegWriter(os.path.join(opt.results_dst_dir, depth_video_filename),
                                             outputdict={'-pix_fmt': 'yuv420p', '-crf': '1'})


        ####################### Extract initial surface mesh from the frontal viewpoint #############
        # For more details see section C.1 in the supplementary material.
        if opt.project_noise:
            with torch.no_grad():
                frontal_surface_out = surface_g_ema([sample_z],
                                                    frontal_cam_pose,
                                                    frontal_focals,
                                                    frontal_near,
                                                    frontal_far,
                                                    truncation=opt.truncation_ratio,
                                                    truncation_latent=surface_mean_latent,
                                                    return_sdf=True)
                frontal_sdf = frontal_surface_out[2].cpu()

            print('Extracting Identity {} Frontal view Marching Cubes for consistent video rendering'.format(i))

            frostum_aligned_frontal_sdf = align_volume(frontal_sdf)
            del frontal_sdf

            try:
                frontal_marching_cubes_mesh = extract_mesh_with_marching_cubes(frostum_aligned_frontal_sdf)
            except ValueError:
                frontal_marching_cubes_mesh = None

            if frontal_marching_cubes_mesh != None:
                frontal_marching_cubes_mesh_filename = os.path.join(opt.results_dst_dir,'sample_{}_frontal_marching_cubes_mesh{}.obj'.format(i,suffix))
                with open(frontal_marching_cubes_mesh_filename, 'w') as f:
                    frontal_marching_cubes_mesh.export(f,file_type='obj')

            del frontal_surface_out
            torch.cuda.empty_cache()
        #############################################################################################

        for j in tqdm(range(0, num_frames, chunk)):
            with torch.no_grad():
                out = g_ema([sample_z],
                            sample_cam_extrinsics[j:j+chunk],
                            sample_focals[j:j+chunk],
                            sample_near[j:j+chunk],
                            sample_far[j:j+chunk],
                            truncation=opt.truncation_ratio,
                            truncation_latent=mean_latent,
                            clip_shape_latent=text_shape_direction,
                            clip_style_latent=text_style_direction,
                            randomize_noise=False,
                            project_noise=opt.project_noise,
                            mesh_path=frontal_marching_cubes_mesh_filename if opt.project_noise else None)

                rgb = out[0].cpu()
                # sample_locations = trajectory[:,:2],
                # for k in range(chunk):
                #     curr_locations = sample_locations[j:j+chunk]
                #     loc_str = '_azim{}_elev{}'.format(int(curr_locations[k][0] * 180 / np.pi),
                #                                       int(curr_locations[k][1] * 180 / np.pi))

                utils_a.save_image(rgb,
                os.path.join(opt.results_dst_dir,'{}.png'.format(str(i*1000 + j).zfill(7))),
                nrow=1,
                normalize=True,
                padding=0,
                range=(-1, 1),)

                # this is done to fit to RTX2080 RAM size (11GB)
                del out
                torch.cuda.empty_cache()

                # Convert RGB from [-1, 1] to [0,255]
                rgb = 127.5 * (rgb.clamp(-1,1).permute(0,2,3,1).cpu().numpy() + 1)

                # Add RGB, frame to video
                for k in range(chunk):
                    writer.writeFrame(rgb[k])

                ########## Extract surface ##########
                opt.no_surface_renderings = False
                if not opt.no_surface_renderings:
                    scale = surface_g_ema.renderer.out_im_res / g_ema.renderer.out_im_res
                    surface_sample_focals = sample_focals * scale
                    # surface_out = surface_g_ema([sample_z],
                    #                             sample_cam_extrinsics[j:j+chunk],
                    #                             surface_sample_focals[j:j+chunk],
                    #                             sample_near[j:j+chunk],
                    #                             sample_far[j:j+chunk],
                    #                             truncation=opt.truncation_ratio,
                    #                             truncation_latent=surface_mean_latent,
                    #                             return_xyz=True)
                    surface_out = surface_g_ema([sample_z],
                                                sample_cam_extrinsics[j:j+chunk],
                                                surface_sample_focals[j:j+chunk],
                                                sample_near[j:j+chunk],
                                                sample_far[j:j+chunk],
                                                truncation=opt.truncation_ratio,
                                                truncation_latent=surface_mean_latent,
                                                clip_shape_latent=text_shape_direction,
                                                clip_style_latent=text_style_direction,
                                                return_xyz=True
                                                )
                    xyz = surface_out[2].cpu()

                    # this is done to fit to RTX2080 RAM size (11GB)
                    del surface_out
                    torch.cuda.empty_cache()

                    # Render mesh for video
                    depth_mesh = xyz2mesh(xyz)
                    mesh = Meshes(
                        verts=[torch.from_numpy(np.asarray(depth_mesh.vertices)).to(torch.float32).to(device)],
                        faces = [torch.from_numpy(np.asarray(depth_mesh.faces)).to(torch.float32).to(device)],
                        textures=None,
                        verts_normals=[torch.from_numpy(np.copy(np.asarray(depth_mesh.vertex_normals))).to(torch.float32).to(device)],
                    )
                    mesh = add_textures(mesh)
                    cameras = create_cameras(azim=np.rad2deg(trajectory[j,0].cpu().numpy()),
                                             elev=np.rad2deg(trajectory[j,1].cpu().numpy()),
                                             fov=2*trajectory[j,2].cpu().numpy(),
                                             dist=1, device=device)
                    renderer = create_mesh_renderer(cameras, image_size=512,
                                                    light_location=((0.0,1.0,5.0),), specular_color=((0.2,0.2,0.2),),
                                                    ambient_color=((0.1,0.1,0.1),), diffuse_color=((0.65,.65,.65),),
                                                    device=device)

                    mesh_image = 255 * renderer(mesh).cpu().numpy()
                    mesh_image = mesh_image[...,:3]
                    print(mesh_image.shape)
                    mesh_to_save = Image.fromarray(np.uint8(mesh_image.squeeze()))
                    mesh_to_save.save(os.path.join(opt.results_dst_dir,'mesh_{}.png'.format(str(i*1000 + j).zfill(7))))
                    
                    # Add depth frame to video
                    for k in range(chunk):
                        depth_writer.writeFrame(mesh_image[k])

        # Close video writers
        writer.close()
        if not opt.no_surface_renderings:
            depth_writer.close()


if __name__ == "__main__":
    device = "cuda"
    opt = BaseOptions().parse()
    opt.model.is_test = True
    opt.model.style_dim = 256
    opt.model.freeze_renderer = False
    opt.inference.size = opt.model.model_size
    opt.inference.camera = opt.camera
    opt.inference.renderer_output_size = opt.model.renderer_spatial_output_dim
    opt.inference.style_dim = opt.model.style_dim
    opt.inference.project_noise = opt.model.project_noise
    opt.rendering.perturb = 0
    opt.rendering.force_background = True
    opt.rendering.static_viewdirs = True
    opt.rendering.return_sdf = True
    opt.rendering.N_samples = 64
    opt.no_surface_renderings = False

     # find checkpoint directory
    # check if there's a fully trained model
    # checkpoints_dir = '/mnt/lustre/chenzhuo.vendor/workspace/StyleGAN-nada/ZSSGAN/frozen_model'
    # checkpoints_dir = '/home/chenzhuo/workspace/3DAnimationGAN/ZSSGAN/model/stylesdf_red_eyes_model/checkpoint'
    # checkpoint_path = os.path.join(checkpoints_dir, opt.experiment.expname + '.pt')
   
    checkpoint_path = '/home/chenzhuo/workspace/3DAnimationGAN/ZSSGAN/model/stylesdf_pixarhyperall_layers_model/checkpoint/001500.pt'
    # checkpoint_path = '/home/chenzhuo/workspace/3DAnimationGAN/ZSSGAN/model/stylesdf_hyper_style_model/checkpoint/005000.pt'
    # checkpoint_path_i = '/home/chenzhuo/workspace/3DAnimationGAN/ZSSGAN/model/stylesdf_faceshape_model/checkpoint/000600.pt'
    checkpoint_path_i = '/home/chenzhuo/workspace/3DAnimationGAN/ZSSGAN/model/stylesdf_bearded_face_model/checkpoint/000450.pt'
    # checkpoint_path_pre = '/home/chenzhuo/workspace/cartoonGAN/ZSSGAN/model/stylesdf_elf_model/checkpoint/000150.pt'
    checkpoint_path_pre = '/home/chenzhuo/workspace/cartoonGAN/ZSSGAN/model/stylesdf_fat_face_model/checkpoint/000150.pt'
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

    results_dir_basename = os.path.join(opt.inference.results_dir, opt.experiment.expname)
    opt.inference.results_dst_dir = os.path.join(results_dir_basename, result_model_dir, 'videos')
    if opt.model.project_noise:
        opt.inference.results_dst_dir = os.path.join(opt.inference.results_dst_dir, 'with_noise_projection')

    os.makedirs(opt.inference.results_dst_dir, exist_ok=True)

    # load saved model
    checkpoint = torch.load(checkpoint_path)
    checkpoint_i = torch.load(checkpoint_path_i)
    checkpoint_pre = torch.load(checkpoint_path_pre)

    # load image generation model
    g_ema = Generator(opt.model, opt.rendering).to(device)

    # temp fix because of wrong noise sizes
    pretrained_weights_dict = checkpoint["g_ema"]
    pretrained_weights_dict_i = checkpoint_i["g_ema"]
    pretrained_weights_dict_pre = checkpoint_pre["g_ema"]
    interpolate_flag = 0.0
    model_dict = g_ema.state_dict()
    for k, v in model_dict.items():
        ptsObj = re.search( r'hyper_shape_linears', k, re.M|re.I)
        ptsObj_hyper = re.search( r'hyper_linears', k, re.M|re.I)
        ######################
        ptsObj_linear = re.search( r'pts_linears', k, re.M|re.I)
        ptsObj_0 = re.search( r'pts_linears.0', k, re.M|re.I)
        ptsObj_1 = re.search( r'pts_linears.1', k, re.M|re.I)
        ptsObj_2 = re.search( r'pts_linears.2', k, re.M|re.I)
        ptsObj_3 = re.search( r'pts_linears.3', k, re.M|re.I)
        ########################
        # print(k)
        if ptsObj != None:
            # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            k_replace = k.replace('hyper_shape_linears', 'hyper_linears')
            # print(k_replace, k)
            model_dict[k] = pretrained_weights_dict_i[k_replace]
        # decoderObj = re.search( r'decoder', k, re.M|re.I)
        # if decoderObj != None:
        #     interpolate_flag -= 0.016
        #     print(interpolate_flag, k)
        else:
            if ptsObj_hyper != None or ptsObj_0 != None or ptsObj_1 != None or ptsObj_2 != None or ptsObj_3 != None:
            # if ptsObj_hyper != None:
                model_dict[k] = pretrained_weights_dict[k]
                print(k)
            else:
                model_dict[k] = (1 - interpolate_flag) * pretrained_weights_dict[k] + interpolate_flag * pretrained_weights_dict_pre[k]
    # for k, v in pretrained_weights_dict.items():
    #     if v.size() == model_dict[k].size():
    #         model_dict[k] = v

    g_ema.load_state_dict(model_dict)

    # load a the volume renderee to a second that extracts surfaces at 128x128x128
    if not opt.inference.no_surface_renderings or opt.model.project_noise:
        opt['surf_extraction'] = Munch()
        opt.surf_extraction.rendering = opt.rendering
        opt.surf_extraction.model = opt.model
        opt.surf_extraction.model.renderer_spatial_output_dim = 128
        opt.surf_extraction.rendering.N_samples = opt.surf_extraction.model.renderer_spatial_output_dim
        opt.surf_extraction.rendering.return_xyz = True
        opt.surf_extraction.rendering.return_sdf = True
        opt.inference.surf_extraction_output_size = opt.surf_extraction.model.renderer_spatial_output_dim
        surface_g_ema = Generator(opt.surf_extraction.model, opt.surf_extraction.rendering, full_pipeline=False).to(device)

        # Load weights to surface extractor
        surface_extractor_dict = surface_g_ema.state_dict()
        # for k, v in pretrained_weights_dict.items():
        #     if k in surface_extractor_dict.keys() and v.size() == surface_extractor_dict[k].size():
        #         surface_extractor_dict[k] = v
        for k, v in surface_extractor_dict.items():
            ptsObj = re.search( r'hyper_shape_linears', k, re.M|re.I)
            ptsObj_hyper = re.search( r'hyper_linears', k, re.M|re.I)
            # print(k)
            if ptsObj != None:
                # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                k_replace = k.replace('hyper_shape_linears', 'hyper_linears')
                # print(k_replace, k)
                surface_extractor_dict[k] = pretrained_weights_dict_i[k_replace]
            # decoderObj = re.search( r'decoder', k, re.M|re.I)
            # if decoderObj != None:
            #     interpolate_flag -= 0.016
            #     print(interpolate_flag, k)
            else:
                if ptsObj_hyper != None:
                    surface_extractor_dict[k] = pretrained_weights_dict[k]
                else:
                    surface_extractor_dict[k] = (1 - interpolate_flag) * pretrained_weights_dict[k] + interpolate_flag * pretrained_weights_dict_pre[k]


        surface_g_ema.load_state_dict(surface_extractor_dict)
    else:
        surface_g_ema = None

    # get the mean latent vector for g_ema
    if opt.inference.truncation_ratio < 1:
        with torch.no_grad():
            mean_latent = g_ema.mean_latent(opt.inference.truncation_mean, device)
    else:
        mean_latent = None

    # get the mean latent vector for surface_g_ema
    if not opt.inference.no_surface_renderings or opt.model.project_noise:
        surface_mean_latent = mean_latent[0]
    else:
        surface_mean_latent = None

    source_class = "photo"
    source_shape_class = "face"
    target_shape_class = "bearded_face"
    target_style_class = "3D_Render_in_the_style_of_Pixar"
    # target_style_class = "sketch"
    source_img_dir = "output/video/photo_img"
    target_img_dir = "output/video/0"
    model_path = "/home/chenzhuo/workspace/3DAnimationGAN/model_zoo/ViT-B-16.pt"
    clip_model = CLIPLoss(device, 
                         lambda_direction=0, 
                         lambda_patch=0, 
                         lambda_global=0, 
                         lambda_manifold=0, 
                         lambda_texture=0,
                         clip_model=model_path)

    
    with torch.no_grad():
        
        text_shape_direction = clip_model.compute_text_direction(source_shape_class, target_shape_class)
        text_shape_direction = text_shape_direction.float()
        # print(text_shape_direction)
        text_style_direction = clip_model.compute_text_direction(source_class, target_style_class)
        text_style_direction = text_style_direction.float()

    render_video(opt.inference, g_ema, surface_g_ema, device, mean_latent, surface_mean_latent, text_shape_direction, text_style_direction)
