import sys
import os
sys.path.insert(0, os.path.abspath('../'))
# sys.path.append('/mnt/lustre/chenzhuo.vendor/workspace/StyleGAN-nada/pi-GAN/pretrained_model/CelebA/generator.pth')


import torch
import torchvision.transforms as transforms

import numpy as np
import copy

from functools import partial
from ZSSGAN.model_stylesdf.model import Generator


from ZSSGAN.criteria.clip_loss import CLIPLoss
from ZSSGAN.criteria.tv_loss import TVLoss 
from random import sample
from PIL import Image    

from model_stylesdf.utils import (
    generate_camera_params,
    align_volume,
    extract_mesh_with_marching_cubes,
    xyz2mesh,
    make_noise,
    mixing_noise
)

from torchvision import utils as utils_a

# SIREN = getattr(siren, SPATIALSIRENBASELINE)

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

class SG2Generator(torch.nn.Module):
    def __init__(self, opt, checkpoint_path_o, checkpoint_path_i, latent_size=256, device='cuda:0'):
        
        super(SG2Generator, self).__init__()

        self.opt = opt
        checkpoint_o = torch.load(checkpoint_path_o)

        checkpoint_i = torch.load(checkpoint_path_i)
        # self.generator = checkpoint.to(device)

        self.generator = Generator(opt.model, opt.rendering).to(device)
        
        pretrained_weights_dict_o = checkpoint_o["g_ema"]
        pretrained_weights_dict_i = checkpoint_i["g_ema"]

        weight_o = 0.8
        weight_i = 1 - weight_o
        
        # self.generator.load_state_dict(pretrained_weights_dict)
        
        model_dict = self.generator.state_dict()
        for k, v in pretrained_weights_dict_o.items():
            if v.size() == model_dict[k].size():
                model_dict[k] = weight_o * v + weight_i * pretrained_weights_dict_i[k]

        self.generator.load_state_dict(model_dict)

        # # TODO
        with torch.no_grad():
            self.mean_latent = self.generator.mean_latent(4096, device=device)

    def get_all_layers(self):
        # print("line 65 generator_children:", list(self.generator.children()))
        return list(self.generator.children())

    # def get_training_layers(self, phase):

    #     if phase == 'texture':
    #         # learned constant + first convolution + layers 3-10
    #         return list(self.get_all_layers())[1:3] + list(self.get_all_layers()[4][2:10])   
    #     if phase == 'shape':
    #         # layers 1-2
    #          return list(self.get_all_layers())[1:3] + list(self.get_all_layers()[4][0:2])
    #     if phase == 'no_fine':
    #         # const + layers 1-10
    #          return list(self.get_all_layers())[1:3] + list(self.get_all_layers()[4][:10])
    #     if phase == 'shape_expanded':
    #         # const + layers 1-10
    #          return list(self.get_all_layers())[1:3] + list(self.get_all_layers()[4][0:3])
    #     if phase == 'all':
    #         # everything, including mapping and ToRGB
    #         return self.get_all_layers() 
    #     else: 
    #         # everything except mapping and ToRGB
    #         return list(self.get_all_layers())[1:3] + list(self.get_all_layers()[4][:])  

    def freeze_layers(self, layer_list=None):
        '''
        Disable training for all layers in list.
        '''
        if layer_list is None:
            self.freeze_layers(self.get_all_layers())
        else:
            for layer in layer_list:
                requires_grad(layer, False)

    def unfreeze_layers(self, layer_list=None):
        '''
        Enable training for all layers in list.
        '''
        if layer_list is None:
            self.unfreeze_layers(self.get_all_layers())
        else:
            for layer in layer_list:
                requires_grad(layer, True)

    # def style(self, styles):
    #     '''
    #     Convert z codes to w codes.
    #     '''
    #     styles = [self.generator.style(s) for s in styles]
    #     return styles

    # def get_s_code(self, styles, input_is_latent=False):
    #     return self.generator.get_s_code(styles, input_is_latent)

    # def modulation_layers(self):
    #     return self.generator.modulation_layers

    def forward(self, sample_z, cam_extrinsics, focal, near, far, input_is_latent=False):
        # print(latent_z)
        # print(latent_z.shape)
        out = self.generator(sample_z,
                            cam_extrinsics,
                            focal,
                            near,
                            far,
                            input_is_latent=input_is_latent,
                            truncation=0.7,
                            truncation_latent=self.mean_latent,
                            return_normal=False)
        return out

class ZSSGAN(torch.nn.Module):
    def __init__(self, args, opt):
        super(ZSSGAN, self).__init__()

        self.args = args

        self.device = 'cuda:0'

        # Set up frozen (source) generator
        self.generator_frozen = SG2Generator(opt, args.frozen_gen_ckpt, args.train_gen_ckpt).to(self.device)
        self.generator_frozen.freeze_layers()
        self.generator_frozen.eval()

        # Set up trainable (target) generator
        self.generator_trainable = SG2Generator(opt, args.frozen_gen_ckpt, args.train_gen_ckpt).to(self.device)
        # self.generator_trainable.freeze_layers()
        # self.generator_trainable.unfreeze_layers(self.generator_trainable.get_training_layers(args.phase))
        self.generator_trainable.train()

        # Losses
        self.clip_loss_models = {model_name: CLIPLoss(self.device, 
                                                      lambda_direction=args.lambda_direction, 
                                                      lambda_patch=args.lambda_patch, 
                                                      lambda_global=args.lambda_global, 
                                                      lambda_manifold=args.lambda_manifold, 
                                                      lambda_texture=0.001,
                                                      clip_model=model_name) 
                                for model_name in args.clip_models}

        self.clip_model_weights = {model_name: weight for model_name, weight in zip(args.clip_models, args.clip_model_weights)}

        self.mse_loss  = torch.nn.MSELoss()
        self.ce_loss = torch.nn.CrossEntropyLoss()
        self.l1_loss = torch.nn.L1Loss()
        self.smooth_loss = TVLoss()

        self.source_class = args.source_class
        self.target_class = args.target_class

        self.auto_layer_k     = args.auto_layer_k
        self.auto_layer_iters = args.auto_layer_iters

        if args.target_img_list is not None:
            self.set_img2img_direction()
        # test
        # self.region_index = self.determine_region()
        # self.text_index = self.determine_text_region()

    def set_img2img_direction(self):
        with torch.no_grad():
            # print("set_img2img_direction, ", self.args.target_img_list)
            fixed_z = mixing_noise(self.args.batch, self.args.style_dim, self.args.mixing, self.device)
            fixed_cam_extrinsics, fixed_focal, fixed_near, fixed_far, fixed_gt_viewpoints = generate_camera_params(self.args.renderer_output_size, self.device, batch=self.args.batch,
                                                                                                                   uniform=self.args.camera.uniform, azim_range=self.args.camera.azim,
                                                                                                                   elev_range=self.args.camera.elev, fov_ang=self.args.camera.fov,
                                                                                                                   dist_radius=self.args.camera.dist_radius)

            
            generated = self.generator_trainable(fixed_z, fixed_cam_extrinsics, fixed_focal, fixed_near, fixed_far)[0]
            # generated = self.generator_trainable([sample_z])[0]

            for _, model in self.clip_loss_models.items():
                direction = model.compute_img2img_direction(generated, self.args.target_img_list)
                # direction = model.compute_eye2eye_direction(generated, self.args.target_img_list)

                model.target_direction = direction

    def determine_opt_layers(self):

        # initial_w_codes = self.generator_frozen.style([sample_z])
        # initial_w_codes = initial_w_codes[0].unsqueeze(1).repeat(1, self.generator_frozen.generator.n_latent, 1)

        # w_codes = torch.Tensor(initial_w_codes.cpu().detach().numpy()).to(self.device)

        # w_codes.requires_grad = True

        # w_optim = torch.optim.SGD([w_codes], lr=0.01)

        # for _ in range(self.auto_layer_iters):
        #     w_codes_for_gen = w_codes.unsqueeze(0)
        #     generated_from_w = self.generator_trainable(w_codes_for_gen, input_is_latent=True)[0]

        #     w_loss = [self.clip_model_weights[model_name] * self.clip_loss_models[model_name].global_clip_loss(generated_from_w, self.target_class) for model_name in self.clip_model_weights.keys()]
        #     w_loss = torch.sum(torch.stack(w_loss))
            
        #     w_optim.zero_grad()
        #     w_loss.backward()
        #     w_optim.step()
        
        # layer_weights = torch.abs(w_codes - initial_w_codes).mean(dim=-1).mean(dim=0)
        # chosen_layer_idx = torch.topk(layer_weights, self.auto_layer_k)[1].cpu().numpy()

        all_layers = list(self.generator_trainable.get_all_layers())
        # print("all_layers", all_layers)

        render_mapping_layers = list(all_layers[1].children())
        # print("render_mapping_layers", render_mapping_layers)
        volume_render_layers = list(all_layers[2].children())
        # print("volume_render_layers", volume_render_layers)
        stylegan_layers = list(all_layers[3].children())
        # print("stylegan_layers", stylegan_layers)
        idx_to_layer = render_mapping_layers

        print("Number of layers:", len(idx_to_layer))
        chosen_layers = idx_to_layer
                
        return chosen_layers

    def region_mask(self, trained_img):

        number_of_classes = 19
        region_classes = [3]
        reverse_mask = True
        region_weights = torch.zeros([number_of_classes])
        model_n = self.args.clip_models[0]
        # for region_index in range(number_of_classes):
        for region_index in region_classes:
            f_mask = self.clip_loss_models[model_n].face_mask(trained_img, region_index)
        
        return f_mask

    def determine_text_region(self):

        source_text = ''
        model_n = self.args.clip_models[0]
        region_distance = self.clip_loss_models[model_n].compute_region_distance(source_text)

        print("region_distance", region_distance)
        exit()


        if self.training and self.auto_layer_iters > 0:
            self.generator_frozen.freeze_layers()
            self.generator_trainable.unfreeze_layers()
            train_layers = self.determine_opt_layers()

            if not isinstance(train_layers, list):
                train_layers = [train_layers]

            self.generator_trainable.freeze_layers()
            self.generator_trainable.unfreeze_layers(train_layers)

        initial_z = [torch.randn(self.args.auto_layer_batch, 256, device=self.device)]

        # confidence map
        initial_cam_extrinsics, initial_focal, initial_near, initial_far, initial_gt_viewpoints = generate_camera_params(self.args.renderer_output_size, self.device, batch=self.args.auto_layer_batch,
                                                                                                            uniform=self.args.camera.uniform, azim_range=self.args.camera.azim,
                                                                                                            elev_range=self.args.camera.elev, fov_ang=self.args.camera.fov,
                                                                                                            dist_radius=self.args.camera.dist_radius)
        initial_img = self.generator_frozen(initial_z, initial_cam_extrinsics, initial_focal, initial_near, initial_far)[0]

        for _ in range(self.auto_layer_iters):
            sample_z = [torch.randn(self.args.auto_layer_batch, 256, device=self.device)]

            sample_cam_extrinsics, sample_focal, near, far, gt_viewpoints = generate_camera_params(self.args.renderer_output_size, self.device, batch=self.args.auto_layer_batch,
                                                                                                uniform=self.args.camera.uniform, azim_range=self.args.camera.azim,
                                                                                                elev_range=self.args.camera.elev, fov_ang=self.args.camera.fov,
                                                                                                dist_radius=self.args.camera.dist_radius)
            
            with torch.no_grad():       
                frozen_img = self.generator_frozen(sample_z, sample_cam_extrinsics, sample_focal, near, far)[0]

            trainable_img = self.generator_trainable(sample_z, sample_cam_extrinsics, sample_focal, near, far)[0]
            d_loss = [self.clip_model_weights[model_name] * self.clip_loss_models[model_name].clip_directional_loss(frozen_img, self.source_class, trainable_img, self.target_class) for model_name in self.clip_model_weights.keys()]
            d_loss = torch.sum(torch.stack(d_loss))
                
            self.generator_trainable.zero_grad()
            d_loss.backward()
            g_optim.step()

        trained_img = self.generator_trainable(initial_z, initial_cam_extrinsics, initial_focal, initial_near, initial_far)[0]

        number_of_classes = 19
        reverse_mask = True
        region_weights = torch.zeros([number_of_classes])
        model_n = self.args.clip_models[0]
        for region_index in range(number_of_classes):
            f_img, t_img = self.clip_loss_models[model_n].face_segmentation(initial_img, trained_img, region_index, reverse_mask)
            number_region_pixel = torch.nonzero(f_img).shape[0] / (1024 * 1024) + 1e-8
            # number_region_pixel = torch.nonzero(f_img - t_img).shape[0] / (1024 * 1024)
            print("number_region_pixel", number_region_pixel)
            region_weights[region_index] = torch.abs(f_img - t_img).mean() / number_region_pixel
        print("region_weights", region_weights)
        chosen_region_idx = torch.topk(region_weights, self.auto_layer_k)[1].cpu().numpy()
        print("chosen_region_idx", chosen_region_idx)

        exit()

    def forward(self,latent_z,cam_extrinsics,focal,near,far,input_is_latent=False):

        # if self.training and self.auto_layer_iters > 0:
        #     self.generator_trainable.unfreeze_layers()
        #     train_layers = self.determine_opt_layers()

        #     if not isinstance(train_layers, list):
        #         train_layers = [train_layers]

        #     self.generator_trainable.freeze_layers()
        #     self.generator_trainable.unfreeze_layers(train_layers)

        # if input_is_latent:
        #     w_styles = styles
        # else:
        #     w_styles = self.generator_frozen.style(styles)

        with torch.no_grad():
            preprocessed_images = None
            # if self.args.target_img_list is not None:
            #     texture_images = sample(self.args.target_img_list, latent_z[0].shape[0])
                

            #     for _, model in self.clip_loss_models.items():
            #         initail_image = texture_images[0]
                    
            #         preprocessed_images = model.preresize(Image.open(initail_image))
                    
            #         preprocessed_images = preprocessed_images.unsqueeze(0).to(self.device)
                    
            #         for texture_image in texture_images[1:]:
            #             preprocessed = model.preresize(Image.open(texture_image)).unsqueeze(0).to(self.device)
            #             # torchvision.utils.save_image(preprocessed, '/lustre/chenzhuo.vendor/workspace/StyleGAN-nada/ZSSGAN/pixar_img/%08d.jpg'%(1))
            #             preprocessed_images = torch.cat((preprocessed_images, preprocessed), 0)

                # print("texture_image", preprocessed_images.shape)
            
            frozen_output = self.generator_frozen(latent_z, cam_extrinsics, focal, near, far, input_is_latent)
            frozen_img = frozen_output[0]
            # f_hair_mask = frozen_output[-1]
            # real_f_hair = frozen_output[-2]
            # f_fg_hair = real_f_hair * frozen_img

        trainable_output = self.generator_trainable(latent_z, cam_extrinsics, focal, near, far, input_is_latent)
        trainable_img = trainable_output[0]

        # remake_hair
        # t_hair_mask = trainable_output[-1]
        # real_t_hair = trainable_output[-2]
        # t_fg_hair = real_t_hair * trainable_img
        # t_hair = t_hair_mask * trainable_img
        # hair_loss = self.l1_loss(t_hair, f_fg_hair)
        # original_loss = self.l1_loss(t_fg_hair, f_fg_hair)
        # print("hair_loss", hair_loss)
        # print("original_loss", original_loss)
        # hair end
        
        # trainable_normal = trainable_output[-1] * self.region_mask(trainable_output[1])

        # trainable_img = trainable_img * self.region_mask(trainable_img)
        # frozen_img = frozen_img * self.region_mask(frozen_img)

        clip_loss = 0
        if (self.args.target_img_list is not None) and (preprocessed_images is not None):
            clip_loss = torch.sum(torch.stack([self.clip_model_weights[model_name] * self.clip_loss_models[model_name](frozen_img, self.source_class, trainable_img, self.target_class, preprocessed_images) for model_name in self.clip_model_weights.keys()]))
        else:
            clip_loss = torch.sum(torch.stack([self.clip_model_weights[model_name] * self.clip_loss_models[model_name](frozen_img, self.source_class, trainable_img, self.target_class) for model_name in self.clip_model_weights.keys()]))

        # clip_loss = hair_loss

        return [frozen_img, trainable_img], clip_loss
        # return [f_fg_hair, t_fg_hair], clip_loss

    def pivot(self):
        par_frozen = dict(self.generator_frozen.named_parameters())
        par_train  = dict(self.generator_trainable.named_parameters())

        for k in par_frozen.keys():
            par_frozen[k] = par_train[k]
