# HyperStyle3D

 # <p align="center">  under construction </p>

#### <p align="center">[Paper](https://arxiv.org/abs/2304.09463v1) | [Project Page](https://windlikestone.github.io/HyperStyle3D-website/) </p>



<p align="center">
  <img src="./assets/pipeline.png"/>
</p>

# Introduction

This repository contains the official implementation of HyperStyle3D: Text-Guided 3D Portrait Stylization via Hypernetworks.
Our model enables style transfer, attribute editing, shape deformation, and their overlying manipulations under the guidance of text prompts, while preserving 3D consistency. A more detailed introduction can be viewed on our [Project Page](https://windlikestone.github.io/HyperStyle3D-website/).

## :desktop_computer: Requirements

NVIDIA GPUs are required for this project.
We conduct all the training on NVIDIA RTX 3090-24GiB and NVIDIA RTX A6000-48GiB. 
We recommend using anaconda to manage the python environments.

The environment can be created via ```conda env create -f environment.yml```, and activated via ```conda activate hyperstyle3d```.

## :running_woman: Inference

### Download Models

The pretrained stage-1 VAE and stage-2 LDM can be downloaded via [OneDrive](https://entuedu-my.sharepoint.com/:f:/g/personal/yushi001_e_ntu_edu_sg/ErdRV9hCYvlBioObT1v_LZ4Bnwye3sv6p5qiVZPNhI9coQ?e=nJgp8t).

Put the downloaded checkpoints under ```checkpoints``` folder for inference. The checkpoints directory layout should be

    checkpoints
    ├── ffhq
    │         └── model_joint_denoise_rec_model1580000.pt
    ├── objaverse
    │        ├── model_rec1680000.pt
    │        └── model_joint_denoise_rec_model2310000.pt
    ├── shapenet
    │        └── car
    │                 └── model_joint_denoise_rec_model1580000.pt
    │        └── chair
    │                 └── model_joint_denoise_rec_model2030000.pt
    │        └── plane
    │                 └── model_joint_denoise_rec_model770000.pt
    └── ...
```

# Demo

In this section, we present an example to edting 3D model. First, generate the mlp texture.

```
sh run/texture.sh
```

Then, edit 3D model's texuture.

```
sh run/direction_edit.sh
```

# Acknowledgement
Thanks to [NVdiffrec](https://github.com/NVlabs/nvdiffrec), [Stable-DreamFusion](https://github.com/ashawkey/stable-dreamfusion) and [Fantasia3D](https://github.com/Gorilla-Lab-SCUT/Fantasia3D), our code is partially borrowing from them.

# Citation

If you find our work useful, please consider citing:
```
@ARTICLE{10542240,
  author={Chen, Zhuo and Xu, Xudong and Yan, Yichao and Pan, Ye and Zhu, Wenhan and Wu, Wayne and Dai, Bo and Yang, Xiaokang},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={HyperStyle3D: Text-Guided 3D Portrait Stylization via Hypernetworks}, 
  year={2024},
  volume={},
  number={},
  pages={1-1},
  keywords={Three-dimensional displays;Shape;Generators;Solid modeling;Semantics;Deformation;Deformable models;3D-aware GAN;Style Transfer;Hyper-network},
  doi={10.1109/TCSVT.2024.3407135}}

```
