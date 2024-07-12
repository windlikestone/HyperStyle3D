import torch
import torch.nn as nn
import torch.functional as tf


from warp_2D.warp_sparse_image_2D import sparse_image_warp


class WarpController(nn.Module):
    """
    
    Warp Controller network.
    
    """

    def __init__(self, n_ldmark, device):
        """

        
        
        """
        super().__init__()

        self.n_ldmark    = n_ldmark
        self.device      = device

        
    def forward(self, images_rendered: torch.Tensor, landmarks_source: torch.Tensor, landmarks_target: torch.Tensor) -> tuple:

        landmarks_src = torch.reshape(landmarks_source.detach().clone(), (-1, self.n_ldmark, 2)).to(self.device)
        landmarks_dst = torch.reshape((landmarks_target).detach().clone(), (-1, self.n_ldmark, 2)).to(self.device)

        # shape: (1)
        landmarks_norm = torch.mean(torch.norm(landmarks_src - landmarks_dst, dim=(1, 2)))

        images_rendered = images_rendered.to(self.device)

        images_transformed, dense_flow = sparse_image_warp(self.device, images_rendered, landmarks_src, landmarks_dst, regularization_weight = 1e-6, num_boundary_points = 0)

        return images_transformed
