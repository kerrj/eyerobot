import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from torchvision.transforms import Normalize
from typing import List, Tuple
import math
from transformers import SiglipImageProcessor, SiglipVisionModel
from einops import rearrange
import open_clip
import numpy as np
  
def crop_sizes_from_levels(levels: int, window_size: int, max_resoluiton: int) -> List[int]:
    if levels == 1:
        return [max_resoluiton]
    return [int(i) for i in list(np.linspace(window_size, max_resoluiton, levels))]

def create_foveated_batch(
    x: torch.Tensor, crop_sizes: List[int], window_size: int
) -> torch.Tensor:
    """
    Creates a batch of foveated images at different scales.

    Args:
        x (torch.Tensor): Input tensor in format (B, C, H, W)
        crop_sizes (List[int]): List of crop sizes for each scale
        window_size (int): Size to resize each crop to

    Returns:
        torch.Tensor: Batch of foveated images
    """
    _, _, h, w = x.shape

    # Compute maximum crop size\
    max_crop_size = max(crop_sizes)

    # Calculate necessary padding
    pad_h = max(0, max_crop_size - h)
    pad_w = max(0, max_crop_size - w)
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    # Pad the input tensor to the maximum required size
    x_padded = F.pad(
        x, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0
    )

    foveated_images = []

    for crop_size in crop_sizes:
        # Calculate start indices for cropping
        start_h = (x_padded.shape[2] - crop_size) // 2
        start_w = (x_padded.shape[3] - crop_size) // 2

        # Crop the central region
        cropped = x_padded[
            :, :, start_h : start_h + crop_size, start_w : start_w + crop_size
        ]

        # Resize to window_size x window_size using bilinear interpolation
        resized = F.interpolate(
            cropped,
            size=(window_size, window_size),
            mode="bilinear",
            align_corners=False,
            antialias=True
        )
        resized = resized.clamp(0, 1)

        foveated_images.append(resized)

    # Concatenate along the batch dimension
    return torch.cat(foveated_images, dim=0)

class MultiScaleViT(nn.Module):
    """Base class for multi-scale vision transformer encoders"""
    
    def __init__(
        self,
        crop_sizes: List[int] = [224, 448, 896, 1792],
        window_size: int = 224,
        device: str = "cpu",
        embedding_dim: int = 384,
        patch_size: int = 16,
        pool_size: int = 4,
        freeze_encoder: bool = True
    ):
        super().__init__()
        
        # Store configuration
        self.window_size = window_size
        self.crop_sizes = crop_sizes
        self.num_levels = len(crop_sizes)
        self.embedding_dim = embedding_dim
        self.patch_size = patch_size
        
        # Initialize positional embeddings
        W = max(crop_sizes)
        self.pool_size = pool_size
        self.avg_pooling = nn.AvgPool2d(kernel_size=pool_size, stride=pool_size)
        
        # This pools spatially from 16x16 to 4x4 over the H, W dimensions for (L, C, H, W) tensors
        self.rope_ids = torch.empty(
            (self.num_levels, self.window_size//(self.patch_size*self.pool_size), self.window_size//(self.patch_size*self.pool_size), 2),
            device=device
        )
        # Also calculate the rope embeddings for the image
        rows,cols = torch.meshgrid(
            torch.arange(W),
            torch.arange(W),
            indexing='ij'
        )
        ids = torch.stack([rows, cols], dim=-1) # shape (W, W, 2)
        
        for i in range(self.num_levels):
            crop_start = W//2 - self.crop_sizes[i]//2
            crop_end = W//2 + self.crop_sizes[i]//2
            ids_crop = ids[crop_start:crop_end, crop_start:crop_end]
            ids_crop = ids_crop.unsqueeze(0).permute(0, 3, 1, 2)
            # interpolate the ids to the window size and then call embednd
            ids_crop = F.interpolate(
                ids_crop.double(), 
                size=(self.window_size//(self.patch_size*self.pool_size), self.window_size//(self.patch_size*self.pool_size)), 
                mode='bilinear',
                align_corners=True,
                antialias=True
            ).permute(0, 2, 3, 1)
            self.rope_ids[i] = ids_crop

        # we unsqueeze twice since rope expects it to be (B, 1, L, D, 2, 2)
        self.rope_ids = self.rope_ids.reshape(1, -1, 2)

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def get_trainable_parameters(self):
        return []

class MultiScaleDino(MultiScaleViT):
    def __init__(self, **kwargs):
        super().__init__(embedding_dim=384, **kwargs)
        
        self.model = torch.hub.load("/home/jkerr/dinov3", 'dinov3_vits16', source='local', weights="/home/jkerr/dinov3/models/dinov3_vits16.pth").to(kwargs.get('device', 'cpu'), dtype=torch.bfloat16)
        self.model.eval()
        # self.model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14_reg").to(kwargs.get('device', 'cpu'), dtype=torch.bfloat16)
        self.normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        if kwargs['freeze_encoder']:
            for param in self.model.parameters():
                param.requires_grad = False

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = x.reshape(-1, *x.shape[2:])
        x = self.normalize(x)

        patch_tokens = self.model.get_intermediate_layers(x, reshape=True)[0] # (b n) c h w
        if self.pool_size > 1:
            patch_tokens = self.avg_pooling(patch_tokens)# (b n) c h/4 w/4
            patch_tokens = rearrange(patch_tokens, '(b n) c h w -> b n h w c', b=B, n=self.num_levels)    
        else:
            # When pool_size <= 1, use the patch tokens directly without pooling
            patch_tokens = rearrange(patch_tokens, '(b n) c h w -> b n h w c', b=B, n=self.num_levels)
        patch_tokens = patch_tokens.reshape(B, -1, self.embedding_dim)
        return patch_tokens