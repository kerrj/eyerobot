import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_block_mask
from torch import Tensor
from functools import lru_cache
from einops import rearrange, repeat
from typing import Union
from eye.foveal_encoders import MultiScaleDino
from eye.transformer import TransformerEncoder
from eye.agent_configs import RobotAgentConfig


@lru_cache
def get_robot_block_mask(seq_len: int, feats_per_t: int, act_size: int, img_feat_size: int):
    def memory_mask(b, h, q_idx, kv_idx):
        q_t = q_idx // feats_per_t
        kv_t = kv_idx // feats_per_t
        q_t_mod = q_idx % feats_per_t
        kv_t_mod = kv_idx % feats_per_t

        # --- Basic temporal causality (within window 1) ---
        causal_mask = (q_t >= kv_t) & (q_t - kv_t < 1)

        # --- Token Type Identification ---
        # Order: proprio(1), image(img_feat_size), joint(act_size)
        proprio_idx_mod = 0
        img_start_mod = 1
        img_end_mod = img_start_mod + img_feat_size
        joint_start_mod = img_end_mod

        is_q_img = (q_t_mod >= img_start_mod) & (q_t_mod < img_end_mod)
        is_kv_img = (kv_t_mod >= img_start_mod) & (kv_t_mod < img_end_mod)
        is_q_joint = (q_t_mod >= joint_start_mod)
        is_kv_joint = (kv_t_mod >= joint_start_mod)
        is_kv_proprio = (kv_t_mod == proprio_idx_mod)

        # --- Masking Rules ---
        # 1. Images cannot attend to other images
        img_to_img_mask = ~(is_q_img & is_kv_img)

        # Combine all masks
        final_mask = (
            causal_mask
            & img_to_img_mask
        )
        return final_mask

    # _compile is IMPORTANT to save memory on gpu
    print("Recomputing block mask")
    block_mask = create_block_mask(memory_mask, B=None, H=None, Q_LEN = seq_len, KV_LEN = seq_len, _compile = True)
    print("Done recomputing block mask")
    return block_mask


class RobotAgent(nn.Module):
    def __init__(self, config: Union[RobotAgentConfig, dict, None] = None, **kwargs):
        super().__init__()
        
        # Handle config initialization
        if config is None:
            self.config = RobotAgentConfig(**kwargs)
        elif isinstance(config, dict):
            self.config = RobotAgentConfig.from_dict(config)
        else:
            self.config = config
            
        # Set instance variables from config
        self.device = self.config.device
        self.action_chunk_size = self.config.action_chunk_size
        self.proprio_dropout = self.config.proprio_dropout
        self.relative_act = self.config.relative_act
        self.dino_resize_to = ((self.config.crop_size)//14)*14
        self.dino_encoder = MultiScaleDino(
            crop_sizes=[self.dino_resize_to], window_size=self.dino_resize_to, device=self.config.device
        )
        self.img_size = self.config.crop_size
        tok_dimension = 384
        self.tok_dim = tok_dimension
        self.hand_transformer = TransformerEncoder(
            hidden_size=tok_dimension,
            mlp_ratio=4.0,
            num_heads=12,
            depth=self.config.num_blocks,
            axes_dim=[8,12,12],
            theta = 1000
        )
        self.joint_token = nn.Parameter(torch.randn(1, 1, 1, tok_dimension))
        self.joint_actor = nn.Sequential(
            nn.LayerNorm(tok_dimension, elementwise_affine=False),
            nn.Linear(tok_dimension, 256),
            nn.LayerNorm(256),
            nn.GELU(approximate='tanh'),
            nn.Linear(256, 128),
            nn.GELU(approximate='tanh'),
            nn.Linear(128, 7),
        )
        self.proprio_proj = nn.Sequential(
            nn.Linear(7, tok_dimension),
            nn.LayerNorm(tok_dimension),
            nn.GELU(approximate='tanh'),
            nn.Dropout(self.proprio_dropout),
            nn.Linear(tok_dimension, tok_dimension),
            nn.GELU(approximate='tanh'),
            nn.Dropout(self.proprio_dropout),
            nn.Linear(tok_dimension, tok_dimension),
        )
        # freeze dino weights
        for param in self.dino_encoder.parameters():
            param.requires_grad = False

    def forward(self, x):
        assert "exo_image" in x or "wrist_image" in x, "Image must be in x"
        assert "proprio" in x, "Proprio must be in x"
        all_imgs = []
        if "wrist_image" in x:
            all_imgs.append(x["wrist_image"].unsqueeze(2))
        if "exo_image" in x:
            all_imgs.append(x["exo_image"].unsqueeze(2))
        all_imgs = torch.cat(all_imgs, dim=2)
        T, B, N_img, C, H, W = all_imgs.shape
        device = all_imgs.device

        # --- Image Features ---
        imgs = rearrange(all_imgs, 't b n_img c h w -> (t b n_img) c h w')

        # Calculate scaling factor to maintain aspect ratio
        scale = self.dino_resize_to / max(H, W)
        new_H, new_W = int(H * scale), int(W * scale)

        # Resize maintaining aspect ratio
        resized_imgs = F.interpolate(imgs, size=(new_H, new_W), mode='bilinear', align_corners=False)

        # Calculate padding
        pad_h = self.dino_resize_to - new_H
        pad_w = self.dino_resize_to - new_W
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        # Pad the image to dino_resize_to x dino_resize_to
        padded_imgs = F.pad(resized_imgs, (pad_left, pad_right, pad_top, pad_bottom), "constant", 0)

        # Add dimension for dino encoder
        padded_imgs = padded_imgs.unsqueeze(1) # shape: (T*B, 1, C, dino_resize_to, dino_resize_to)
        
        img_feats = self.dino_encoder(padded_imgs)
        img_feats = rearrange(img_feats, '(t b n_img) l d -> b t (n_img l) d', t=T, b=B, n_img=N_img)
        img_feat_size = img_feats.shape[2]
        img_ids = self.dino_encoder.rope_ids.unsqueeze(0).repeat(B, T, N_img, 1)
        if N_img > 1:
            # offset the rope ids for second image
            img_ids[:,:,1,1] += self.dino_resize_to

        # --- Proprio Tokens ---
        proprio_input = x["proprio"]
        proprio_input = rearrange(proprio_input, 't b c -> (t b) c')
        proprio_tokens = self.proprio_proj(proprio_input).unsqueeze(1)
        proprio_tokens = rearrange(proprio_tokens, '(t b) l d -> b t l d', t=T, b=B)
        proprio_ids = torch.full((B, T, 1, 2), self.dino_resize_to/2.0, device=device)

        # --- Joint Tokens ---
        joint_tokens = self.joint_token.repeat(B, T, self.action_chunk_size, 1)
        joint_ids = torch.full((B, T, self.action_chunk_size, 2), self.dino_resize_to/2.0, device=device)

        # --- Concatenate Tokens and IDs ---
        tokens = torch.cat([proprio_tokens, img_feats, joint_tokens], dim=2)
        ids = torch.cat([proprio_ids, img_ids, joint_ids], dim=2).div_(14.0)

        # --- Time IDs ---
        time_ids = torch.arange(T, device=device).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        time_ids = time_ids.repeat(B, 1, tokens.shape[2], 1)

        # Assign future time ids for joint slots
        joint_start_tok_idx = 1 + img_feat_size
        future_time_ids = rearrange(torch.arange(self.action_chunk_size, device=device), 'l -> 1 1 l 1')
        future_time_ids = repeat(future_time_ids, '1 1 l 1 -> b t l 1', b=B, t=T)
        future_time_ids = future_time_ids + rearrange(torch.arange(T, device=device), 't -> 1 t 1 1')
        time_ids[:,:, joint_start_tok_idx : joint_start_tok_idx + self.action_chunk_size, :] = future_time_ids

        # Finalize IDs and Reshape
        ids = torch.cat([time_ids, ids], dim=-1)
        ids = rearrange(ids, 'b t l d -> b (t l) d')
        tokens = rearrange(tokens, 'b t l d -> b (t l) d')

        # --- Transformer ---
        feats_per_t = tokens.shape[1] // T
        block_mask = get_robot_block_mask(tokens.shape[1], feats_per_t, self.action_chunk_size, img_feat_size)
        transformed = self.hand_transformer(tokens, ids, block_mask)
        transformed = rearrange(transformed, 'b (t l) d -> t b l d', t=T)

        # --- Joint Actor ---
        joint_outputs = transformed[:, :, joint_start_tok_idx : joint_start_tok_idx + self.action_chunk_size, :]
        joint_actions = self.joint_actor(joint_outputs)
        if self.relative_act:
            joint_actions = x['proprio'].unsqueeze(2) + joint_actions

        return joint_actions