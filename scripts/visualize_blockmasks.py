from eye.agents.eye_robot_agent import get_eye_block_mask
from eye.agents.visual_search_agent import get_vis_agent_block_mask
import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import Callable


def get_dense_mask(mask_mod: Callable, seq_length: int, device: torch.device = torch.device('cpu')):
    """
    Vectorized computation of dense mask from mask_mod function.
    Replaces O(L²) nested loops with batched tensor operations using torch.vmap.
    
    Args:
        mask_mod: Function with signature (batch_tensor, head_tensor, q_idx: int, kv_idx: int) -> bool
        seq_length: Length of the sequence
        device: Device to perform computation on
        
    Returns:
        Dense mask tensor of shape (seq_length, seq_length)
    """
    # Create dummy batch/head tensors (mask_mod signature expects these)
    dummy_batch = torch.zeros(1, device=device)
    dummy_head = torch.zeros(1, device=device)
    
    # Create indices for vectorization
    q_indices = torch.arange(seq_length, device=device)
    kv_indices = torch.arange(seq_length, device=device)
    
    # Vectorize over both dimensions using nested vmap
    # Inner vmap: vectorize over kv_idx for a fixed q_idx
    def mask_for_q_idx(q_idx):
        return torch.vmap(
            lambda kv_idx: mask_mod(dummy_batch, dummy_head, q_idx, kv_idx),
            in_dims=0
        )(kv_indices)
    
    # Outer vmap: vectorize over q_idx
    vmapped_mask = torch.vmap(mask_for_q_idx, in_dims=0)
    
    # Compute the full mask in one vectorized operation
    mask_dense = vmapped_mask(q_indices)
    return mask_dense.float()

if __name__ == "__main__":
    img_feat_size = 100
    window_len = 3
    act_size = 10

    # feats_per_t = img_feat_size + act_size + 2 + 1 + 1 + 1 # actor/critic, proprio, target, eye
    # seq_len = feats_per_t * 3
    # # learn(2), eye(1), target(1), proprio(1), joint(act_size), features(img_feat_size)
    # block_mask = get_eye_block_mask(seq_len=seq_len, feats_per_t=feats_per_t, img_feat_size=img_feat_size, window_len=window_len, act_size=act_size)


    feats_per_t = img_feat_size + 3 # learn, eye, target

    seq_len = feats_per_t * 3
    block_mask = get_vis_agent_block_mask(seq_len=seq_len, feats_per_t=feats_per_t, img_feat_size=img_feat_size, window_len=window_len, mask_past_img_attn=True)
    print(f"Computing dense mask for sequence length: {seq_len}")
    
    # Use vectorized version
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    dense_robot_mask = get_dense_mask(block_mask.mask_mod, seq_len, device).cpu()
    
    sparsity = (dense_robot_mask == 0).sum().item() / dense_robot_mask.numel()
    print(f"Sparsity of the matrix: {sparsity:.2%}")
    
    plt.figure(figsize=(10, 10))
    plt.imshow(dense_robot_mask, vmin=0, vmax=1, cmap='RdYlBu_r')
    plt.title(f'Eye-Robot Agent Block Mask (seq_len={seq_len})')
    plt.xlabel('Key/Value Index')
    plt.ylabel('Query Index')
    
    # Set explicit grid for encoder mask
    rows, cols = dense_robot_mask.shape
    plt.xticks(np.arange(-0.5, cols, 1))
    plt.yticks(np.arange(-0.5, rows, 1))
    plt.grid(True, color='grey', linewidth=0.5)
    plt.tick_params(axis='both', which='both', length=0, labelbottom=False, labelleft=False)
    
    plt.tight_layout()
    plt.savefig("block_masks.png", dpi=150, bbox_inches='tight')
    print("Saved visualization to block_masks.png")