"""
Pared down implementation of the DiT block from Flux.
"""
import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor
from functools import lru_cache
from torch.nn.attention.flex_attention import (
    create_block_mask,
    flex_attention,
    BlockMask
)
from torch.utils.checkpoint import checkpoint
from typing import Optional, Callable
flex_attention = torch.compile(flex_attention, dynamic=True)

class EmbedND(nn.Module):
    def __init__(self, dim: int, theta: int, axes_dim: list[int]):
        """
        dim is the length of the embedding vector (in this case the dimension of each head)
        theta is the maximum period of the sine wave
        axes_dim is the amount of dimensions to use for each axis of the position index
        """
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: Tensor) -> Tensor:
        """
        Takes in B N (num_dims) shape tensor of ids
        ids is the position index of the embedding vector
        len(axes_dim) == ids.shape[-1]  because the position coordinates must be broken down into the corresponding length vectors in axes_dim
        """
        assert len(self.axes_dim) == ids.shape[-1]
        n_axes = ids.shape[-1]
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            dim=-3,
        )


        return emb.unsqueeze(1)

def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    assert dim % 2 == 0
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos, omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.float()


def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)


def compute_dense_mask_from_block_mask(block_mask, seq_length: int, device: torch.device) -> Tensor:
    """
    Vectorized computation of dense mask from block mask using torch.vmap.
    Replaces O(L²) nested loops with batched tensor operations.
    
    Args:
        block_mask: BlockMask object with mask_mod function
        seq_length: Sequence length
        device: Device to perform computation on
        
    Returns:
        Dense boolean mask tensor of shape (seq_length, seq_length)
    """
    dummy_batch = torch.zeros(1, device=device)
    dummy_head = torch.zeros(1, device=device)
    
    q_indices = torch.arange(seq_length, device=device)
    kv_indices = torch.arange(seq_length, device=device)
    
    def mask_for_q_idx(q_idx):
        return torch.vmap(
            lambda kv_idx: block_mask.mask_mod(dummy_batch, dummy_head, q_idx, kv_idx),
            in_dims=0
        )(kv_indices)
    
    vmapped_mask = torch.vmap(mask_for_q_idx, in_dims=0)
    mask_dense = vmapped_mask(q_indices).bool()
    return mask_dense


def explicit_attention(q: Tensor, k: Tensor, v: Tensor, pe: Tensor, attn_mask = None) -> tuple[Tensor, Tensor]:
    """
    Alternative to flex_attention that explicitly computes and returns the attention matrix.
    Returns both the output and the attention weights for visualization.
    """
    q, k = apply_rope(q, k, pe)  # These are B H L D
    
    # Compute attention scores
    scale = (q.size(-1)) ** -0.5
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # B H L L
    
    # Apply block mask if provided
    if attn_mask is not None:
        # Convert block mask to dense mask using vectorized computation
        B, H, L, _ = scores.shape
        mask_dense = compute_dense_mask_from_block_mask(attn_mask, L, scores.device)
        
        # Apply mask (set masked positions to large negative value)
        scores = scores.masked_fill(~mask_dense.unsqueeze(0).unsqueeze(0), float('-inf'))
    
    # Compute attention weights
    attn_weights = torch.softmax(scores, dim=-1)  # B H L L
    
    # Apply attention to values
    x = torch.matmul(attn_weights, v)  # B H L D
    x = rearrange(x, "B H L D -> B L (H D)")
    
    return x, attn_weights


def attention(q: Tensor, k: Tensor, v: Tensor, pe: Tensor, attn_mask = None, score_mod: Optional[Callable] = None) -> Tensor:
    q, k = apply_rope(q, k, pe) # These are B H L D

    x = flex_attention(q, k, v, block_mask=attn_mask, score_mod=score_mod)
    x = rearrange(x, "B H L D -> B L (H D)")

    return x

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor):
        x_dtype = x.dtype
        x = x.float()
        rrms = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-6)
        return (x * rrms).to(dtype=x_dtype) * self.scale


class QKNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q.to(v), k.to(v)


class EncoderBlock(nn.Module):
    """
    A DiT block with parallel linear layers as described in
    https://arxiv.org/abs/2302.05442 but with the modulation torn out
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qk_scale: float | None = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_size
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        # qkv and mlp_in
        self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim)
        # proj and mlp_out
        self.linear2 = nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size)

        self.norm = QKNorm(head_dim)

        self.hidden_size = hidden_size
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.mlp_act = nn.GELU(approximate="tanh")

    def forward(self, x: Tensor, pe: Tensor, mask: Optional[BlockMask] = None, score_mod: Optional[Callable] = None, return_attention: bool = False) -> Tensor | tuple[Tensor, Tensor]:
        """
        Args:
            x: Input tensor of shape (B, L, D)
            pe: Positional embeddings
            mask: Optional block mask for attention
            score_mod: Optional score modification function
            return_attention: Whether to return attention weights for visualization
            
        Returns:
            If return_attention=False: Output tensor of shape (B, L, D)
            If return_attention=True: Tuple of (output_tensor, attention_weights)
                - output_tensor: Shape (B, L, D)
                - attention_weights: Shape (B, H, L, L) where H is num_heads
        """
        qkv, mlp = torch.split(self.linear1(self.pre_norm(x)), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1)

        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)

        # compute attention
        if return_attention:
            assert score_mod is None, "Score mod not supported for return_attention"
            attn, attn_weights = explicit_attention(q, k, v, pe=pe, attn_mask=mask)
        else:
            attn = attention(q, k, v, pe=pe, attn_mask=mask, score_mod=score_mod)
            attn_weights = None
        
        # compute activation in mlp stream, cat again and run second linear layer
        output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
        
        if return_attention:
            return x + output, attn_weights
        else:
            return x + output
    
@lru_cache
def get_decoder_mask(seq_len: int, memory_len: int) -> Tensor:
    def memory_mask(b, h, q_idx, kv_idx):
        # returns true if it's "active" to attend to
        return q_idx < seq_len
    block_mask = create_block_mask(memory_mask, B=None, H=None, Q_LEN = seq_len + memory_len, KV_LEN = seq_len + memory_len, _compile=True)
    return block_mask

class TransformerEncoder(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        mlp_ratio: float,
        num_heads: int,
        depth: int,
        axes_dim: list[int],
        theta: int = 10_000,
    ):
        super().__init__()

        if hidden_size % num_heads != 0:
            raise ValueError(
                f"Hidden size {hidden_size} must be divisible by num_heads {num_heads}"
            )
        # axes_dim specifies how to split up the rope embedding across each **head**, and so it must sum to (hidden_size//num_heads)
        pe_dim = hidden_size // num_heads
        if sum(axes_dim) != pe_dim:
            raise ValueError(f"Got {axes_dim} but expected positional dim {pe_dim}")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.pe_embedder = EmbedND(dim=pe_dim, theta=theta, axes_dim=axes_dim)

        self.blocks = nn.ModuleList(
            [
                EncoderBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
                for _ in range(depth)
            ]
        )
    
    def forward(
        self,
        x: Tensor,
        ids: Tensor,
        mask: Optional[BlockMask] = None,
        score_mod: Optional[Callable] = None,
        return_attention: bool = False,
    ) -> Tensor | tuple[Tensor, list[Tensor]]:
        """
        Args:
            x: Input tensor of shape (B, L, D)
            ids: Position IDs for positional embeddings
            mask: Optional block mask for attention
            score_mod: Optional score modification function
            return_attention: Whether to return attention weights from all layers
            
        Returns:
            If return_attention=False: Output tensor of shape (B, L, D)
            If return_attention=True: Tuple of (output_tensor, attention_weights_list)
                - output_tensor: Shape (B, L, D)
                - attention_weights_list: List of attention weights from each layer,
                  each with shape (B, H, L, L) where H is num_heads
        """
        pe = self.pe_embedder(ids)
        
        attention_weights = []
        
        for block in self.blocks:
            if return_attention:
                # Cannot use checkpoint with return_attention since it changes the return signature
                result = block(x, pe, mask, score_mod, return_attention=True)
                x, attn_weights = result
                attention_weights.append(attn_weights)
            else:
                # Only use checkpoint if gradients are required (training mode)
                if x.requires_grad and self.training:
                    x = checkpoint(block, x, pe, mask, score_mod, use_reentrant=False)
                else:
                    x = block(x, pe, mask, score_mod)
        
        if return_attention:
            return x, attention_weights
        else:
            return x