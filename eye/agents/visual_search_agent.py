import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from torch.nn.attention.flex_attention import create_block_mask
from torch import Tensor
from functools import lru_cache
from einops import rearrange
from typing import Union
from eye.foveal_encoders import MultiScaleDino
from eye.transformer import TransformerEncoder
from eye.agent_configs import VisualSearchAgentConfig


@lru_cache
def get_vis_agent_block_mask(seq_len: int, feats_per_t: int, window_len: int, img_feat_size: int, mask_past_img_attn: bool = True, decoder_only: bool = False) -> Tensor:
    def memory_mask(b, h, q_idx, kv_idx):
        # returns true if it's "active" to attend to
        # This is a block causal mask in terms of time
        q_t = q_idx // feats_per_t
        kv_t = kv_idx // feats_per_t
        # tokens are torch.cat([learn_tokens, eye_tokens, target_token, features], dim=2) #B, T, big_L, 384
        # therefore the final 1024 tokens are the image tokens
        q_t_mod = q_idx % feats_per_t
        kv_t_mod = kv_idx % feats_per_t
        is_q_img = (q_t_mod >= feats_per_t - img_feat_size) # Identify if query is an image token
        is_kv_img = (kv_t_mod >= feats_per_t - img_feat_size) # Identify if key/value is an image token
        
        # 1. Causal mask within the window
        causal_mask = (q_t >= kv_t) & (q_t - kv_t < window_len)
        # 2. Prevent image tokens from attending to any other image tokens (past or present)
        img_to_img_mask = ~(is_q_img & is_kv_img)
        # 3. Prevent any token from attending to image tokens from past timesteps
        if mask_past_img_attn:
            all_to_past_image_mask = ~(is_kv_img & (kv_t < q_t))
        else:
            all_to_past_image_mask = True

        # 4. Image tokens cannot attend to non-image tokens (decode only from image tokens)
        is_kv_non_img = ~is_kv_img
        img_to_non_img_mask = ~(is_q_img & is_kv_non_img) if decoder_only else True

        return causal_mask & img_to_img_mask & all_to_past_image_mask & img_to_non_img_mask
    # _compile is IMPORTANT to save memory on gpu
    block_mask = create_block_mask(memory_mask, B=None, H=None, Q_LEN = seq_len, KV_LEN = seq_len, _compile = True)
    return block_mask


class VisualSearchAgent(nn.Module):
    def __init__(
        self,
        config: Union[VisualSearchAgentConfig, dict, None] = None,
        **kwargs
    ):
        super().__init__()

        # Handle config initialization
        if config is None:
            self.config = VisualSearchAgentConfig(**kwargs)
        elif isinstance(config, dict):
            self.config = VisualSearchAgentConfig.from_dict(config)
        else:
            self.config = config
            
        # Set instance variables from config
        self.window_length = self.config.window_length
        self.action_type = self.config.action_type
        self.action_max = np.deg2rad(10)  # Maximum action magnitude for continuous actions
        self.decoder_only = self.config.decoder_only
        
        self.magnitudes = self.config.magnitudes
        
        # Initialize vision transformer based on vit_type
        # we only manually apply if its not a my_encoder, otherwise we use rope
        self.feature_extractor = MultiScaleDino(
            crop_sizes=self.config.crop_sizes, 
            window_size=self.config.window_size, 
            device=self.config.device,
            pool_size=self.config.pool_size,
            freeze_encoder=self.config.freeze_encoder
        )
        tok_dimension = 384  # DINO feature dimension
        self.target_proj = nn.Linear(512, tok_dimension)

        self.transformer = TransformerEncoder(
            hidden_size=tok_dimension,
            mlp_ratio=4.0,
            num_heads=12,
            depth=self.config.n_blocks,
            axes_dim=[8,12,12],
            theta = 1000
        )
        
        self.eye_proj = nn.Sequential(
            nn.Linear(3, tok_dimension),
            nn.LayerNorm(tok_dimension),
            nn.GELU(approximate='tanh'),
            nn.Linear(tok_dimension, tok_dimension),
            nn.GELU(approximate='tanh'),
            nn.Linear(tok_dimension, tok_dimension),
        )
        self.learnable_tokens = nn.Parameter(torch.randn(1, 1, 2 + self.config.learnable_registers, tok_dimension))  # one for actor mean, one for critic, and learnable_registers for learnable registers
        if self.action_type == "continuous":
            self.actor = nn.Sequential(
                nn.LayerNorm(tok_dimension, elementwise_affine=False),
                nn.Linear(tok_dimension, 256),
                nn.LayerNorm(256),
                nn.GELU(approximate='tanh'),
                nn.Linear(256, 256),
                nn.GELU(approximate='tanh'),
                nn.Linear(256, 2)  # 2 for mean, 1 for std
            )
            self.actor_logstd = nn.Parameter(torch.log(torch.ones(1, 1) * self.action_max))
        else:
            magnitudes_rad = [np.deg2rad(a) for a in self.magnitudes]
            self.action_dict = [[np.sin(t), np.cos(t)] for t in np.linspace(0, 2*np.pi, 8, endpoint=False)]
            self.action_dict = torch.tensor(self.action_dict).float()
            self.action_dict /= torch.norm(self.action_dict, dim=-1, keepdim=True)
            self.action_dict = [m*self.action_dict for m in magnitudes_rad] + [torch.zeros(1,2)]
            # concat action dict with itself (minus 0 action) and with a small action_max/4
            self.action_dict = torch.cat(self.action_dict).to(self.config.device).bfloat16()
            self.actor = nn.Sequential(
                nn.LayerNorm(tok_dimension, elementwise_affine=False),
                nn.Linear(tok_dimension, 256),
                nn.LayerNorm(256),
                nn.GELU(approximate='tanh'),
                nn.Linear(256, 256),
                nn.GELU(approximate='tanh'),
                nn.Linear(256, len(self.action_dict)),
                nn.Softmax(dim=-1)
            )
        self.critic = nn.Sequential(
            nn.LayerNorm(tok_dimension, elementwise_affine=False),
            nn.Linear(tok_dimension, 256),
            nn.LayerNorm(256),
            nn.GELU(approximate='tanh'),
            nn.Linear(256, 256),
            nn.GELU(approximate='tanh'),
            nn.Linear(256, 1),
        )
        self.tok_dim = tok_dimension

    def extract_features(self, x, window_length=None, attention_hook=None):
        multicrop = x["multicrop"]  # shape: T, B, n_scales, 3, H, W
        T = multicrop.shape[0]
        B = multicrop.shape[1]
        win_len = window_length if window_length is not None else self.window_length
        assert len(multicrop.shape) == 6, (
            f"Multicrop must have shape (T, B, n_scales, 3, H, W), got {multicrop.shape}"
        )
        if 'foveal_tokens' not in x:
            multicrop = rearrange(multicrop, 't b n c h w -> (t b) n c h w')
            features = self.feature_extractor(multicrop) # B, L, 384
            features = rearrange(features, '(t b) l d -> b t l d', t=T)
            x['foveal_tokens'] = rearrange(features, 'b t l d -> t b l d', t=T)
        else:
            # if the number of cached timesteps is less than the window length, we need to compute the features for the missing timesteps
            available_t = x['foveal_tokens'].shape[0]
            if available_t < T:
                new_feats = self.feature_extractor(multicrop[-1]) 
                new_feats = rearrange(new_feats, 'b l d -> 1 b l d')
                x['foveal_tokens'] = torch.cat([x['foveal_tokens'], new_feats], dim=0)
            features = rearrange(x['foveal_tokens'], 't b l d -> b t l d', t=T)
        # rope_ids are 1,L_img,2
        feat_ids = self.feature_extractor.rope_ids.unsqueeze(0).repeat(B, T, 1, 1)

        # At this point features are B, T, L, 384
        eye_token = x["eye_direction"]
        eye_token = rearrange(eye_token, 't b c -> (t b) c')
        eye_tokens = self.eye_proj(eye_token).unsqueeze(1) # B, 1, 384
        eye_tokens = rearrange(eye_tokens, '(t b) l d -> b t l d', t=T)
        eye_ids = torch.full((B, T, 1, 2), 960.0, device=features.device)

        learn_tokens = self.learnable_tokens.repeat(features.shape[0], features.shape[1], 1, 1)
        learn_ids = torch.full((B, T, self.learnable_tokens.shape[2], 2), 960.0, device=features.device)

        target_tokens = rearrange(x["target_clip_vec"], 't b c -> (t b) c')
        target_token = self.target_proj(target_tokens).unsqueeze(1) # B, 1, 384
        target_token = rearrange(target_token, '(t b) l d -> b t l d', t=T)
        target_ids = torch.full((B, T, 1, 2), 960.0, device=features.device)

        tokens = torch.cat([learn_tokens, eye_tokens, target_token, features], dim=2) #B, T, big_L, 384
        # time_ids should be B, T, 1, 1
        time_ids = torch.arange(T, device=features.device).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        time_ids = time_ids.repeat(B, 1, tokens.shape[2], 1)
        ids = torch.cat([learn_ids, eye_ids, target_ids, feat_ids], dim=2).div_(14.0)
        ids = torch.cat([time_ids, ids], dim=-1)
        ids = rearrange(ids, 'b t l d -> b (t l) d')
        tokens = rearrange(tokens, 'b t l d -> b (t l) d')

        feats_per_t = tokens.shape[1] // T
        
        res = self.transformer(tokens, ids, get_vis_agent_block_mask(tokens.shape[1], feats_per_t, win_len, features.shape[2], self.config.mask_past_img_attn, self.decoder_only), return_attention=attention_hook is not None)
        if attention_hook is not None:
            transformed, attention = res
            assert type(attention) == list, "Attention hook must be a list"
            attention_hook.extend(attention)
        else:
            transformed = res
        transformed = rearrange(transformed, 'b (t l) d -> t b l d', t=T)
        return transformed 
    
    def get_action_and_value(self, x, action=None, deterministic=False, window_length=None, attention_hook=None):
        assert len(x["multicrop"].shape) == 6, (
            f"Multicrop must have shape (T, B, n_scales, 3, H, W), got {x['multicrop'].shape}"
        )
        assert len(x["eye_direction"].shape) == 3, (
            f"Eye direction must have shape (T, B, 3), got {x['eye_direction'].shape}"
        )
        assert len(x["target_clip_vec"].shape) == 3, (
            f"Target clip vec must have shape (T, B, 512), got {x['target_clip_vec'].shape}"
        )
        features = self.extract_features(x, window_length, attention_hook) # T B L 384
        actor_toks = features[:,:,0]
        if self.action_type == "continuous":
            actor_out = self.actor(actor_toks)
            action_mean = nn.functional.tanh(actor_out[..., :2]) * self.action_max
            action_std = torch.exp(self.actor_logstd.repeat(actor_out.shape[0], actor_out.shape[1], 2))
            probs = Normal(action_mean, action_std)
            if deterministic:
                action = action_mean
            if action is None:
                action = probs.sample()
            log_prob = probs.log_prob(action).sum(-1)
            entropy = probs.entropy().sum(-1)
        else:
            actor_out = self.actor(actor_toks)
            assert not actor_out.isnan().any(), "Actor output is NaN"
            assert not actor_out.isinf().any(), "Actor output is Inf"
            action_probs = Categorical(probs=actor_out)
            if deterministic:
                action = self.action_dict[action_probs.probs.argmax(dim=-1)]
            act_ids = None
            if action is None:
                act_ids = action_probs.sample()
                action = self.action_dict[act_ids]
            # action is T B 2
            # action_dict is N 2
            # Need to reshape for broadcasting:
            # action -> T B 1 2
            # action_dict -> 1 1 N 2
            if act_ids is None:
                with torch.no_grad():
                    action_expanded = action.unsqueeze(2)  # T B 1 2
                    action_dict_expanded = self.action_dict.unsqueeze(0).unsqueeze(0)  # 1 1 N 2
                    
                    act_matches = torch.all(torch.isclose(action_expanded, action_dict_expanded, rtol=1e-5, atol=1e-5), dim=-1)  # T B N
                    act_ids = torch.argmax(act_matches.float(), dim=-1)  # T B
            log_prob = action_probs.log_prob(act_ids)
            entropy = action_probs.entropy()
        ret = (
            action,
            log_prob,
            entropy,
            self.critic(features[:,:,1]),
        )
        return ret
    
    def get_value(self, x, window_length=None):
        features = self.extract_features(x, window_length)
        return self.critic(features[:,:,1])

    def forward(self, x, action=None, deterministic=False, window_length=None, only_value = False):
        if only_value:
            return self.get_value(x, window_length)
        else:
            return self.get_action_and_value(x, action, deterministic, window_length)