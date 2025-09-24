import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from torch.nn.attention.flex_attention import create_block_mask
from torch import Tensor
from functools import lru_cache
from einops import rearrange, repeat
from typing import Union
from eye.foveal_encoders import MultiScaleDino
from eye.transformer import TransformerEncoder
from eye.agent_configs import EyeRobotAgentConfig
from eye.sim.demo_data import DemonstrationData

@lru_cache
def get_eye_block_mask(seq_len: int, feats_per_t: int, window_len: int, act_size: int, img_feat_size: int, decoder_only: bool = False) -> Tensor:
    def memory_mask(b, h, q_idx, kv_idx):
        # returns true if it's "active" to attend to
        # This is a block causal mask in terms of time
        q_t = q_idx // feats_per_t
        kv_t = kv_idx // feats_per_t
        q_t_mod = q_idx % feats_per_t
        kv_t_mod = kv_idx % feats_per_t

        # Calculate token type boundaries within a timestep
        # Order: learn(2), eye(1), target(1), proprio(1), joint(act_size), features(img_feat_size)
        proprio_idx_mod = 2 + 1 + 1 # Index of proprio token (learn+eye+target)
        img_start_mod = feats_per_t - img_feat_size
        joint_start_mod = img_start_mod - act_size

        # --- Basic temporal causality (within window) ---
        causal_mask = (q_t >= kv_t) & (q_t - kv_t < window_len)

        # --- Token Type Identification ---
        is_q_img = q_t_mod >= img_start_mod
        is_kv_img = kv_t_mod >= img_start_mod
        is_q_joint = (q_t_mod >= joint_start_mod) & (q_t_mod < img_start_mod)
        is_kv_joint = (kv_t_mod >= joint_start_mod) & (kv_t_mod < img_start_mod)
        is_kv_proprio = (kv_t_mod == proprio_idx_mod)

        # --- Masking Rules ---
        # 1. Images cannot attend to other images
        img_to_img_mask = ~(is_q_img & is_kv_img)

        # 2. Joint tokens cannot attend to past joint tokens (only current timestep)
        same_timestep = (q_t == kv_t)
        joint_to_past_joint_mask = ~(is_q_joint & is_kv_joint & ~same_timestep)

        # 3. ANY tokens at t cannot attend to proprioception tokens before t
        proprio_to_past_proprio_mask = ~(is_kv_proprio & (kv_t < q_t))

        # 4. No tokens can attend to past image tokens 
        all_to_past_image_mask = ~(is_kv_img & (kv_t < q_t))
        
        # 5. Image tokens cannot attend to non-image tokens (decode only from image tokens)
        is_kv_non_img = ~is_kv_img
        img_to_non_img_mask = ~(is_q_img & is_kv_non_img) if decoder_only else True
        
        # Combine all masks
        final_mask = (
            causal_mask
            & img_to_img_mask
            & joint_to_past_joint_mask
            & proprio_to_past_proprio_mask
            & all_to_past_image_mask
            & img_to_non_img_mask
        )
        return final_mask

    # _compile is IMPORTANT to save memory on gpu
    block_mask = create_block_mask(memory_mask, B=None, H=None, Q_LEN = seq_len, KV_LEN = seq_len, _compile = True)
    return block_mask


@lru_cache
def get_hand_block_mask(seq_len, feats_per_t, img_feat_size, decoder_only: bool = False):
    # Create a simple causal mask (no need for complex mask since we have fewer token types)
    # order is [proprio_tokens(1), eye_tokens(1), target_tokens(1),joint_tokens(act_size), features(img_feat_size)]
    def memory_mask(b, h, q_idx, kv_idx):
        q_t = q_idx // feats_per_t
        kv_t = kv_idx // feats_per_t
        q_t_mod = q_idx % feats_per_t
        kv_t_mod = kv_idx % feats_per_t
        
        # Simple causal masking with window length 1
        causal_mask = (q_t >= kv_t) & (q_t - kv_t < 1)
        
        # Image tokens cannot attend to other image tokens
        is_q_img = q_t_mod >= (feats_per_t - img_feat_size)
        is_kv_img = kv_t_mod >= (feats_per_t - img_feat_size)
        img_to_img_mask = ~(is_q_img & is_kv_img)
        
        # Image tokens cannot attend to non-image tokens (decode only from image tokens)
        is_kv_non_img = ~is_kv_img
        img_to_non_img_mask = ~(is_q_img & is_kv_non_img) if decoder_only else True
        
        return causal_mask & img_to_img_mask & img_to_non_img_mask
    
    block_mask = create_block_mask(memory_mask, B=None, H=None, Q_LEN=seq_len, KV_LEN=seq_len, _compile=True)
    return block_mask


class EyeRobotAgent(nn.Module):
    def __init__(
        self,
        config: Union[EyeRobotAgentConfig, dict, None] = None,
        **kwargs
    ):
        super().__init__()
        
        # Handle config initialization
        if config is None:
            self.config = EyeRobotAgentConfig(**kwargs)
        elif isinstance(config, dict):
            self.config = EyeRobotAgentConfig.from_dict(config)
        else:
            self.config = config
        # Set instance variables from config
        self.window_length = self.config.window_length
        self.action_max = np.deg2rad(10)  # Maximum action magnitude for continuous actions
        self.action_chunk_size = self.config.action_chunk_size
        self.proprio_dropout = self.config.proprio_dropout
        self.proprio_hidden_dropout = self.config.proprio_hidden_dropout
        self.use_learnable_joint_token = self.config.use_learnable_joint_token
        self.relative_act = self.config.relative_act
        self.use_se3_actions = self.config.use_se3_actions
        self.se3_mode = self.config.use_se3_actions
        self.decoder_only = self.config.decoder_only
        
        self.eye_magnitudes = self.config.eye_magnitudes
        
        # Initialize vision transformer based on vit_type
        # we only manually apply if its not a my_encoder, otherwise we use rope
        self.feature_extractor = MultiScaleDino(
            crop_sizes=self.config.crop_sizes, 
            window_size=self.config.window_size, 
            device=self.config.device,
            pool_size=self.config.pool_size,
            freeze_encoder=self.config.freeze_encoder
        )
        tok_dimension = 384  # DINO feature dimension # was 384
        self.tok_dim = tok_dimension # Store tok_dimension
        self.target_proj = nn.Linear(512, tok_dimension)

        self.transformer = TransformerEncoder(
            hidden_size=tok_dimension,
            mlp_ratio=4.0,
            num_heads=12, 
            depth=self.config.num_blocks,
            axes_dim=[8,12,12],
            theta = 1000
        )
        
        # Hand transformer for processing joint actions
        self.hand_transformer = TransformerEncoder(
            hidden_size=tok_dimension,
            mlp_ratio=4.0,
            num_heads=12,
            depth=self.config.num_blocks,
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
        self.proprio_proj = nn.Sequential(
            nn.Linear(7, tok_dimension),
            nn.LayerNorm(tok_dimension),
            nn.GELU(approximate='tanh'),
            nn.Dropout(self.proprio_hidden_dropout),
            nn.Linear(tok_dimension, tok_dimension),
            nn.GELU(approximate='tanh'),
            nn.Dropout(self.proprio_hidden_dropout),
            nn.Linear(tok_dimension, tok_dimension),
        )
        self.learnable_tokens = nn.Parameter(torch.randn(1, 1, 2, tok_dimension))  # one for actor mean, one for critic
        self.eye_action_type = self.config.eye_action_type
        if self.config.eye_action_type == "continuous":
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
            magnitudes = [np.deg2rad(a) for a in self.eye_magnitudes]
            self.action_dict = [[np.sin(t), np.cos(t)] for t in np.linspace(0, 2*np.pi, 8, endpoint=False)]
            self.action_dict = torch.tensor(self.action_dict).float()
            self.action_dict /= torch.norm(self.action_dict, dim=-1, keepdim=True)
            self.action_dict = [m*self.action_dict for m in magnitudes] + [torch.zeros(1,2)]
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
        # Conditionally initialize joint_token
        if self.config.use_learnable_joint_token:
            self.joint_token = nn.Parameter(torch.randn(1, 1, 1, tok_dimension))
        else:
            self.joint_token = None # Or handle appropriately if needed elsewhere
        
        # Set output dimension based on SE3 vs joint mode
        if self.se3_mode:
            joint_output_dim = 10  # 10-DOF SE3 output (9-DOF SE3 + 1 gripper)
        else:
            joint_output_dim = 7  # 7-DOF joint output
            
        self.joint_actor = nn.Sequential(
            nn.LayerNorm(tok_dimension, elementwise_affine=False),
            nn.Linear(tok_dimension, 256),
            nn.LayerNorm(256),
            nn.GELU(approximate='tanh'),
            nn.Linear(256, 128),
            nn.GELU(approximate='tanh'),
            nn.Linear(128, joint_output_dim),
        )
    
    def get_eye_params(self):
        # Eye specific components
        eye_params = []
        eye_params.extend(self.transformer.parameters())
        eye_params.extend(self.actor.parameters())
        if self.config.eye_action_type == "continuous":
            eye_params.append(self.actor_logstd)
        eye_params.append(self.learnable_tokens)
        eye_params.extend(self.critic.parameters())
        # Common components (feature extractor, target projection)
        eye_params.extend(self.target_proj.parameters())
        return iter(eye_params)

    def get_hand_params(self):
        # Hand specific components
        hand_params = list(self.hand_transformer.parameters())
        hand_params.extend(self.eye_proj.parameters())
        hand_params.extend(self.joint_actor.parameters())
        hand_params.extend(self.proprio_proj.parameters())
        hand_params.extend(self.feature_extractor.get_trainable_parameters())
        if self.config.use_learnable_joint_token:
            hand_params.append(self.joint_token)
        return iter(hand_params)

    def apply_correlated_dropout(self, x, dropout_rate, null_vector):
        """Apply correlated dropout where entire samples are zeroed out with probability dropout_rate"""
        if self.training and dropout_rate > 0:
            # x shape is (t*b, c) where c=7 for proprioception
            batch_size = x.shape[0]
            # Generate random numbers for each sample in the batch
            dropout_mask = torch.rand(batch_size, 1, device=x.device) < dropout_rate
            # Scale by 1/(1-dropout_rate) to maintain expected value
            x = torch.where(dropout_mask, null_vector, x)
        return x

    def _orthogonalize_se3_output(self, se3_raw):
        """
        Internal orthogonalization of SE3 output - always called within agent
        Input: [..., 10] where dims are [x, y, z, r11, r21, r31, r12, r22, r32, gripper]
        Output: [..., 10] with orthogonalized rotation columns
        """
        position = se3_raw[..., :3]  # [x, y, z]
        col1_raw = se3_raw[..., 3:6]  # [r11, r21, r31]
        col2_raw = se3_raw[..., 6:9]  # [r12, r22, r32]
        gripper = se3_raw[..., 9:10]  # [gripper]
        
        # Gram-Schmidt orthogonalization
        col1 = F.normalize(col1_raw, dim=-1)  # Normalize first column
        
        # Remove component of col1 from col2, then normalize
        col2 = col2_raw - torch.sum(col2_raw * col1, dim=-1, keepdim=True) * col1
        col2 = F.normalize(col2, dim=-1)
        
        return torch.cat([position, col1, col2, gripper], dim=-1)  # Return orthogonalized 10-DOF


    def _prepare_common_tokens(self, x):
        multicrop = x["multicrop"]  # shape: T, B, n_scales, 3, H, W
        T = multicrop.shape[0]
        B = multicrop.shape[1]
        device = multicrop.device

        # Process multicrop to get features
        if 'foveal_tokens' not in x:
            multicrop_reshaped = rearrange(multicrop, 't b n c h w -> (t b) n c h w')
            features = self.feature_extractor(multicrop_reshaped) # B, L, embed_dim which is b (n h w) c
            features = rearrange(features, '(t b) l d -> b t l d', t=T)
            x['foveal_tokens'] = rearrange(features, 'b t l d -> t b l d', t=T)
        else:
            available_t = x['foveal_tokens'].shape[0]
            if available_t < T:
                new_feats = self.feature_extractor(multicrop[-1])
                new_feats = rearrange(new_feats, 'b l d -> 1 b l d')
                x['foveal_tokens'] = torch.cat([x['foveal_tokens'], new_feats], dim=0)
            features = rearrange(x['foveal_tokens'], 't b l d -> b t l d', t=T)
        img_feat_size = features.shape[2] # Usually 1024
        feat_ids = self.feature_extractor.rope_ids.unsqueeze(0).repeat(B, T, 1, 1)

        # Process eye tokens
        eye_token_input = x["eye_direction"]
        eye_token_input = rearrange(eye_token_input, 't b c -> (t b) c')
        eye_tokens = self.eye_proj(eye_token_input).unsqueeze(1) # B, 1, embed_dim
        eye_tokens = rearrange(eye_tokens, '(t b) l d -> b t l d', t=T)
        eye_ids = torch.full((B, T, 1, 2), 960.0, device=device)

        # Process proprio tokens
        proprio_input = x["proprio"]
        proprio_input = DemonstrationData.normalize_joint_data(proprio_input)
        proprio_input = rearrange(proprio_input, 't b c -> (t b) c')
        # Apply correlated dropout where n% of the time we zero out entire samples
        proprio_input = self.apply_correlated_dropout(proprio_input, self.proprio_dropout, torch.zeros((1,7), device=proprio_input.device)).unsqueeze(1) # B, 1, embed_dim
        proprio_tokens = self.proprio_proj(proprio_input) # B, 1, embed_dim
        proprio_tokens = rearrange(proprio_tokens, '(t b) l d -> b t l d', t=T)
        proprio_ids = torch.full((B, T, 1, 2), 960.0, device=device)

        # Process target tokens
        target_tokens_input = rearrange(x["target_clip_vec"], 't b c -> (t b) c')
        target_token = self.target_proj(target_tokens_input).unsqueeze(1) # B, 1, embed_dim
        target_token = rearrange(target_token, '(t b) l d -> b t l d', t=T)
        target_ids = torch.full((B, T, 1, 2), 960.0, device=device)

        # Prepare learnable tokens
        learn_tokens = self.learnable_tokens.repeat(B, T, 1, 1)
        learn_ids = torch.full((B, T, self.learnable_tokens.shape[2], 2), 960.0, device=device)

        return {
            "features": features, "feat_ids": feat_ids, "img_feat_size": img_feat_size,
            "eye_tokens": eye_tokens, "eye_ids": eye_ids,
            "proprio_tokens": proprio_tokens, "proprio_ids": proprio_ids,
            "target_token": target_token, "target_ids": target_ids,
            "learn_tokens": learn_tokens, "learn_ids": learn_ids,
            "B": B, "T": T, "device": device
        }

    def run_hand_transformer(self, _, common_data):
        # common_data = self._prepare_common_tokens(x) # Removed: common_data is now passed in
        features = common_data["features"]
        feat_ids = common_data["feat_ids"]
        img_feat_size = common_data["img_feat_size"]
        if self.config.n_hand_levels < self.config.n_levels:
            # prune off levels from features, feat_ids, and update img_feat_size accordingly
            # TODO make 8 not hardcoded
            features = rearrange(features, 'b t (n h w) d -> b t n h w d', n=self.config.n_levels, h = 16, w = 16)
            features = features[:, :, :self.config.n_hand_levels] # foveal crops are the 0th index
            features = rearrange(features, 'b t n h w d -> b t (n h w) d')
            feat_ids = rearrange(feat_ids, 'b t (n h w) d -> b t n h w d', n=self.config.n_levels, h = 16, w = 16)
            feat_ids = feat_ids[:, :, :self.config.n_hand_levels]
            feat_ids = rearrange(feat_ids, 'b t n h w d -> b t (n h w) d')
            img_feat_size = features.shape[2]
        eye_tokens = common_data["eye_tokens"]
        eye_ids = common_data["eye_ids"]
        proprio_tokens = common_data["proprio_tokens"]
        proprio_ids = common_data["proprio_ids"]
        target_tokens = common_data["target_token"]
        target_ids = common_data["target_ids"]
        B = common_data["B"]
        T = common_data["T"]
        device = common_data["device"]
        
        # Create input tokens for joint action slots
        if self.config.use_learnable_joint_token:
            joint_tokens = self.joint_token.repeat(B, T, self.action_chunk_size, 1)
        else:
            # Use repeated proprio tokens as input for the joint action slots
            joint_tokens = proprio_tokens.detach().clone().repeat(1, 1, self.action_chunk_size, 1)
        joint_ids = torch.full((B, T, self.action_chunk_size, 2), 960.0, device=device)
        
        # Concatenate all tokens for hand transformer - only including image, proprio, eye, and joint tokens
        tokens = torch.cat([proprio_tokens, eye_tokens, target_tokens, joint_tokens, features], dim=2) # B, T, big_L, embed_dim
        ids = torch.cat([proprio_ids, eye_ids, target_ids, joint_ids, feat_ids], dim=2).div_(14.0) # B, T, big_L, 2
        
        # Create time_ids
        time_ids = torch.arange(T, device=device).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        time_ids = time_ids.repeat(B, 1, tokens.shape[2], 1)
        
        # Calculate joint token start index
        joint_start_tok_idx = 1 + 1  # After proprio and eye tokens
        
        # Create future time ids for joint slots
        future_time_ids = rearrange(torch.arange(self.action_chunk_size, device=device), 'l -> 1 1 l 1')
        future_time_ids = repeat(future_time_ids, '1 1 l 1 -> b t l 1', b=B, t=T)
        # Add offset of arange(T) to future_time_ids along T dimension
        future_time_ids = future_time_ids + rearrange(torch.arange(T, device=device), 't -> 1 t 1 1')
        
        # Assign future time ids to joint token slots
        time_ids[:,:, joint_start_tok_idx:joint_start_tok_idx + self.action_chunk_size, :] = future_time_ids
        
        # Finalize IDs and reshape
        ids = torch.cat([time_ids, ids], dim=-1) # Add time_id as first dimension
        ids = rearrange(ids, 'b t l d -> b (t l) d') # d becomes 3 (time, x, y)
        tokens = rearrange(tokens, 'b t l d -> b (t l) d')
        
        feats_per_t = tokens.shape[1] // T
        block_mask = get_hand_block_mask(tokens.shape[1], feats_per_t, img_feat_size, self.decoder_only)
        
        transformed = self.hand_transformer(tokens, ids, block_mask)
        transformed = rearrange(transformed, 'b (t l) d -> t b l d', t=T)
        
        return transformed, feats_per_t
    
    def run_eye_transformer(self, x, common_data, window_length=None):
        # common_data = self._prepare_common_tokens(x) # Removed: common_data is now passed in
        features = common_data["features"]
        feat_ids = common_data["feat_ids"]
        img_feat_size = common_data["img_feat_size"]
        multicrop = x["multicrop"]  # shape: T, B, n_scales, 3, H, W
        T = multicrop.shape[0]
        B = multicrop.shape[1]
        win_len = window_length if window_length is not None else self.window_length
        assert len(multicrop.shape) == 6, (
            f"Multicrop must have shape (T, B, n_scales, 3, H, W), got {multicrop.shape}"
        )
        # rope_ids are 1,L_img,2

        # At this point features are B, T, L, embed_dim
        eye_tokens = common_data["eye_tokens"].detach()
        eye_ids = common_data["eye_ids"]

        proprio_tokens = common_data["proprio_tokens"].detach()
        proprio_ids = common_data["proprio_ids"]

        learn_tokens = common_data["learn_tokens"]
        learn_ids = common_data["learn_ids"]

        target_token = common_data["target_token"]
        target_ids = common_data["target_ids"]

        # Order: learn, eye, target, proprio, features
        tokens_to_cat = [learn_tokens, eye_tokens, target_token, proprio_tokens, features]
        ids_to_cat = [learn_ids, eye_ids, target_ids, proprio_ids, feat_ids]

        tokens = torch.cat(tokens_to_cat, dim=2) #B, T, big_L, embed_dim
        ids = torch.cat(ids_to_cat, dim=2).div_(14.0) # B, T, big_L, 2

        # time_ids should be B, T, 1, 1
        time_ids = torch.arange(T, device=features.device).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        time_ids = time_ids.repeat(B, 1, tokens.shape[2], 1)


        # Finalize IDs and reshape
        ids = torch.cat([time_ids, ids], dim=-1) # Add time_id as the first dimension
        ids = rearrange(ids, 'b t l d -> b (t l) d') # d becomes 3 (time, x, y)
        tokens = rearrange(tokens, 'b t l d -> b (t l) d')

        feats_per_t = tokens.shape[1] // T

        # Create a modified block mask that doesn't include joint tokens
        # For simplicity, we can reuse the same function but with joint chunk size set to 0
        block_mask = get_eye_block_mask(
            tokens.shape[1], feats_per_t, win_len, 0, img_feat_size, self.decoder_only
        )
        transformed = self.transformer(tokens, ids, block_mask)
        transformed = rearrange(transformed, 'b (t l) d -> t b l d', t=T)
        # Return transformed features and feats_per_t
        return transformed, feats_per_t

    def get_action_and_value(self, x, action=None, deterministic=False, window_length=None, inference=False):
        # Prepare common tokens once
        common_data = self._prepare_common_tokens(x)
        
        # Get features and feats_per_t from extract_features, passing common_data
        eye_features, eye_feats_per_t = self.run_eye_transformer(x, common_data, window_length=window_length) # T B L embed_dim
        
        # Get features from hand transformer separately, passing common_data
        if inference:
            # chop of all the time tokens for hand transformer and only leave last one
            for k in common_data:
                if isinstance(common_data[k], torch.Tensor):
                    common_data[k] = common_data[k][:,-1:]
            common_data['T'] = 1
            x['proprio'] = x['proprio'][-1:]
        hand_features, hand_feats_per_t = self.run_hand_transformer(x, common_data) # T B L embed_dim
        
        # Eye Actor (uses first learnable token from eye transformer)
        actor_toks = eye_features[:,:,0]
        if self.config.eye_action_type == "continuous":
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

        # Joint Actor (uses joint tokens' outputs from hand transformer)
        # The joint tokens start after proprio and eye tokens in hand_features
        joint_start_tok_idx = 2 # After proprio and eye tokens
        joint_actions_raw = self.joint_actor(hand_features[:,:,joint_start_tok_idx : joint_start_tok_idx + self.action_chunk_size])
        
        # Apply internal orthogonalization if in SE3 mode
        if self.se3_mode:
            joint_actions = self._orthogonalize_se3_output(joint_actions_raw.float())  # Return 9-DOF SE3
        else:
            joint_actions = joint_actions_raw  # Return 7-DOF joint angles
        # Critic (uses second learnable token from eye transformer)
        value = self.critic(eye_features[:,:,1])

        ret = (
            action,
            log_prob,
            entropy,
            value,
            joint_actions, # T, B, action_chunk_size, 7
        )
        return ret

    def get_value(self, x, window_length=None):
        # Prepare common tokens once
        common_data = self._prepare_common_tokens(x)
        # Pass common_data to run_eye_transformer
        features, _ = self.run_eye_transformer(x, common_data, window_length=window_length)
        return self.critic(features[:,:,1])

    def forward(self, x, action=None, deterministic=False, window_length=None, only_value = False):
        if only_value:
            return self.get_value(x, window_length)
        else:
            return self.get_action_and_value(x, action, deterministic, window_length)