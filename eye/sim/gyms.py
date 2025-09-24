import torch
import numpy as np
import torch.nn.functional as F
from eye.transforms import SO3
from tqdm import tqdm
from typing import List, Dict, Optional, Any, Union
import open_clip
import torchvision
import random
import matplotlib.pyplot as plt
import math
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import glob
import os

from .spherical_video import SphericalVideo
from .rewards import ClipReward, ConstancyReward
from .demo_data import DemonstrationData, line_distance, compute_fk_batch


class EyeGym:
    """
    Simplified environment for eye movement simulation that works directly with tensors
    """

    def __init__(
        self,
        demo_path_dict: Dict[str, List],
        max_steps: int,
        negative_prompts: List[str],
        device="cuda",
        openclip_model=None,
        preprocess=None,
        tokenizer=None,
        constancy_alpha: float = 0.0,
        clip_alpha: float = 10.0,
        ema_alpha: float = 0.5,  # Add EMA alpha parameter
        # Video creation parameters
        video_width: int = 1920,
        video_height: int = 1200,
        K: torch.Tensor = None,
        dist_coeffs: torch.Tensor = None,
        is_fisheye: bool = False,
        crop_sizes: List[int] = None,
        window_size: int = 224,
        decoder_device: str = 'cpu',
    ):
        # Store demo path structure
        self.filepaths = demo_path_dict["filepaths"]
        self.positive_crops = demo_path_dict["positive_crops"]
        self.prompts = demo_path_dict["prompts"]
        assert len(self.filepaths) == len(self.positive_crops), "Number of filepaths must match number of positive_crops"
        assert len(self.filepaths) == len(self.prompts), "Number of filepaths must match number of prompts"

        self.max_steps = max_steps
        self.device = device
        self._eye_state = None
        self.constancy_alpha = constancy_alpha
        self.clip_alpha = clip_alpha
        self.t = None
        self.ema_alpha = ema_alpha  # Store EMA alpha

        # Store video creation parameters for later use
        self.video_width = video_width
        self.video_height = video_height
        self.K = K
        self.dist_coeffs = dist_coeffs
        self.is_fisheye = is_fisheye
        self.crop_sizes = crop_sizes
        self.window_size = window_size
        self.decoder_device = decoder_device

        # Store negative prompts for clip reward creation
        self.negative_prompts = negative_prompts

        if openclip_model is None:
            openclip_model, _, _ = open_clip.create_model_and_transforms(
                "ViT-B-16", pretrained="laion2b_s34b_b88k", device=device, precision='fp16'
            )
            openclip_model.eval()
            preprocess = torchvision.transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            )
            tokenizer = open_clip.get_tokenizer("ViT-B-16")

        # Store CLIP components for later use
        self.openclip_model = openclip_model
        self.preprocess = preprocess
        self.tokenizer = tokenizer

        # Pre-cache text embeddings for all prompts (like DemoEyeGym)
        self.cached_prompt_embeddings = {}
        with torch.no_grad():
            for task_prompts in self.prompts:
                for prompt in task_prompts:  # Always a list now
                    if prompt not in self.cached_prompt_embeddings:
                        tokens = tokenizer([prompt]).to(self.device)
                        text_features = openclip_model.encode_text(tokens)
                        text_features /= text_features.norm(dim=-1, keepdim=True)
                        self.cached_prompt_embeddings[prompt] = text_features

        # Initialize variables that will be set in reset()
        self.video = None
        self.task_idx = None
        self.selected_prompt = None
        self.clip_reward = None
        self.constancy_reward = ConstancyReward(device)

    @property
    def eye_so3(self):
        return SO3.from_z_radians(self._eye_state[:,0]) @ SO3.from_x_radians(self._eye_state[:,1])

    @torch.inference_mode()
    def step(self, action: torch.Tensor):
        """
        Apply action to update environment state
        Args:
            action: tensor [d_azimuth, d_elev] in radians
        Returns:
            obs: dict with 'multicrop' and 'eye_direction' tensors
                    'multicrop' shape: (4, 3, 224, 224)
                    'eye_direction' shape: (3,)
            reward: tensor scalar
            terminated: bool
            truncated: bool
        """
        torch.cuda.synchronize() # this is critical for some reason to prevent NaN
        # I have no idea why... but adding this somehow fixes it.
        term1 = self._smoothed_action * self.ema_alpha
        converted_action = action.contiguous().to(self._smoothed_action.dtype)
        term2 = converted_action * (1.0 - self.ema_alpha)
        self._smoothed_action = term1 + term2
        # Debug: Check for NaN in smoothed action
        if self._smoothed_action.isnan().any():
            print(f"[DEBUG] NaN detected in smoothed_action: {self._smoothed_action}")
            print(f"[DEBUG] Raw action: {action}")
            print(f"[DEBUG] smoothed_action dtype: {self._smoothed_action.dtype}")
            print(f"[DEBUG] smoothed_action device: {self._smoothed_action.device}")
            print(f"[DEBUG] smoothed_action.isnan(): {self._smoothed_action.isnan()}")
            print(f"[DEBUG] smoothed_action.isfinite(): {self._smoothed_action.isfinite()}")
            print(f"[DEBUG] smoothed_action.isinf(): {self._smoothed_action.isinf()}")
            print(f"[DEBUG] Action dtype: {action.dtype}")
            print(f"[DEBUG] _smoothed_action dtype: {self._smoothed_action.dtype if self._smoothed_action is not None else 'None'}")
            print(f"[DEBUG] ema_alpha: {self.ema_alpha}")
            breakpoint()

        # Update eye state using smoothed action
        self._eye_state = self._eye_state + self._smoothed_action

        # clip elevation to [-np.pi,0]
        self._eye_state[:,1] = torch.clip(self._eye_state[:,1], -np.pi, 0)
        # clip azimuth
        self._eye_state[:,0] = torch.clip(self._eye_state[:,0], -np.pi/2 - np.deg2rad(130), -np.pi/2 + np.deg2rad(130))

        # Get observation
        eye_so3 = self.eye_so3

        multicrop_render = self.video.render_multicrop(eye_so3).unsqueeze(0)

        assert eye_so3.as_matrix().shape == (1, 3, 3), f"Eye state matrix shape: {eye_so3.as_matrix().shape}"
        eye_vec = eye_so3.as_matrix()[:, :, 2]

        # Use cached prompt embeddings for the selected prompt
        obs = {"multicrop": multicrop_render, "eye_direction": eye_vec, "target_clip_vec": self.cached_prompt_embeddings[self.selected_prompt]}

        self.t += 1 
        terminated = False
        truncated = False#not self.video.advance() or self.t >= self.max_steps

        # index multicrop by [0] to get rid of batch dimension
        # the multicrop is organized foveal->periphery, ie index 0 is the foveal crop
        constancy_reward = self.constancy_reward(multicrop_render[0,0]) if self.constancy_alpha > 0 else torch.tensor(0.0, device=self.device) 
        if self.target_so3 is not None:
            R1 = self.target_so3.as_matrix().float()
            R2 = eye_so3.as_matrix().float()
            vec = torch.tensor([0.0, 0.0, 1.0], device=R1.device).expand(1, -1)
            v1 = (R1 @ vec.unsqueeze(-1)).squeeze(-1) # Target viewing vector
            v2 = (R2 @ vec.unsqueeze(-1)).squeeze(-1) # Current viewing vector
            cos_sim = (v1 * v2).sum(-1).clamp(-1.0, 1.0).mean() # Cosine similarity
            # convert the cosine similarity to angle difference
            angle_diff = torch.acos(cos_sim)
            
            # Scale reward: 1 when angle_diff = 0, 0 when angle_diff >= 50 degrees
            max_angle = np.deg2rad(50)
            reward = torch.clamp(1.0 - (angle_diff / max_angle), 0.0, 1.0)
        else:
            reward = torch.tensor(0.0, device=self.device)
        reward = reward + self.constancy_alpha * constancy_reward
        
        
        return obs, reward.to(self.device), terminated, truncated, {}

    def reset(self):
        """
        Reset environment to initial state
        """
        # Reset step counter
        self.t = 0

        # Close previous video if it exists
        if self.video is not None:
            self.video.close()

        # First randomly sample a task (prompt_idx like in DemoEyeGym)
        self.task_idx = random.randint(0, len(self.prompts) - 1)

        # Then randomly sample a demo filepath for that task
        demo_idx = random.randint(0, len(self.filepaths[self.task_idx]) - 1)
        demo_path = self.filepaths[self.task_idx][demo_idx]

        # Find the actual video file within the demo directory (like DemoEyeGym)
        video_path = glob.glob(os.path.join(demo_path, "downsampled.mp4"))[0]

        # Create video from selected video path
        self.video = SphericalVideo(
            Path(video_path),
            self.K,
            self.dist_coeffs,
            self.video_height,
            self.video_width,
            self.is_fisheye,
            self.device,
            crop_sizes=self.crop_sizes,
            decoder_device=self.decoder_device,
            window_size=self.window_size
        )
        self.video.reset(random_time=True)

        # Randomly select the same index for both positive crop and prompt (they correspond)
        task_positive_crop_options = self.positive_crops[self.task_idx]
        task_prompt_options = self.prompts[self.task_idx]

        # Select same index for both crop and prompt
        option_idx = random.randint(0, len(task_positive_crop_options) - 1)
        selected_positive_crops = task_positive_crop_options[option_idx]
        self.selected_prompt = task_prompt_options[option_idx]

        # Create clip rewards for this specific task
        # Use selected positive crops for the task and all negative prompts
        all_other_crops = []
        for i, crop_options in enumerate(self.positive_crops):
            if i != self.task_idx:
                for crop_option in crop_options:
                    all_other_crops.extend(crop_option)

        self.clip_reward = ClipReward(
            selected_positive_crops,
            self.negative_prompts + all_other_crops,
            self.openclip_model,
            self.preprocess,
            self.tokenizer,
            self.device
        )

        azimuth_center = -math.pi / 2
        azimuth_range = math.radians(90)
        a = (torch.rand(1, device=self.device) * 2 * azimuth_range) + (azimuth_center - azimuth_range)
        e = torch.rand(1, device=self.device) * (2 * np.deg2rad(15)) - np.deg2rad(15) - math.pi/2 # -pi/2 offset to center the distribution
        self._eye_state = torch.tensor([[a, e]], device=self.device)
        self._smoothed_action = torch.zeros(1, 2, device=self.device) # Reset smoothed action

        # Get initial observation
        multicrop_render = self.video.render_multicrop(self.eye_so3).unsqueeze(0)
        eye_vec = self.eye_so3.as_matrix()[:, :, 2]

        # compute the relevancy over the image.
        equi_img = self.video._frame
        equi_img_batch = equi_img.unsqueeze(0)  # Add batch dim -> (1, C, H, W)
        k = int(equi_img.shape[2]*0.1)
        # Calculate padding needed to ensure corner coverage
        pad_h = k // 2
        pad_w = k // 2

        # Apply padding to the input tensor before unfolding
        equi_img_padded = F.pad(equi_img_batch, (pad_w, pad_w, pad_h, pad_h), mode='constant', value=0)
        # Or use 'reflect' padding to avoid border artifacts:
        # equi_img_padded = F.pad(equi_img_batch, (pad_w, pad_w, pad_h, pad_h), mode='reflect')

        # Then unfold with the padded image
        patches_flat = torch.nn.functional.unfold(equi_img_padded, kernel_size=k, stride=k//2)
        # patches_flat will be (1, C*k*k, L) where L is number of patches

        # Reshape to (1, C, k*k, L)
        patches = patches_flat.view(1, equi_img.shape[0], k*k, -1)

        # Reshape to get the k×k spatial dimensions: (1, C, k, k, L)
        patches = patches.view(1, equi_img.shape[0], k, k, -1)

        # To get batch of patches in format (B, C, k, k) where B = L:
        img_patches = patches.squeeze(0).permute(3, 0, 1, 2)  # (L, C, k, k)

        clip_reward = self.clip_reward.get_relevancy(img_patches, clipped=False)
        if clip_reward.max() < .01:
            print("No clip reward found, defaulting to no target")
            self.target_so3 = None
        else:
            # Calculate number of patches in each dimension
            n_patches_h = ((equi_img.shape[1] + 2*pad_h - k) // (k//2)) + 1
            n_patches_w = ((equi_img.shape[2] + 2*pad_w - k) // (k//2)) + 1

            # Reshape reward to match patch grid dimensions
            reward_img = clip_reward.view(n_patches_h, n_patches_w)

            # Resize to match original image dimensions
            reward_img_full = F.interpolate(
                reward_img.unsqueeze(0).unsqueeze(0),  # Add batch and channel dims
                size=(equi_img.shape[1], equi_img.shape[2]),
                mode='bilinear',
                align_corners=False
            )[0, 0]  # Remove batch and channel dims

            # Plot and save visualization
            # Convert tensors to numpy arrays for plotting
            reward_img_np = reward_img_full.cpu().float().numpy()  # (H, W)
            
            # Find maximum reward coordinates
            max_y, max_x = np.unravel_index(reward_img_np.argmax(), reward_img_np.shape)
            # zero point for eye is -pi/2, -pi/2
            a_target = (-max_x/equi_img.shape[2])* 2 * np.pi + np.pi/2
            e_target = (max_y/equi_img.shape[1]) * - np.pi
            self.target_so3 = SO3.from_z_radians(torch.tensor(a_target, device=self.device)) @ SO3.from_x_radians(torch.tensor(e_target, device=self.device))
        

        # Use cached prompt embeddings for the selected prompt
        obs = {"multicrop": multicrop_render, "eye_direction": eye_vec, "target_clip_vec": self.cached_prompt_embeddings[self.selected_prompt]}

        return obs, {}

    def render(self):
        """
        Render the environment's current state
        Returns:
            tensor: (3, H, W) RGB image
        """
        return self.video.render_image(self.eye_so3)


class VectorizedEyeGym:
    """
    Vectorized version of EyeGym which runs many gyms in parallel using CUDA streams
    for improved GPU utilization
    """

    def __init__(self, gyms: List[EyeGym], device: torch.device):
        self.gyms = gyms
        self.device = device
        
        # Create a CUDA stream for each gym
        if device.type == 'cuda':
            self.streams = [torch.cuda.Stream(device=device) for _ in range(len(gyms))]
        else:
            self.streams = [None] * len(gyms)

    def step(self, actions: torch.Tensor):
        """
        Apply actions to update environment state using CUDA streams for parallelization
        """
        # Run step on each gym and collect results
        all_obs = []
        all_rewards = []
        all_terminated = []
        all_truncated = []
        all_info = []

        # Process each gym with its own stream
        for i, (gym, action) in enumerate(zip(self.gyms, actions)):
            if self.device.type == 'cuda':
                with torch.cuda.stream(self.streams[i]):
                    obs, reward, terminated, truncated, info = gym.step(action)
            else:
                obs, reward, terminated, truncated, info = gym.step(action)
                    
            all_obs.append(obs)
            all_rewards.append(reward)
            all_terminated.append(terminated)
            all_truncated.append(truncated)
            all_info.append(info)

        # Synchronize all streams before combining results
        if self.device.type == 'cuda':
            torch.cuda.synchronize()

        # Combine observations into batched tensor
        batched_obs = {}
        for key in all_obs[0].keys():  # Use first obs dict to get keys
            batched_obs[key] = torch.concatenate([o[key] for o in all_obs])

        # Convert rewards and dones to tensors
        rewards = torch.tensor(all_rewards, device=self.device)
        terminated = torch.tensor(all_terminated, device=self.device)
        truncated = torch.tensor(all_truncated, device=self.device)

        return batched_obs, rewards, terminated, truncated, all_info

    def reset(self):
        """
        Reset environment to initial state using CUDA streams for parallelization
        """
        # Reset each gym and collect results
        all_obs = []
        all_info = []

        # Process each gym with its own stream
        for i, gym in enumerate(self.gyms):
            if self.device.type == 'cuda':
                with torch.cuda.stream(self.streams[i]):
                    obs, info = gym.reset()
            else:
                obs, info = gym.reset()
                    
            all_obs.append(obs)
            all_info.append(info)

        # Synchronize all streams before combining results
        if self.device.type == 'cuda':
            torch.cuda.synchronize()

        # Combine observations into batched tensor
        batched_obs = {}
        for key in all_obs[0].keys():  # Use first obs dict to get keys
            batched_obs[key] = torch.concatenate([o[key] for o in all_obs])

        return batched_obs, all_info

    def render(self):
        """
        Render the environments' current states using CUDA streams for parallelization
        Returns:
            tensor: (B, 3, H, W) RGB images
        """
        renders = [None] * len(self.gyms)
            
        for i, gym in enumerate(self.gyms):
            if self.device.type == 'cuda':
                with torch.cuda.stream(self.streams[i]):
                    renders[i] = gym.render()
            else:
                renders[i] = gym.render()
                
        # Synchronize all streams before stacking results
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
                
        return torch.stack(renders)


class DemoEyeGym:
    def __init__(self, demo_path_list: Dict[str, List[str]], n_envs: int, local_rank: int, episode_length: int,
                 fps_downsamp: int = 1, n_scales: int = 4,
                 ema_smoothing: float = 0.0, time_pause: int = 0, fovea_res: int = 224, window_size: int = 224, 
                 use_se3_reward: bool = True, decoder_device: str = 'cuda',
                 sphere_size: int = 1920,
                 ):
        """
        Args:
            demo_path_list: Dict with keys "filepaths" and "prompts" containing lists of demo paths and corresponding prompts
            n_envs: Number of environments to run in parallel
            local_rank: The local rank for distributed processing
        """
        import jax
        import jax.numpy as jnp
        import jaxlie
        import numpy as onp
        from pyroki import Robot
        import yourdfpy
        self.sphere_size = sphere_size
        self.episode_length = episode_length
        self.fovea_res = fovea_res
        self.local_rank = local_rank
        self.demo_path_list = demo_path_list
        self.filepaths = demo_path_list["filepaths"]
        self.prompts = demo_path_list["prompts"]
        assert len(self.filepaths) == len(self.prompts), "Number of filepaths must match number of prompts"
        self.n_envs = n_envs
        self.device = f"cuda:{self.local_rank}" if self.local_rank != -1 else "cuda"
        
        # Initialize CLIP model temporarily for text embedding
        openclip_model, _, _ = open_clip.create_model_and_transforms(
            "ViT-B-16", pretrained="laion2b_s34b_b88k", device=self.device, precision='fp16'
        )
        openclip_model.eval()
        tokenizer = open_clip.get_tokenizer("ViT-B-16")
        
        # Pre-cache text embeddings for all prompts
        self.cached_prompt_embeddings = {}
        with torch.no_grad():
            for prompt in self.prompts:
                tokens = tokenizer([prompt]).to(self.device)
                text_features = openclip_model.encode_text(tokens)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                self.cached_prompt_embeddings[prompt] = text_features
        
        # Clean up CLIP model to free memory
        del openclip_model, tokenizer
        torch.cuda.empty_cache()
        
        # Track which prompt corresponds to each environment
        self.env_prompt_assignments = [None] * n_envs
        
        self.current_demos = []
        self.prefetch_executor = ThreadPoolExecutor(max_workers=n_envs)
        self.fps_downsamp = fps_downsamp
        self.n_scales = n_scales
        self.ema_smoothing = ema_smoothing

        self.time_pause = time_pause
        self.window_size = window_size
        self.use_se3_reward = use_se3_reward
        self.decoder_device = decoder_device
        urdf = yourdfpy.URDF.load(Path('./urdf/ur5e_with_robotiq_gripper.urdf'), load_meshes=False) 
        self.pk_robot_obj = Robot.from_urdf(urdf)
        # Compute Forward Kinematics
        self.ee_link_index = self.pk_robot_obj.links.names.index("tool0")
        self.prefetch_futures = []
        self.initiate_prefetch_demos()

    @property
    def eye_so3(self):
        return SO3.from_z_radians(self._eye_state[:,0]) @ SO3.from_x_radians(self._eye_state[:,1])

    @torch.inference_mode()
    def step(self, actions: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        "eye": N, 2
        "joint_pos": N, W, 7 (usually W=30)

        Returns:
        """
        import jax.numpy as jnp
        import numpy as onp

        self.elapsed_time += 1
        # Update eye state
        torch.cuda.synchronize()
        self.smoothed_action = self.smoothed_action * self.ema_smoothing + actions['eye'] * (1 - self.ema_smoothing)
        self._eye_state = self._eye_state + self.smoothed_action
        # clip elevation to [-np.pi,0]
        self._eye_state[:,1] = torch.clip(self._eye_state[:,1], -np.pi, 0)
        self._eye_state[:,0] = torch.clip(self._eye_state[:,0], -np.pi/2 - np.deg2rad(130), -np.pi/2 + np.deg2rad(130))


        # Get observation
        eye_so3 = self.eye_so3 # this is (self.n_envs, 3, 3)
        eye_vec = eye_so3.as_matrix()[:, :, 2].bfloat16()
        obs = {}
        obs["multicrop"] = torch.stack([self.current_demos[i].video.render_multicrop(SO3(eye_so3.wxyz[i:i+1])) for i in range(self.n_envs)], dim=0)
        obs["eye_direction"] = eye_vec
        target_embeddings = []
        for i in range(self.n_envs):
            prompt_idx = self.env_prompt_assignments[i]
            prompt = self.prompts[prompt_idx]
            target_embeddings.append(self.cached_prompt_embeddings[prompt])
        obs["target_clip_vec"] = torch.cat(target_embeddings, dim=0)
        obs["proprio"] = torch.cat([self.current_demos[i].robot_data["joint_and_gripper_data"][self.current_frame[i]] for i in range(self.n_envs)], dim=0)

        # reward calculation
        rewards = torch.zeros(self.n_envs, device=self.device, dtype=torch.float32)
        act_size = actions["joint_pos"].shape[1]
        is_se3_actions = actions.get("is_se3", False)
        
        for i in range(self.n_envs):
            # grab the right chunk of the ground truth action
            gt_act_chunk = self.current_demos[i].robot_data["joint_and_gripper_data"][self.current_frame[i]:self.current_frame[i] + act_size]
            # figure out how much data we can actually subtract/compare
            available_size = gt_act_chunk.shape[0]
            compare_size = min(act_size, available_size)

            # Calculate reward based on selected method
            pred_chunk = actions["joint_pos"][i, :compare_size]
            gt_act_chunk = gt_act_chunk[:compare_size]
            
            # Handle SE3 vs joint actions
            if is_se3_actions:
                # pred_chunk is already SE3 poses (10-DOF: 9-DOF SE3 + 1 gripper)
                # Convert ground truth joints to SE3 for comparison
                gt_act_torch = gt_act_chunk.float().cpu().numpy()
                gt_act_jnp = jnp.array(onp.array(gt_act_torch))
                
                # Convert gt joints to SE3 poses
                ee_gt_wxyz_xyz = compute_fk_batch(gt_act_jnp, self.pk_robot_obj, self.ee_link_index)
                ee_gt_torch = torch.from_numpy(onp.array(ee_gt_wxyz_xyz)).to(self.device)
                
                # Extract position from SE3 predictions (first 3 elements)
                pred_positions = pred_chunk[..., :3]  # (A, 3)
                gt_positions = ee_gt_torch[..., 4:]  # Extract xyz from wxyz_xyz format
                
                # Calculate SE3 reward based on position distance
                rewards[i] -= line_distance(gt_positions, pred_positions)
            elif not self.use_se3_reward:
                # Original joint space reward calculation
                rewards[i] -= (pred_chunk - gt_act_chunk).norm(dim=-1).mean()
                # also include the gripper in the reward
                # weight by a factor (se3 is on order of meters)
                # so this corresponds to roughly 20cm of "error"
                rewards[i] -= 0.2*(pred_chunk[..., -1] - gt_act_chunk[..., -1]).abs().mean()
            else:
                # SE3 reward using forward kinematics on joint predictions
                world_preds = pred_chunk
                world_gt = gt_act_chunk
                gt_pred_torch = torch.cat([world_gt, world_preds], dim=0).float().cpu().numpy()
                gt_pred_jnp = jnp.array(onp.array(gt_pred_torch))
                
                ee_gt_pred_wxyz_xyz = compute_fk_batch(gt_pred_jnp, self.pk_robot_obj, self.ee_link_index)
                ee_gt_pred_wxyz_xyz = torch.from_numpy(onp.array(ee_gt_pred_wxyz_xyz)).to(self.device)
                rewards[i] -= line_distance(ee_gt_pred_wxyz_xyz[:ee_gt_pred_wxyz_xyz.shape[0]//2,...,-3:], ee_gt_pred_wxyz_xyz[ee_gt_pred_wxyz_xyz.shape[0]//2:,...,-3:])


        truncated = torch.zeros(self.n_envs, device=self.device)
        terminated = torch.zeros(self.n_envs, device=self.device)
        
        if self.elapsed_time > self.time_pause:
            for i in range(self.n_envs):
                hit_end = self.current_frame[i] >= self.current_demos[i].video.total_frames - 1 - self.fps_downsamp
                if not hit_end:
                    self.current_demos[i].video.advance()
                    self.current_frame[i] += self.fps_downsamp
                terminated[i] = hit_end
        return obs, rewards, terminated, truncated, {'joint_switch': torch.ones((self.n_envs, 1), device=self.device) * (self.elapsed_time > self.time_pause)}
    

    def _prefetch_demo(self):
        # First randomly sample a prompt
        prompt_idx = random.randint(0, len(self.prompts) - 1)
        
        # Then randomly sample a demo filepath for that prompt
        demo_idx = random.randint(0, len(self.filepaths[prompt_idx]) - 1)
        demo_path = self.filepaths[prompt_idx][demo_idx]
        
        video_path = glob.glob(os.path.join(demo_path, "downsampled.mp4"))[0]
        h5_path = glob.glob(os.path.join(demo_path, "*.h5"))[0]
        data = DemonstrationData(video_path, h5_path, local_rank=self.local_rank,
                                                    fps_downsamp=self.fps_downsamp,
                                                    n_scales=self.n_scales, 
                                                    fovea_res=self.fovea_res, window_size=self.window_size,
                                                    decoder_device=self.decoder_device, sphere_size=self.sphere_size)
        safety_margin_size = 1 if self.episode_length is None else (self.episode_length - self.time_pause)*self.fps_downsamp
        start_frame = random.randint(0, max(data.video.total_frames - safety_margin_size - 1,0))
        data.video.reset(random_time = False, frame_idx = start_frame)
        return data, start_frame, prompt_idx, demo_idx

    def initiate_prefetch_demos(self):
        # create futures for the demonstrations, n_envs long
        for i in range(self.n_envs):
            self.prefetch_futures.append(self.prefetch_executor.submit(self._prefetch_demo))

    def reset(self, path_ids: Optional[torch.Tensor] = None, n_envs: Optional[int] = None):
        if n_envs is not None:
            self.n_envs = n_envs
        self.smoothed_action = torch.zeros(self.n_envs, 2, device=self.device)
        azimuth_center = -math.pi / 2
        azimuth_range = math.radians(90)
        a = (torch.rand(self.n_envs, 1, device=self.device) * 2 * azimuth_range) + (azimuth_center - azimuth_range)
        e = torch.rand(self.n_envs, 1, device=self.device) * np.deg2rad(30) - np.deg2rad(15) - math.pi/2 # -pi/2 offset to center the distribution
        self._eye_state = torch.cat([a, e], dim=1)
        self.current_frame = torch.zeros(self.n_envs,1,dtype = torch.int32)
        for demo in self.current_demos:
            demo.video.close()
        self.current_demos.clear()
        for i, demo_fut in enumerate(self.prefetch_futures):
            data, start_frame, prompt_idx, demo_idx = demo_fut.result()
            self.current_demos.append(data)
            self.current_frame[i] = start_frame
            self.env_prompt_assignments[i] = prompt_idx
        self.prefetch_futures.clear()
        self.initiate_prefetch_demos()
        self.start_frames = self.current_frame.clone()
        # Get initial observation
        eye_so3 = self.eye_so3
        eye_vec = eye_so3.as_matrix()[:, :, 2].bfloat16()
        obs = {}
        obs["multicrop"] = torch.stack([self.current_demos[i].video.render_multicrop(SO3(eye_so3.wxyz[i:i+1])) for i in range(self.n_envs)], dim=0)
        obs["eye_direction"] = eye_vec
        target_embeddings = []
        for i in range(self.n_envs):
            prompt_idx = self.env_prompt_assignments[i]
            prompt = self.prompts[prompt_idx]
            target_embeddings.append(self.cached_prompt_embeddings[prompt])
        obs["target_clip_vec"] = torch.cat(target_embeddings, dim=0)
        obs["proprio"] = torch.cat([self.current_demos[i].robot_data["joint_and_gripper_data"][self.current_frame[i]] for i in range(self.n_envs)], dim=0)
        self.elapsed_time = 0
        return obs, {}