# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import os
import random
import time
from dataclasses import dataclass, field
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from eye.sim.spherical_video import SphericalVideo
from eye.sim.gyms import EyeGym, VectorizedEyeGym
from eye.camera import get_default_video_config
import cv2
import glob
from eye.agents import VisualSearchAgent
from eye.agent_configs import VisualSearchAgentConfig, AgentConfigManager
import math
from eye.foveal_encoders import crop_sizes_from_levels
from transformers import SiglipModel
import open_clip
import torchvision
import torch.amp as amp
from typing import List, Tuple
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import sys
from torchvision.transforms import v2

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = False
    """if toggled, `torch.backends.cudnn.deterministic=True` for reproducible results"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = 'TODO'
    """the entity (team) of wandb's project"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""
    num_envs: int = 10
    """the number of parallel game environments"""

    # Algorithm specific arguments
    total_timesteps: int = 2_000_000
    """total timesteps of the experiments"""
    learning_rate: float = 1e-4
    """the learning rate of the optimizer"""
    num_steps: int = 100
    """the number of steps to run in each environment per policy rollout"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 2
    """the number of mini-batches"""
    grad_accumulation_steps: int = 1
    """the number of gradient accumulation steps"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = False
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.02
    """coefficient of the entropy"""
    ent_coef_end: float = 0.02
    """end coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 1.0
    """the maximum norm for the gradient clipping"""
    target_kl: float = .02
    """the target KL divergence threshold"""

    # EyeGym specific arguments
    constancy_alpha: float = 0.00
    """coefficient of the constancy reward"""
    clip_alpha: float = 10.0
    """coefficient of the clip reward"""

    # Agent configuration arguments
    agent_args: VisualSearchAgentConfig = VisualSearchAgentConfig(
        action_type="discrete",
        freeze_encoder=True,
        learnable_registers=0,
        n_levels=3,
        window_length=10,
        fovea_size=320,
        window_size=256,
        magnitudes=(10.0, 3.0, 1.0),
        mask_past_img_attn=True,
        n_blocks=3,
        pool_size=1,
        sphere_size=1920,
        decoder_only=True
    )
    """Agent configuration parameters"""
    
    # Other training specific arguments
    vit_type: str = "dino" 
    ema_alpha: float = 0.0
    window_lengths: List[int] = field(default_factory=lambda:[10])
    window_transitions: List[int] = field(default_factory=lambda: [0])
    
    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""
    warmup_global_steps: int = 200_000
    """number of global steps for learning rate warmup"""
    decay_global_steps: int = 500_000
    """number of global steps for learning rate decay at the end"""

    # Add DDP-specific arguments
    local_rank: int = field(default_factory=lambda: int(os.environ.get("LOCAL_RANK", -1)))
    """local rank for distributed training"""

    gpu_cache: bool = True
    """if toggled, environment data will be cached on GPU instead of CPU"""
    
    # Simple augmentation parameters
    use_augmentations: bool = True
    """if toggled, apply color jitter and blur augmentations during training"""


def make_env(args, model, preprocess, tokenizer):
    def thunk():
        w = 1920
        h = 1200
        K, dist_coeffs, is_fisheye = get_default_video_config(w, h)

        # Create demo path dict using same structure as ppo_robot
        demo_path_dict = {
            "filepaths": [
                glob.glob('data/demos/towel/*/processed/*'),
                glob.glob('data/demos/eraser/*/processed/*'),
                glob.glob('data/demos/estop/*/processed/*'),
                glob.glob('data/demos/brush/*/processed/*'),
                glob.glob('data/demos/wood_x/*/processed/*')
            ],
            "positive_crops": [
                [glob.glob("data/crops/towel/*"), glob.glob("data/crops/bucket/*")],
                [glob.glob("data/crops/eraser/*"), glob.glob("data/crops/expo_stand/*")],
                [glob.glob("data/crops/estop/*")],
                [glob.glob("data/crops/brush/*")] ,
                [glob.glob("data/crops/wood_x/*")] 
            ],
            "prompts": [
                ["a towel", "a white tray"], 
                ["an eraser", "an expo stand"],
                ["an emergency stop button"],
                ["a scrub brush"],
                ["a wooden square"]
            ]
        }

        return EyeGym(
            demo_path_dict,
            500,
            negative_prompts=glob.glob("data/crops/negatives/*"),
            device=f"cuda:{args.local_rank}" if args.local_rank != -1 else "cuda",
            openclip_model=model,
            preprocess=preprocess,
            tokenizer=tokenizer,
            constancy_alpha=args.constancy_alpha,
            clip_alpha=args.clip_alpha,
            ema_alpha=args.ema_alpha,
            # Video creation parameters
            video_width=w,
            video_height=h,
            K=K,
            dist_coeffs=dist_coeffs,
            is_fisheye=is_fisheye,
            crop_sizes=crop_sizes_from_levels(args.agent_args.n_levels, args.agent_args.fovea_size, args.agent_args.sphere_size),
            window_size=args.agent_args.window_size,
            decoder_device='cpu',
        )

    return thunk


def get_current_window_length(global_step, window_lengths, window_transitions):
    """
    Determine the current window length based on training progress.
    
    Args:
        global_step: Current training step
        window_lengths: List of window lengths to use at different stages
        window_transitions: List of step counts at which to transition to new window lengths
    
    Returns:
        The appropriate window length for the current training step
    """
    for i in range(len(window_transitions)):
        if global_step < window_transitions[i]:
            return window_lengths[i]
    
    # If we've passed all transitions, use the final window length
    return window_lengths[-1]


def main():
    torch._dynamo.config.cache_size_limit = 128
    
    # Read LOCAL_RANK before parsing args
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        print(f"Found LOCAL_RANK environment variable: {local_rank}")
        # Add the local_rank to sys.argv so tyro can parse it
        sys.argv.extend(["--local_rank", str(local_rank)])
    
    args = tyro.cli(Args)
    
    # Print the local_rank for debugging
    print(f"Using local_rank: {args.local_rank}")
    
    if args.local_rank == -1:
        # Not using distributed mode
        args.local_rank = 0
        rank = 0
        world_size = 1
        is_distributed = False
    else:
        # Initialize distributed mode
        is_distributed = True
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend='nccl')
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        print(f"Initialized process group: rank {rank} of {world_size}")
        
    # Only track and log with rank 0 process
    args.track = args.track and rank == 0
    this_rank_total_timesteps = int(args.total_timesteps / world_size)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = this_rank_total_timesteps  // args.batch_size
    run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"
    
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            save_code=True,
        )
    
    writer = None
    if rank == 0:
        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s"
            % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )

    # TRY NOT TO MODIFY: seeding
    # Make sure each process has a different seed
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup - simplified since we're not using vectorized envs
    model, _, _ = open_clip.create_model_and_transforms(
            "ViT-B-16", pretrained="laion2b_s34b_b88k", device=device, precision='fp16'
        )
    model.eval()
    preprocess = torchvision.transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711],
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-16")
    env = VectorizedEyeGym(
        [make_env(args, model, preprocess, tokenizer)() for _ in range(args.num_envs)], device
    )

    # Initialize agent with observation/action spaces
    # Initialize agent with config
    agent = VisualSearchAgent(config=args.agent_args).to(device)
        
    
    # Wrap agent in DDP if using distributed training
    if is_distributed:
        agent = DDP(agent, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
        agent_module = agent.module
    else:
        agent_module = agent
    
    # Optimizer and Scheduler Setup
    agent_params_to_optimize = [p for p in agent.parameters() if p.requires_grad]

    if rank == 0: print("Using AdamW optimizer setup.")
    optimizer = None
    if agent_params_to_optimize: 
        optimizer = optim.AdamW(
            agent_params_to_optimize,
            lr=args.learning_rate,
            eps=1e-7,
            weight_decay=0.01,
            betas=(0.9, 0.95) 
        )
    else:
        raise ValueError("No parameters to optimize")

    # --- Scheduler Setup ---
    # Calculate phases: warmup -> stable -> decay
    total_global_steps = args.total_timesteps
    stable_steps = total_global_steps - args.warmup_global_steps - args.decay_global_steps
    scheduler = None
    if optimizer:
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                # Warmup phase
                optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=0.1, end_factor=1.0, total_iters=args.warmup_global_steps
                ),
                # Stable phase
                optim.lr_scheduler.ConstantLR(
                    optimizer, factor=1.0, total_iters=stable_steps
                ),
                # Decay phase
                optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=args.decay_global_steps,
                    eta_min=args.learning_rate * 0.01,
                ),
            ],
            milestones=[args.warmup_global_steps, args.warmup_global_steps + stable_steps],
        )

    # Storage setup - modified to include foveal_tokens
    cache_device = device if args.gpu_cache else torch.device("cpu")
    
    obs = {
        "multicrop": torch.zeros(
            (args.num_steps, args.num_envs, args.agent_args.n_levels, 3, args.agent_args.window_size, args.agent_args.window_size), dtype=torch.bfloat16, device=cache_device
        ),
        "eye_direction": torch.zeros((args.num_steps, args.num_envs, 3), dtype=torch.bfloat16, device=cache_device),
        "target_clip_vec": torch.zeros((args.num_steps, args.num_envs, 512), dtype=torch.bfloat16, device=cache_device),
        # Add space for foveal tokens - exact size depends on your model
        "foveal_tokens": torch.zeros((args.num_steps, args.num_envs, args.agent_args.n_levels * (args.agent_args.window_size//(16*args.agent_args.pool_size))**2, 384), dtype=torch.bfloat16, device=cache_device),
    }
    actions = torch.zeros((args.num_steps, args.num_envs, 2), dtype=torch.bfloat16, device=cache_device)
    logprobs = torch.zeros((args.num_steps, args.num_envs), device=cache_device)
    rewards = torch.zeros((args.num_steps, args.num_envs), device=cache_device)
    dones = torch.zeros((args.num_steps, args.num_envs), device=cache_device)
    values = torch.zeros((args.num_steps, args.num_envs), device=cache_device)
    # Start the game
    save_model_freq = 100  # Save model every 100 iterations
    global_step = 0
    num_update_epochs = 0
    start_time = time.time()
    for iteration in range(1, args.num_iterations + 1):
        next_obs, _ = env.reset()
        next_done = torch.zeros(args.num_envs).to(device)

        # --- Scheduler Step ---
        if scheduler is not None:
            for _ in range(args.num_envs * args.num_steps * (dist.get_world_size() if is_distributed else 1)):
                scheduler.step()

        for step in range(0, args.num_steps):
            if is_distributed:
                global_step += args.num_envs * dist.get_world_size()
            else:
                global_step += args.num_envs

            # Store observations (modified to handle device placement)
            # Store directly on GPU
            obs["multicrop"][step] = next_obs["multicrop"].to(cache_device)
            obs["eye_direction"][step] = next_obs["eye_direction"].to(cache_device)
            obs["target_clip_vec"][step] = next_obs["target_clip_vec"].to(cache_device)
            dones[step] = next_done.to(cache_device)

            # ALGO LOGIC: action logic
            # Get current window length based on training progress
            current_window_length = get_current_window_length(
                global_step, args.window_lengths, args.window_transitions
            )
            
            # Create a context window using only available observations
            window_obs = {}
            for k in next_obs:
                # Collect available history (up to current_window_length)
                past_obs = []
                # Get previous observations from buffer (as many as available up to current_window_length-1)
                for s in range(max(0, step - (current_window_length - 1)), step):
                    # If data is already on GPU, no need to transfer
                    if args.gpu_cache:
                        past_obs.append(obs[k][s])
                    else:
                        past_obs.append(obs[k][s].to(device))
                # Add current observation
                past_obs.append(next_obs[k].to(device))
                # Stack along time dimension (T, B, ...)
                window_obs[k] = torch.stack(past_obs)
            # populate the foveal_tokens for window_len-1 frames
            if step > 0:
                window_obs['foveal_tokens'] = obs['foveal_tokens'][max(0, step - (current_window_length - 1)):step].to(device)
            agent_module.eval()
            torch.compiler.cudagraph_mark_step_begin()
            with torch.no_grad(), amp.autocast('cuda', dtype=torch.bfloat16):
                action, logprob, _, value = agent_module.get_action_and_value(window_obs, window_length=current_window_length)
            # Store foveal_tokens if they were computed
            if 'foveal_tokens' in window_obs:
                obs["foveal_tokens"][step] = window_obs['foveal_tokens'][-1].to(cache_device)
            action = action[-1]
            value = value[-1]
            logprob = logprob[-1]
            # For context window, we take the last value and action (intermediates are ignored during rollout)
            values[step] = value.flatten().to(cache_device)
            actions[step] = action.to(cache_device)
            logprobs[step] = logprob.to(cache_device)

            # Execute the game and log data
            next_obs, reward, terminated, truncated, info = env.step(action.squeeze(0))
            next_done = torch.logical_or(terminated, truncated).to(
                device, torch.float32
            )
            rewards[step] = reward.to(cache_device).view(-1)

            # Store frames for video logging (new code)
            if step == 0:
                # Initialize frame storage at start of rollout
                train_frames = []

            # Save frames for first 4 environments
            if args.track and num_update_epochs % 20 == 0:
                num_envs_to_log = min(4, args.num_envs)  # Log up to 4 environments
                rewards_this_step = rewards[step,:num_envs_to_log]
                multicrop = next_obs["multicrop"][
                    :num_envs_to_log
                ]  # Shape: (N,4,3,224,224)
                # Convert multicrop to a single frame dynamically based on n_scales
                frames = multicrop  # Shape: (N, n_scales, 3, 224, 224)
                
                # Apply same augmentations to visualization frames if enabled
                if args.use_augmentations:
                    N, L, C, H, W = frames.shape
                    frames_flat = frames.float().view(-1, C, H, W)
                    vis_augment_transform = v2.Compose([
                        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
                        v2.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
                    ])
                    frames_flat = vis_augment_transform(frames_flat)
                    frames = frames_flat.view(N, L, C, H, W)
                
                n_scales = frames.shape[1]
                if n_scales == 1:
                    frames = frames[:, 0]
                elif n_scales == 2:
                    frames = torch.cat([frames[:, 1], frames[:, 0]], dim=3)
                elif n_scales == 3:
                    top_row = torch.cat(
                        [torch.zeros_like(frames[:, 2]), frames[:, 2]], dim=3
                    )
                    bottom_row = torch.cat([frames[:, 1], frames[:, 0]], dim=3)
                    frames = torch.cat([top_row, bottom_row], dim=2)
                elif n_scales == 4:
                    top_row = torch.cat([frames[:, 3], frames[:, 2]], dim=3)
                    bottom_row = torch.cat([frames[:, 1], frames[:, 0]], dim=3)
                    frames = torch.cat([top_row, bottom_row], dim=2)
                else:
                    raise ValueError(
                        f"Unsupported number of scales: {n_scales}. Must be between 2 and 4."
                    )
                frames = (frames * 255).byte().cpu().numpy()
                # Write reward to frames
                for i in range(frames.shape[0]):
                    # Convert to format expected by OpenCV (H,W,C) from (C,H,W)
                    frame = frames[i].transpose(1, 2, 0).copy()
                    reward_text = f"Reward: {rewards_this_step[i].item():.3f}"
                    cv2.putText(
                        frame, 
                        reward_text, 
                        (10, 30),  # Position (x, y) from top-left
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7,  # Font scale
                        (0, 255, 0),  # Green color (BGR)
                        2  # Line thickness
                    )
                    # Convert back to (C,H,W) format
                    frames[i] = frame.transpose(2, 0, 1)
                train_frames.append(frames)

        # Log training video to wandb
        if args.track and rank == 0 and len(train_frames) > 0:
            train_frames_np = np.stack(train_frames, axis=1)  # Shape: (B,T,3,H,W)
            for i in range(train_frames_np.shape[0]):
                gym = env.gyms[i]
                wandb.log(
                    {
                        "train/video": wandb.Video(
                            train_frames_np[i], 
                            fps=30, 
                            format="mp4",
                            caption=gym.selected_prompt
                        ),
                        "global_step": global_step,
                    }
                )

        # Get current window length based on training progress
        current_window_length = get_current_window_length(
            global_step, args.window_lengths, args.window_transitions
        )
        
        # Use available observations from buffer plus the final observation
        window_obs = {}
        for k in next_obs:
            history = []
            # Get most recent observations from buffer (up to current_window_length-1)
            start_idx = max(0, args.num_steps - (current_window_length - 1))
            for s in range(start_idx, args.num_steps):
                history.append(obs[k][s].to(device))
            # Add final observation
            history.append(next_obs[k].to(device))
            window_obs[k] = torch.stack(history)
            
        torch.compiler.cudagraph_mark_step_begin()
        with torch.no_grad(), amp.autocast('cuda', dtype=torch.bfloat16):
            next_value = agent_module.get_value(window_obs, window_length=current_window_length)[-1].reshape(1, -1).to(cache_device)
        advantages = torch.zeros_like(rewards)  # Already on the right device
        lastgaelam = 0
        for t in reversed(range(args.num_steps)):
            if t == args.num_steps - 1:
                nextnonterminal = 1.0 - next_done.to(cache_device)
                nextvalues = next_value.to(cache_device)
            else:
                nextnonterminal = 1.0 - dones[t + 1]
                nextvalues = values[t + 1]
            delta = (
                rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
            )
            advantages[t] = lastgaelam = (
                delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            )
        returns = advantages + values

        # flatten the batch (keep on CPU until minibatch sampling)
        b_logprobs = logprobs.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Normalize advantages over the entire batch before starting epochs
        if args.norm_adv:
            if is_distributed:
                # Ensure b_advantages is on the GPU for all_gather
                advantages_on_device = b_advantages.to(device)
                
                world_size = dist.get_world_size()
                
                # Create a list to store all advantages tensors from all processes
                all_advantages_list = [torch.empty_like(advantages_on_device) for _ in range(world_size)]
                
                # Gather all b_advantages tensors
                dist.all_gather(all_advantages_list, advantages_on_device)
                
                # Concatenate all gathered tensors to form a single tensor
                # This tensor contains all advantages from all processes
                global_advantages = torch.cat(all_advantages_list, dim=0)
                
                # Calculate mean and std dev from the global advantages
                global_mean = global_advantages.mean()
                global_std = global_advantages.std()
                
                # Normalize the local b_advantages using the global mean and std
                # No need to use advantages_on_device here, normalize the original b_advantages
                # which might be on cache_device or device. The result should be on cache_device.
                b_advantages = (b_advantages.to(device) - global_mean) / (global_std + 1e-5)
                b_advantages = b_advantages.to(cache_device)

            else:
                # Original normalization for non-distributed case
                b_advantages = (b_advantages - b_advantages.mean()) / (
                    b_advantages.std() + 1e-6
                )


        # Optimizing the policy and value network
        envsperbatch = args.num_envs // args.num_minibatches
        envinds = np.arange(args.num_envs)
        flatinds = np.arange(args.batch_size).reshape(args.num_steps, args.num_envs)
        clipfracs = []
        num_update_epochs += 1
        agent_module.train()
        for epoch in range(args.update_epochs):
            np.random.shuffle(envinds)
            for start in range(0, args.num_envs, envsperbatch):
                end = start + envsperbatch
                mbenvinds = envinds[start:end]
                mb_inds = flatinds[:, mbenvinds].ravel()  # Gets indices for all timesteps for selected envs
                # Transfer minibatch data to GPU
                b_obs_mb = {k: v[:,mbenvinds].to(device) for k, v in obs.items()}
                
                # Apply simple augmentations to multicrop images during training
                if args.use_augmentations and 'multicrop' in b_obs_mb:
                    if torch.rand(1).item() < 0.5:  # 50% chance of augmentation
                        T, B, L, C, H, W = b_obs_mb['multicrop'].shape
                        images = b_obs_mb['multicrop'].float().view(-1, C, H, W)  # Flatten to (T*B*L, C, H, W)
                        
                        # Simple torchvision transforms applied to entire batch
                        augment_transform = v2.Compose([
                            v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
                            v2.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
                        ])
                        
                        # Apply transforms to the entire batch
                        images = augment_transform(images)
                        b_obs_mb['multicrop'] = images.view(T, B, L, C, H, W).to(b_obs_mb['multicrop'].dtype)
                
                b_actions_mb = actions[:, mbenvinds].to(device)
                torch.compiler.cudagraph_mark_step_begin()
                with amp.autocast('cuda', dtype=torch.bfloat16):
                    # Note: The agent should handle the window length internally during training
                    # since we're passing the full sequence of observations
                    _, newlogprob, entropy, newvalue = agent(
                        b_obs_mb, action=b_actions_mb, window_length=get_current_window_length(global_step, args.window_lengths, args.window_transitions)
                    )
                    newlogprob = newlogprob.flatten()
                    newvalue = newvalue.flatten()
                    entropy = entropy.flatten()

                    logratio = newlogprob - b_logprobs[mb_inds].to(device)
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [
                            ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                        ]

                    mb_advantages = b_advantages[mb_inds].to(device)

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(
                        ratio, 1 - args.clip_coef, 1 + args.clip_coef
                    )
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    mb_returns = b_returns[mb_inds].to(device)
                    mb_values = b_values[mb_inds].to(device)
                    if args.clip_vloss:
                        v_loss_unclipped = (newvalue - mb_returns) ** 2
                        v_clipped = mb_values + torch.clamp(
                            newvalue - mb_values,
                            -args.clip_coef,
                            args.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - mb_returns) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()

                    # Add this function somewhere appropriate in your script
                    def schedule_entropy_coefficient(start_value, end_value, current_step, total_steps):
                        """
                        Linearly schedule entropy coefficient from start_value to end_value based on progress.
                        """
                        progress_fraction = min(current_step / total_steps, 1.0)
                        return start_value + progress_fraction * (end_value - start_value)
                    
                    # Then use current_ent_coef instead of ent_coef in your loss calculation
                    current_ent_coef = schedule_entropy_coefficient(
                        args.ent_coef, 
                        args.ent_coef_end, 
                        global_step,  # Assuming you have a global_step variable that tracks overall training progress
                        args.total_timesteps  # This would be the total expected steps for the entire training
                    )
                    
                    entropy_loss = entropy.mean()
                    loss = pg_loss - current_ent_coef * entropy_loss + v_loss * args.vf_coef

                loss = loss/args.grad_accumulation_steps
                loss.backward()

                # Only perform optimization step after accumulating gradients
                if (start + envsperbatch) % (envsperbatch * args.grad_accumulation_steps) == 0:
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    if optimizer:
                        optimizer.step()
                        optimizer.zero_grad()

            # Add synchronization for KL divergence across all processes
            if args.target_kl is not None:
                # First compute local approx_kl
                local_approx_kl = approx_kl.clone()
                
                if is_distributed:
                    # Create a tensor to hold the gathered values
                    gathered_kl = torch.zeros(dist.get_world_size(), device=device)
                    
                    # All-gather the approx_kl values from all processes
                    dist.all_gather_into_tensor(gathered_kl, local_approx_kl.view(1))
                    # Compute the average KL across all processes
                    global_approx_kl = gathered_kl.mean().item()
                    
                    # Make sure all processes have the same decision
                    should_stop = global_approx_kl > args.target_kl
                    
                    # Broadcast the decision from rank 0 to ensure consistency
                    should_stop_tensor = torch.tensor([1 if should_stop else 0], device=device)
                    dist.broadcast(should_stop_tensor, src=0)
                    should_stop = bool(should_stop_tensor.item())
                else:
                    # Non-distributed case - just use the local value
                    global_approx_kl = local_approx_kl.item()
                    should_stop = global_approx_kl > args.target_kl
                
                if should_stop:
                    if rank == 0:
                        print(f"Early stopping at step {epoch} due to reaching target KL: {global_approx_kl:.4f} > {args.target_kl:.4f}")
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if rank == 0:
            # Log learning rate
            if optimizer: 
                writer.add_scalar(
                    "charts/learning_rate", optimizer.param_groups[0]["lr"], global_step
                )
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar(
                "charts/episodic_return", b_returns.mean().item(), global_step
            )
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            writer.add_scalar("losses/explained_variance", explained_var, global_step)

            print("SPS:", int(global_step / (time.time() - start_time)))
            writer.add_scalar(
                "charts/SPS", int(global_step / (time.time() - start_time)), global_step
            )

        # Only evaluate every eval_freq iterations or on the final iteration
        if args.save_model and rank == 0 and iteration % save_model_freq == 0:
            checkpoint_dir = f"runs/{run_name}"
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{iteration}.pt")
            
            # Save the model state dict - if using DDP, need to access .module
            model_state_dict = agent_module.state_dict() if is_distributed else agent.state_dict()
            
            optimizer_state_dict = optimizer.state_dict() if optimizer else None
            scheduler_state_dict = scheduler.state_dict() if scheduler is not None else None
            
            save_dict = {
                "iteration": iteration,
                "model_state_dict": model_state_dict,
            }
            if optimizer_state_dict:
                save_dict["optimizer_state_dict"] = optimizer_state_dict
            if scheduler_state_dict:
                save_dict["scheduler_state_dict"] = scheduler_state_dict
                
            torch.save(save_dict, checkpoint_path)
            
            # Use AgentConfigManager to save config with checkpoint
            AgentConfigManager.save_config_with_checkpoint(args.agent_args, checkpoint_path)
            print(f"Saved agent config with AgentConfigManager")
                
            print(f"Saved model checkpoint to {checkpoint_path}")

    # Final checkpoint save only on rank 0
    if rank == 0:
        checkpoint_dir = f"runs/{run_name}"
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_final.pt")
        
        # Save the model state dict - if using DDP, need to access .module
        model_state_dict = agent_module.state_dict() if is_distributed else agent.state_dict()
        
        optimizer_state_dict = optimizer.state_dict() if optimizer else None
        scheduler_state_dict = scheduler.state_dict() if scheduler is not None else None
        
        save_dict = {
            "iteration": iteration, # Current iteration value at the end of training
            "model_state_dict": model_state_dict,
        }
        if optimizer_state_dict:
            save_dict["optimizer_state_dict"] = optimizer_state_dict
        if scheduler_state_dict:
            save_dict["scheduler_state_dict"] = scheduler_state_dict

        torch.save(save_dict, checkpoint_path)
        
        # Use AgentConfigManager to save config with checkpoint
        AgentConfigManager.save_config_with_checkpoint(args.agent_args, checkpoint_path)
        print(f"Saved agent config with AgentConfigManager")
            
        print(f"Saved model checkpoint to {checkpoint_path}")
        
        if writer:
            writer.close()

    # Clean up distributed process group
    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
