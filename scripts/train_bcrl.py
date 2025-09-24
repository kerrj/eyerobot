import os
os.environ['MUJOCO_GL'] = 'egl'
os.environ['JAX_PLATFORMS'] = 'cpu'
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
import glob
from eye.sim.gyms import DemoEyeGym
import jaxlie
from eye.sim.demo_data import DemonstrationData, compute_fk_batch
import jax.numpy as jnp
import numpy as onp
from eye.camera import get_default_video_config
import cv2
from eye.agents import EyeRobotAgent
from eye.agent_configs import EyeRobotAgentConfig, AgentConfigManager
import math
from eye.foveal_encoders import crop_sizes_from_levels
from transformers import SiglipModel
import open_clip
import torchvision
import torch.amp as amp
from typing import List, Tuple, Optional
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
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = 'TODO'
    """the entity (team) of wandb's project"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""
    num_envs: int = 6
    """the number of parallel game environments"""
    decoder_device: str = 'cpu'
    """the device to use for the decoder"""

    # Algorithm specific arguments
    total_timesteps: int = 14_000_000
    """total timesteps of the experiments"""
    learning_rate: float = 1e-4
    """the learning rate of the optimizer"""
    eye_lr_scale: float = 0.5
    """the learning rate scale for the eye optimizer as a proportion of the overall learning rate"""
    num_steps: int = 130
    """the number of steps to run in each environment per policy rollout"""
    time_pause: int = 30
    """the number of steps where time is paused per policy rollout"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 2
    """the number of mini-batches"""
    grad_accumulation_steps: int = 1
    """the number of gradient accumulation steps"""
    rl_update_epochs: int = 10
    """the K epochs to update the eye policy"""
    bc_update_epochs: int = 10
    """the K epochs to update the bc policy"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = False
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    ent_coef_end: float = 0.003
    """end coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 1.0
    """the maximum norm for the gradient clipping"""
    target_kl: float = .02
    """the target KL divergence threshold"""

    bc_coef: float = 1.0
    """coefficient of the bc loss"""
    # Agent configuration arguments
    agent_args: EyeRobotAgentConfig = EyeRobotAgentConfig(
        freeze_encoder=True,
        eye_action_type="discrete", 
        eye_magnitudes=(10.0, 3.0, 1.0),
        n_levels=3,
        n_hand_levels=3,
        fovea_size=320,
        window_size=256,
        window_length=10,
        action_chunk_size=30,
        proprio_dropout=0.0,# full token dropout
        proprio_hidden_dropout=0.1,# hidden dimension of projection dropout
        num_blocks=3,
        relative_act=True,
        pool_size=1,
        sphere_size=1920,
        use_se3_actions=False,
        decoder_only=True
    )
    rl_dropout: float = False
    """use dropout during RL rollout"""
    
    """Agent configuration parameters"""
    window_lengths: List[int] = field(default_factory=lambda:[10])
    window_transitions: List[int] = field(default_factory=lambda: [0])
    load_eye_weights_from: Optional[str] = None
    use_se3_reward: bool = True
    fps_downsamp: int = 2
    """downsample factor for the environment, video+actions. action chunk predictions are NOT downsampled, they are kept at original FPS"""
    eye_ema_smoothing: float = 0.0

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""
    warmup_global_steps: int = 300_000
    """number of global steps for learning rate warmup"""
    decay_global_steps: int = 2_000_000
    """number of global steps for learning rate decay at the end"""

    # Add DDP-specific arguments
    local_rank: int = field(default_factory=lambda: int(os.environ.get("LOCAL_RANK", -1)))
    """local rank for distributed training"""

    gpu_cache: bool = True
    """if toggled, environment data will be cached on GPU instead of CPU"""
    
    # Simple augmentation parameters
    use_augmentations: bool = True
    """if toggled, apply color jitter and blur augmentations during training"""


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
    # Set MuJoCo GL backend to EGL
    os.environ['MUJOCO_GL'] = 'egl'
    
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
    
    # Initialize distributed environment
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
    args.num_iterations = this_rank_total_timesteps // args.batch_size
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

    # Make sure each process has a different seed
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)

    device = 'cuda'

    demo_path_list = {
        "filepaths": [
            glob.glob('data/demos/towel/*/processed/*'),
            glob.glob('data/demos/eraser/*/processed/*'),
            glob.glob('data/demos/estop/*/processed/*'),
            glob.glob('data/demos/brush/*/processed/*'),
            glob.glob('data/demos/wood_x/*/processed/*')
        ],
        "prompts": [
            "a towel",
            "an eraser",
            "an emergency stop button",
            "a scrub brush",
            "a wooden square"
        ]
    }
    
    env = DemoEyeGym(demo_path_list=demo_path_list, 
                     n_envs=args.num_envs, local_rank=args.local_rank, 
                     fps_downsamp=args.fps_downsamp, n_scales = args.agent_args.n_levels,
                     ema_smoothing=args.eye_ema_smoothing, time_pause=args.time_pause, 
                     fovea_res=args.agent_args.fovea_size, window_size=args.agent_args.window_size, episode_length = args.num_steps, 
                     use_se3_reward=args.use_se3_reward, decoder_device=f'cuda:{rank}' if args.decoder_device == 'cuda' else 'cpu',
    sphere_size=args.agent_args.sphere_size)
    # turn off data normalization for relative actions, this is handled by the agent
    # Initialize agent with config
    agent = EyeRobotAgent(config=args.agent_args).to(device)

    eye_params = [p for p in agent.get_eye_params() if p.requires_grad]
    hand_params = [p for p in agent.get_hand_params() if p.requires_grad]

    print("Using AdamW optimizer setup.")
    eye_optimizer = None
    if eye_params:
        eye_optimizer = optim.AdamW(
            eye_params,
            lr=args.learning_rate*args.eye_lr_scale,
            eps=1e-7,
            weight_decay=0.01,
            betas=(0.9, 0.95),
        )
    
    hand_optimizer = None
    if hand_params:
        hand_optimizer = optim.AdamW(
            hand_params,
            lr=args.learning_rate,
            eps=1e-7,
            weight_decay=0.01,
            betas=(0.9, 0.95),
        )
            
    # --- Scheduler Setup ---
    # Calculate phases: warmup -> stable -> decay
    total_global_steps = args.total_timesteps
    stable_steps = total_global_steps - args.warmup_global_steps - args.decay_global_steps
    eye_scheduler = None
    if eye_optimizer:
        eye_scheduler = optim.lr_scheduler.SequentialLR(
            eye_optimizer,
            schedulers=[
                # Warmup phase
                optim.lr_scheduler.LinearLR(
                    eye_optimizer, start_factor=0.1, end_factor=1.0, total_iters=args.warmup_global_steps
                ),
                # Stable phase
                optim.lr_scheduler.ConstantLR(
                    eye_optimizer, factor=1.0, total_iters=stable_steps
                ),
                # Decay phase
                optim.lr_scheduler.CosineAnnealingLR(
                    eye_optimizer,
                    T_max=args.decay_global_steps,
                    eta_min=args.learning_rate * args.eye_lr_scale * 0.01,
                ),
            ],
            milestones=[args.warmup_global_steps, args.warmup_global_steps + stable_steps],
        )

    hand_scheduler = None
    if hand_optimizer:
        hand_scheduler = optim.lr_scheduler.SequentialLR(
            hand_optimizer,
            schedulers=[
                # Warmup phase
                optim.lr_scheduler.LinearLR(
                    hand_optimizer, start_factor=0.1, end_factor=1.0, total_iters=args.warmup_global_steps
                ),
                # Stable phase
                optim.lr_scheduler.ConstantLR(
                    hand_optimizer, factor=1.0, total_iters=stable_steps
                ),
                # Decay phase
                optim.lr_scheduler.CosineAnnealingLR(
                    hand_optimizer,
                    T_max=args.decay_global_steps,
                    eta_min=args.learning_rate * 0.01,
                ),
            ],
            milestones=[args.warmup_global_steps, args.warmup_global_steps + stable_steps],
        )
    
    
    # Load weights from checkpoint if provided
    if args.load_eye_weights_from is not None:
        print(f"Loading weights from checkpoint: {args.load_eye_weights_from}")
        checkpoint = torch.load(args.load_eye_weights_from, map_location=device)
        if "model_state_dict" in checkpoint:
            # Get current state dict
            current_state_dict = agent.state_dict()
            # Get checkpoint state dict
            checkpoint_state_dict = checkpoint["model_state_dict"]
            
            # Create a new state dict with only matching keys
            new_state_dict = {}
            for key in checkpoint_state_dict:
                if key in current_state_dict and checkpoint_state_dict[key].shape == current_state_dict[key].shape:
                    new_state_dict[key] = checkpoint_state_dict[key]
                else:
                    print(f"Skipping key {key} - shape mismatch or key not found in current model")
            #duplicate keys beginning in "transformer" to be also included replaced with "hand_transformer"
            for key in current_state_dict:
                if key.startswith("transformer"):
                    new_state_dict[f"hand_transformer{key[len('transformer'):]}"] = current_state_dict[key].clone()
            # remove all keys which begin with "critic"
            new_state_dict = {k: v for k, v in new_state_dict.items() if not k.startswith("critic")}
            # Load the filtered state dict
            agent.load_state_dict(new_state_dict, strict=False)
            print("Successfully loaded weights from checkpoint")
        else:
            print("Warning: No model_state_dict found in checkpoint")
        # Load optimizer state
        # if "eye_optimizer_state_dict" in checkpoint:
        #     eye_optimizer.load_state_dict(checkpoint["eye_optimizer_state_dict"])
        # if "hand_optimizer_state_dict" in checkpoint:
        #     hand_optimizer.load_state_dict(checkpoint["hand_optimizer_state_dict"])
        
        # # Load scheduler state
        # if "eye_scheduler_state_dict" in checkpoint:
        #     eye_scheduler.load_state_dict(checkpoint["eye_scheduler_state_dict"])
        # if "hand_scheduler_state_dict" in checkpoint:
        #     hand_scheduler.load_state_dict(checkpoint["hand_scheduler_state_dict"])
        
        # # Resume iteration count if available
        # if "iteration" in checkpoint:
        #     start_iteration = checkpoint["iteration"] + 1
        #     # Update global step
        #     global_step = start_iteration * args.batch_size
        #     print(f"Resuming from iteration {start_iteration}")
        
        start_iteration = 1
    else:
        start_iteration = 1
    
    # Wrap agent in DDP if using distributed training
    if is_distributed:
        agent = DDP(agent, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
        agent_module = agent.module
    else:
        agent_module = agent
    agent_module.feature_extractor.compile(dynamic=True)
    
    # Replace the optimizer and add scheduler setup
    global_step = 0
    # After creating the optimizer and scheduler, add this code to load from checkpoint:
    # Storage setup - modified to include foveal_tokens
    cache_device = device if args.gpu_cache else torch.device("cpu")
    n_env_cache = args.num_envs
    obs = {
        "multicrop": torch.zeros(
            (args.num_steps, n_env_cache, args.agent_args.n_levels, 3, args.agent_args.window_size, args.agent_args.window_size), dtype=torch.bfloat16, device=cache_device
        ),
        "eye_direction": torch.zeros((args.num_steps, n_env_cache, 3), dtype=torch.float32, device=cache_device),
        "target_clip_vec": torch.zeros((args.num_steps, n_env_cache, 512), dtype=torch.bfloat16, device=cache_device),
        # Add space for foveal tokens - exact size depends on your model
        "foveal_tokens": torch.zeros((args.num_steps, n_env_cache, args.agent_args.n_levels * (args.agent_args.window_size // (16*args.agent_args.pool_size)) * (args.agent_args.window_size // (16*args.agent_args.pool_size)), 384), dtype=torch.bfloat16, device=cache_device),
        "proprio": torch.zeros((args.num_steps, n_env_cache, 7), dtype=torch.float32, device=cache_device),
    }
    actions = torch.zeros((args.num_steps, n_env_cache, 2), dtype=torch.bfloat16, device=cache_device)
    joint_switch_actions = torch.zeros((args.num_steps, n_env_cache, 1), dtype=torch.bfloat16, device=cache_device)
    act_size = 7 if not args.agent_args.use_se3_actions else 10
    all_joint_actions = torch.zeros((args.num_steps, n_env_cache, args.agent_args.action_chunk_size, act_size), dtype=torch.float32, device=cache_device)
    logprobs = torch.zeros((args.num_steps, n_env_cache), device=cache_device)
    rewards = torch.zeros((args.num_steps, n_env_cache), device=cache_device)
    dones = torch.zeros((args.num_steps, n_env_cache), device=cache_device)
    values = torch.zeros((args.num_steps, n_env_cache), device=cache_device)
    # Start the game
    save_model_freq = 100  # Save model every 100 iterations
    num_update_epochs = 0
    start_time = time.time()
    for iteration in range(start_iteration, args.num_iterations + 1):
        next_obs, _ = env.reset(n_envs = args.num_envs)
        next_done = torch.zeros(args.num_envs).to(device)
        # --- Scheduler Step --- 
        for _ in range(args.num_envs * args.num_steps * dist.get_world_size()):
            eye_scheduler.step()
            hand_scheduler.step()
        # We turn off dropout for inference to get cleaner rewards
        if not args.rl_dropout:
            agent_module.eval()
        for step in range(0, args.num_steps):
            if is_distributed:
                global_step += args.num_envs * dist.get_world_size()
            else:
                global_step += args.num_envs

            # Store observations (modified to handle device placement)
            # Store directly on GPU
            obs["multicrop"][step, :args.num_envs] = next_obs["multicrop"].to(cache_device)
            obs["eye_direction"][step, :args.num_envs] = next_obs["eye_direction"].to(cache_device)
            obs["target_clip_vec"][step, :args.num_envs] = next_obs["target_clip_vec"].to(cache_device)
            obs["proprio"][step, :args.num_envs] = next_obs["proprio"].to(cache_device)
            dones[step, :args.num_envs] = next_done.to(cache_device)

            # ALGO LOGIC: action logic
            # Move observation to GPU for network forward pass# Get current window length based on training progress
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
                    past_obs.append(obs[k][s,:args.num_envs].to(device))
                # Add current observation
                past_obs.append(next_obs[k].to(device))
                # Stack along time dimension (T, B, ...)
                window_obs[k] = torch.stack(past_obs)
            # populate the foveal_tokens for window_len-1 frames
            if step > 0:
                window_obs['foveal_tokens'] = obs['foveal_tokens'][max(0, step - (current_window_length - 1)):step, :args.num_envs].to(device)
            torch.compiler.cudagraph_mark_step_begin()
            with torch.no_grad(), amp.autocast('cuda', dtype=torch.bfloat16):
                action, logprob, _, value, joint_actions = agent_module.get_action_and_value(window_obs, 
                                                                                                                window_length=current_window_length,
                                                                                                                inference=True)
            # Store foveal_tokens if they were computed
            if 'foveal_tokens' in window_obs:
                obs["foveal_tokens"][step, :args.num_envs] = window_obs['foveal_tokens'][-1].to(cache_device)
            action = action[-1]
            joint_actions = joint_actions[-1]
            if args.agent_args.use_se3_actions:
                # SE3 actions are already in world coordinates, no denormalization needed
                pass  # joint_actions remain as 10-DOF SE3 poses
            elif args.agent_args.relative_act:
                joint_actions = window_obs['proprio'][-1].unsqueeze(1).float() + DemonstrationData.denormalize_relative_data(joint_actions, method="mean_std").float()
            else:
                joint_actions = DemonstrationData.denormalize_predictions(joint_actions.float())
            value = value[-1]
            logprob = logprob[-1]
            # For context window, we take the last value and action (intermediates are ignored during rollout)
            values[step, :args.num_envs] = value.flatten().to(cache_device)
            actions[step, :args.num_envs] = action.to(cache_device)
            all_joint_actions[step, :args.num_envs] = joint_actions.to(cache_device)
            logprobs[step, :args.num_envs] = logprob.to(cache_device)

            # Execute the game and log data
            acts = {
                "eye": action.squeeze(0),
                "joint_pos": joint_actions.squeeze(0),
                "is_se3": args.agent_args.use_se3_actions
            }
            next_obs, reward, terminated, truncated, info = env.step(acts)
            joint_switch_actions[step, :args.num_envs] = info['joint_switch'].to(cache_device)
            next_done = torch.logical_or(terminated, truncated).to(
                device, torch.float32
            )
            rewards[step, :args.num_envs] = reward.to(cache_device).view(-1)

            # Store frames for video logging (new code)
            if step == 0:
                # Initialize frame storage at start of rollout
                train_frames = []

            # Save frames for first 4 environments
            if args.track and num_update_epochs % 10 == 0 and rank == 0:
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
                for i in range(num_envs_to_log):
                    if joint_switch_actions[step][i, 0] == 0:
                        # make red
                        frames[i, :, :50, :50] = torch.tensor([1, 0, 0.0]).view(3,1,1)
                    else:
                        # make green
                        frames[i, :, :50, :50] = torch.tensor([0, 1, 0.0]).view(3,1,1)
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
                wandb.log(
                    {
                        "train/video": wandb.Video(
                            train_frames_np[i], 
                            fps=30, 
                            format="mp4",
                        ),
                        "global_step": global_step,
                    }
                )

        # bootstrap value if not done (modified to handle GPU/CPU caching)
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
                history.append(obs[k][s, :args.num_envs].to(device))
            # Add final observation
            history.append(next_obs[k].to(device))
            window_obs[k] = torch.stack(history)
        torch.compiler.cudagraph_mark_step_begin()
        with torch.no_grad(), amp.autocast('cuda', dtype=torch.bfloat16):# Get current window length based on training progress
            next_value = agent_module.get_value(window_obs, window_length=current_window_length)[-1].reshape(1, -1).to(cache_device)
        advantages = torch.zeros_like(rewards)  # Already on the right device
        lastgaelam = 0
        for t in reversed(range(args.num_steps)):
            if t == args.num_steps - 1:
                nextnonterminal = 1.0 - next_done.to(cache_device)
                nextvalues = next_value.to(cache_device)
            else:
                nextnonterminal = 1.0 - dones[t + 1, :args.num_envs]
                nextvalues = values[t + 1, :args.num_envs]
            delta = (
                rewards[t, :args.num_envs] + args.gamma * nextvalues * nextnonterminal - values[t, :args.num_envs]
            )
            advantages[t, :args.num_envs] = lastgaelam = (
                delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            )
        returns = advantages + values

        # flatten the batch (keep on CPU until minibatch sampling)
        b_logprobs = logprobs[:args.num_steps, :args.num_envs].reshape(-1)
        b_advantages = advantages[:args.num_steps, :args.num_envs].reshape(-1)
        b_returns = returns[:args.num_steps, :args.num_envs].reshape(-1)
        b_values = values[:args.num_steps, :args.num_envs].reshape(-1)

        # Normalize advantages over the entire batch before starting epochs
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
            b_advantages = (b_advantages.to(device) - global_mean) / (global_std + 1e-6)
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
        # turn train back on 
        agent_module.train()
        for epoch in range(max(args.rl_update_epochs, args.bc_update_epochs)):
            np.random.shuffle(envinds)
            for start in range(0, args.num_envs, envsperbatch):
                end = start + envsperbatch
                mbenvinds = envinds[start:end]
                mb_inds = flatinds[:, mbenvinds].ravel()  # Gets indices for all timesteps for selected envs
                mb_start_frames = env.start_frames[mbenvinds].to(device)# N, 1
                # Transfer minibatch data to GPU
                b_obs_mb = {k: v[:args.num_steps,mbenvinds].to(device) for k, v in obs.items()}
                
                # Apply simple augmentations to multicrop images during training
                if args.use_augmentations and 'multicrop' in b_obs_mb:
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
            
                b_actions_mb = actions[:args.num_steps, mbenvinds].to(device)
                b_joint_switch_actions_mb = joint_switch_actions[:args.num_steps, mbenvinds].to(device)
                # Note: The agent should handle the window length internally during training
                # since we're passing the full sequence of observations
                torch.compiler.cudagraph_mark_step_begin()
                with amp.autocast('cuda', dtype=torch.bfloat16):
                    _, newlogprob, entropy, newvalue, joint_actions_mb = agent(
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
                    def schedule_entropy_coefficient(start_value, end_value, current_step, warmup_steps, decay_steps, total_steps):
                        """
                        Schedule entropy coefficient to decay only during the decay phase.
                        - Warmup + Stable phase: use start_value
                        - Decay phase: linearly decay from start_value to end_value
                        """
                        decay_start_step = total_steps - decay_steps

                        if current_step < decay_start_step:
                            # Warmup + stable phase: use start value
                            return start_value
                        else:
                            # Decay phase: linear decay
                            decay_progress = (current_step - decay_start_step) / decay_steps
                            decay_progress = min(decay_progress, 1.0)
                            return start_value + decay_progress * (end_value - start_value)

                    # Then use current_ent_coef instead of ent_coef in your loss calculation
                    current_ent_coef = schedule_entropy_coefficient(
                        args.ent_coef,
                        args.ent_coef_end,
                        global_step,
                        args.warmup_global_steps,
                        args.decay_global_steps,
                        args.total_timesteps
                    )
                    
                    entropy_loss = entropy.mean()
                    if epoch < args.rl_update_epochs:
                        loss = pg_loss - current_ent_coef * entropy_loss + v_loss * args.vf_coef
                    bc_loss = torch.tensor(0.0, device=device)
                    if epoch < args.bc_update_epochs:
                        # cumsum the jointswitches to get the frames for each env and step
                        # we multiply by fps_downsamp to get the frames in the original video
                        joint_switch_cumsum = torch.cumsum(b_joint_switch_actions_mb, dim=0) * args.fps_downsamp
                        # shift it by 1 since having an index of 1 only affects the NEXT frame
                        demo_frames = (mb_start_frames + torch.clip(joint_switch_cumsum - args.fps_downsamp, min=0)).long() # T N 1
                        count = 0
                        for i, env_ind in enumerate(mbenvinds):
                            # i stores the index within this minibatch (eg 0 or 1) and env_ind stores the index of the environment
                            # we index into current_demos using env_ind but all the _mb tensors use the indices i
                            gt_robot_data = env.current_demos[env_ind].robot_data['joint_and_gripper_data']
                            # Add padding at the beginning to maintain output size
                            padding = args.agent_args.action_chunk_size - 1
                            padded_data = torch.nn.functional.pad(gt_robot_data, (0, 0, 0, padding), mode='constant', value = torch.nan)
                            # Use unfold to create sliding windows
                            windows = padded_data.unfold(dimension=0, size=args.agent_args.action_chunk_size, step=1).permute(0, 2, 1) # D A 7
                            # clamp the frames to the range of data, which accounts for terminated episodes
                            clamped_frames = demo_frames[:,i,0].clamp(0, gt_robot_data.shape[0]-1)
                            gt_chunks = windows[clamped_frames] # T A 7
                            # ignore bc loss when cumsum overflows data
                            ignore_bc_on = (joint_switch_cumsum[:,i]>=gt_robot_data.shape[0]) | (joint_switch_cumsum[:,i]==0) # T 1
                            if ignore_bc_on.all():
                                continue
                            count += 1

                            # New predictions (already calculated by the forward pass)
                            b_joint_actions_new_mb = joint_actions_mb[:, i]
                        
                            # Handle SE3 vs joint space BC loss
                            if args.agent_args.use_se3_actions:
                                # Convert ground truth joint chunks to 9-DOF SE3 poses using compute_fk_batch
                                # gt_chunks shape: T, A, 7 (time, action_chunk, joints+gripper)
                                T, A, _ = gt_chunks.shape
                                gt_chunks_flat = gt_chunks.view(-1, 7).float().cpu().numpy()  # Flatten to (T*A, 7)
                                gt_chunks_jnp = jnp.array(onp.array(gt_chunks_flat))
                                
                                # Use gym's pyroki object for forward kinematics
                                ee_poses_wxyz_xyz = compute_fk_batch(gt_chunks_jnp, env.pk_robot_obj, env.ee_link_index)
                                ee_poses_matrix = jaxlie.SE3(ee_poses_wxyz_xyz).as_matrix()
                                ee_poses_torch = torch.from_numpy(onp.array(ee_poses_matrix)).to(device)
                                
                                # Convert from wxyz_xyz format to 9-DOF format (xyz + rotation matrix columns)
                                positions = ee_poses_torch[..., :3, 3]  # xyz (last 3 elements)
                                # Extract first two columns of rotation matrix for 9-DOF format
                                col1 = ee_poses_torch[..., :3, 0]  # First column
                                col2 = ee_poses_torch[..., :3, 1]  # Second column
                                # Add gripper information (last column of gt_chunks)
                                gripper = torch.from_numpy(gt_chunks_flat[:, -1:]).to(device)  # Extract gripper values
                                gripper_reshaped = gripper.view(T, A, 1)  # Reshape gripper to (T, A, 1)
                                
                                gt_se3_9dof = torch.cat([positions, col1, col2], dim=-1)  # 9-DOF format
                                gt_se3_9dof_reshaped = gt_se3_9dof.view(T, A, 9)  # Reshape to (T, A, 9)
                                gt_se3_chunks = torch.cat([gt_se3_9dof_reshaped, gripper_reshaped], dim=-1)  # Add gripper -> (T, A, 10)
                                
                                # Network predictions are ALREADY orthogonalized internally
                                predicted_se3 = b_joint_actions_new_mb  # Already orthogonal from agent
                                
                                # Simple comparison - SE3 poses are already well normalized
                                bc_loss_unclipped = torch.abs(gt_se3_chunks - predicted_se3)
                            else:
                                # Original joint space BC loss
                                if args.agent_args.relative_act:
                                    # Make ground truth chunks relative (subtract current proprio)
                                    current_proprio = b_obs_mb['proprio'][:, i]  # T, 7
                                    gt_chunks_relative = gt_chunks - current_proprio.unsqueeze(1)  # T, A, 7
                                    gt_chunks_normalized = DemonstrationData.normalize_relative_data(gt_chunks_relative, method = "mean_std")
                                else:
                                    gt_chunks_normalized = DemonstrationData.normalize_joint_data(gt_chunks.float())
                                bc_loss_unclipped = torch.abs(gt_chunks_normalized - b_joint_actions_new_mb)
                            bc_loss = bc_loss + bc_loss_unclipped[~ignore_bc_on.squeeze(-1)].nanmean()
                            
                        if count > 0:
                            bc_loss = bc_loss / count
                            loss = loss + bc_loss * args.bc_coef
                        loss = loss/args.grad_accumulation_steps
                    if rank == 0:
                        writer.add_scalar("charts/bc_loss", bc_loss.item(), global_step)
                loss.backward()
                # Only perform optimization step after accumulating gradients
                if (start + envsperbatch) % (envsperbatch * args.grad_accumulation_steps) == 0:
                    # nn.utils.clip_grad_norm_(agent_module.parameters(), args.max_grad_norm)
                    nn.utils.clip_grad_norm_(agent_module.get_eye_params(), args.max_grad_norm)
                    nn.utils.clip_grad_norm_(agent_module.get_hand_params(), args.max_grad_norm)
                    # Step optimizers
                    if epoch < args.bc_update_epochs and hand_optimizer:
                        hand_optimizer.step()
                        hand_optimizer.zero_grad()
                    if epoch < args.rl_update_epochs and eye_optimizer:
                        eye_optimizer.step()
                        eye_optimizer.zero_grad()

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
            # Log learning rates
            if eye_optimizer:
                writer.add_scalar(
                    "charts/learning_rate_eye", eye_optimizer.param_groups[0]["lr"], global_step
                )
            if hand_optimizer:
                writer.add_scalar(
                    "charts/learning_rate_hand", hand_optimizer.param_groups[0]["lr"], global_step
                )
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar(
                "charts/episodic_return", b_returns.mean().item(), global_step
            )
            
            #also log the percent of on switches
            writer.add_scalar("charts/percent_on_switches", (b_joint_switch_actions_mb==1).float().mean().item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            writer.add_scalar("losses/explained_variance", explained_var, global_step)
            writer.add_scalar("charts/entropy_coef", current_ent_coef, global_step)

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
            
            # Prepare optimizer and scheduler states
            eye_optimizer_state_dict = eye_optimizer.state_dict() if eye_optimizer else None
            hand_optimizer_state_dict = hand_optimizer.state_dict() if hand_optimizer else None
            eye_scheduler_state_dict = eye_scheduler.state_dict() if eye_scheduler else None
            hand_scheduler_state_dict = hand_scheduler.state_dict() if hand_scheduler else None
            
            save_dict = {
                "iteration": iteration,
                "model_state_dict": model_state_dict,
            }
            if eye_optimizer_state_dict:
                save_dict["eye_optimizer_state_dict"] = eye_optimizer_state_dict
            if hand_optimizer_state_dict:
                save_dict["hand_optimizer_state_dict"] = hand_optimizer_state_dict
            if eye_scheduler_state_dict:
                save_dict["eye_scheduler_state_dict"] = eye_scheduler_state_dict
            if hand_scheduler_state_dict:
                save_dict["hand_scheduler_state_dict"] = hand_scheduler_state_dict
                
            torch.save(save_dict, checkpoint_path)
            
            # Use AgentConfigManager to save config with checkpoint
            AgentConfigManager.save_config_with_checkpoint(args.agent_args, checkpoint_path)
            
            print(f"Saved model checkpoint to {checkpoint_path}")
            print(f"Saved agent config with AgentConfigManager")

    # Final checkpoint save only on rank 0
    if rank == 0:
        checkpoint_dir = f"runs/{run_name}"
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_final.pt")
        
        # Save the model state dict - if using DDP, need to access .module
        model_state_dict = agent_module.state_dict() if is_distributed else agent.state_dict()
        
        # Prepare optimizer and scheduler states
        eye_optimizer_state_dict = eye_optimizer.state_dict() if eye_optimizer else None
        hand_optimizer_state_dict = hand_optimizer.state_dict() if hand_optimizer else None
        eye_scheduler_state_dict = eye_scheduler.state_dict() if eye_scheduler else None
        hand_scheduler_state_dict = hand_scheduler.state_dict() if hand_scheduler else None
        
        save_dict = {
            "iteration": iteration,
            "model_state_dict": model_state_dict,
        }
        if eye_optimizer_state_dict:
            save_dict["eye_optimizer_state_dict"] = eye_optimizer_state_dict
        if hand_optimizer_state_dict:
            save_dict["hand_optimizer_state_dict"] = hand_optimizer_state_dict
        if eye_scheduler_state_dict:
            save_dict["eye_scheduler_state_dict"] = eye_scheduler_state_dict
        if hand_scheduler_state_dict:
            save_dict["hand_scheduler_state_dict"] = hand_scheduler_state_dict

        torch.save(save_dict, checkpoint_path)
        
        # Use AgentConfigManager to save config with checkpoint
        AgentConfigManager.save_config_with_checkpoint(args.agent_args, checkpoint_path)
        
        print(f"Saved model checkpoint to {checkpoint_path}")
        print(f"Saved agent config with AgentConfigManager")
        
        if writer:
            writer.close()

    # Clean up distributed process group
    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
