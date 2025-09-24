# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
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
from eye.sim.gyms import DemoEyeGym
from eye.sim.demo_data import DemonstrationData
from eye.camera import get_default_video_config
import cv2
from eye.agents import RobotAgent
from eye.agent_configs import RobotAgentConfig, AgentConfigManager
import math
from eye.foveal_encoders import crop_sizes_from_levels
from transformers import SiglipModel
import open_clip
import torchvision
import torch.amp as amp
from typing import List, Tuple, Optional
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import functional as F
from concurrent.futures import ThreadPoolExecutor
import sys


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
    wandb_entity: str = 'luv'
    """the entity (team) of wandb's project"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""
    num_envs: int = 10
    """the number of parallel game environments"""
    bc_num_envs: int = 15
    decoder_device: str = 'cpu'
    """the device to use for the decoder"""

    # Algorithm specific arguments
    total_timesteps: int = 750_000
    """total timesteps of the experiments"""
    learning_rate: float = 1e-4
    """the learning rate of the optimizer"""
    eye_lr_scale: float = 0.5
    num_steps: int = 10
    bc_num_steps: int = 10
    bc_num_minibatches: int = 5
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 2
    """the number of mini-batches"""
    grad_accumulation_steps: int = 1
    """the number of gradient accumulation steps"""
    update_epochs: int = 5
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    ent_coef_end: float = 0.01
    """end coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 1.0
    """the maximum norm for the gradient clipping"""
    target_kl: float = .02
    """the target KL divergence threshold"""

    # Agent configuration arguments
    agent_args: RobotAgentConfig = RobotAgentConfig(
        crop_size=360,
        action_chunk_size=60,
        proprio_dropout=0.1,
        num_blocks=3,
        relative_act=False
    )
    """Agent configuration parameters"""
    
    # Other training specific arguments
    freeze_encoder: bool = True
    vit_type: str = "dino" 
    n_scales: int = 1
    load_eye_weights_from: Optional[str] = None
    proprio_noise_std: float = 0.0
    use_learnable_joint_token: bool = True
    n_cycles: int = 1 # number of cycles for cosine annealing scheduler
    """Number of cycles for the cosine annealing learning rate scheduler"""
    interleave_bcrl: bool = False
    fps_downsamp: int = 20
    """downsample factor for the environment, video+actions. action chunk predictions are NOT downsampled, they are kept at original FPS"""
    eye_ema_smoothing: float = 0.0

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""
    warmup_steps: int = 20
    """number of steps for learning rate warmup"""

    # Add DDP-specific arguments
    local_rank: int = field(default_factory=lambda: int(os.environ.get("LOCAL_RANK", -1)))
    """local rank for distributed training"""

    gpu_cache: bool = True
    """if toggled, environment data will be cached on GPU instead of CPU"""
    load_spherical: bool = False
    """if toggled, load the spherical video"""
    load_exo: bool = True
    """if toggled, load the exo video"""
    load_wrist: bool = True
    """if toggled, load the wrist video"""
    image_size: int = 360



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
    args.total_timesteps = int(args.total_timesteps / world_size)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
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
    # model, _, _ = open_clip.create_model_and_transforms(
    #         "ViT-B-16", pretrained="laion2b_s34b_b88k", device=device, precision='fp16'
    #     )
    # model.eval()
    # preprocess = torchvision.transforms.Normalize(
    #     mean=[0.48145466, 0.4578275, 0.40821073],
    #     std=[0.26862954, 0.26130258, 0.27577711],
    # )
    # tokenizer = open_clip.get_tokenizer("ViT-B-16")
    import glob
    demo_path_list=glob.glob('data/demos/eraser_*/processed/*')
    # env = DemoEyeGym(demo_path_list=glob.glob('data/demos/towel_bucket*/processed/*')+glob.glob('data/demos/towel_hard*/processed/*'), 
    # env = DemoEyeGym(demo_path_list=glob.glob('data/demos/estop*/processed/*'), 
    
    # env = DemoEyeGym(demo_path_list=glob.glob('data/demos/boris_follow_test1/processed/*'), 
    #                  n_envs=args.num_envs, local_rank=args.local_rank)

    # Initialize agent config
    agent_config = RobotAgentConfig(
        device=device,
        crop_size=args.image_size,  # Keep using image_size for RobotAgent
        action_chunk_size=args.agent_args.action_chunk_size,
        proprio_dropout=args.agent_args.proprio_dropout,
        num_blocks=args.agent_args.num_blocks,
        relative_act=args.agent_args.relative_act
    )
    
    # Initialize agent with config
    agent = RobotAgent(config=agent_config).to(device)

    hand_optimizer = optim.AdamW(
        [p for p in agent.parameters() if p.requires_grad],
        lr=args.learning_rate,
        eps=1e-5,
        weight_decay=0.01,
        betas=(0.9, 0.95),
    )

    hand_scheduler = optim.lr_scheduler.SequentialLR(
        hand_optimizer,
        schedulers=[
            # Warmup phase
            optim.lr_scheduler.LinearLR(
                hand_optimizer,
                start_factor=0.9,  # Start at 10% of base lr
                end_factor=1.0,
                total_iters=args.warmup_steps,
            ),
            # Cosine Annealing with Restarts phase
            optim.lr_scheduler.CosineAnnealingWarmRestarts(
                hand_optimizer,
                T_0=(args.num_iterations - args.warmup_steps) // args.n_cycles if args.num_iterations > args.warmup_steps else 1,
                eta_min=args.learning_rate * 0.1,  # Minimum learning rate (10% of initial LR)
            ),
        ],
        # Milestone is the step number where we switch schedulers
        milestones=[args.warmup_steps],
    )
    
    executor = ThreadPoolExecutor(max_workers=8)
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
            
            agent.load_state_dict(new_state_dict, strict=False)
            print("Successfully loaded weights from checkpoint")
        else:
            print("Warning: No model_state_dict found in checkpoint")
        # Load optimizer state
        if "hand_optimizer_state_dict" in checkpoint:
            hand_optimizer.load_state_dict(checkpoint["hand_optimizer_state_dict"])
        
        # Load scheduler state
        if "hand_scheduler_state_dict" in checkpoint:
            hand_scheduler.load_state_dict(checkpoint["hand_scheduler_state_dict"])

        
        # Resume iteration count if available
        if "iteration" in checkpoint:
            start_iteration = checkpoint["iteration"] + 1
            # Update global step
            global_step = start_iteration * args.batch_size
            print(f"Resuming from iteration {start_iteration}")
        start_iteration = 1
    else:
        start_iteration = 1

    
    # Wrap agent in DDP if using distributed training
    if is_distributed:
        agent = DDP(agent, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    
    # Replace the optimizer and add scheduler setup
    global_step = 0
    # After creating the optimizer and scheduler, add this code to load from checkpoint:
    # Storage setup - modified to include foveal_tokens

    # Start the game
    save_model_freq = 100  # Save model every 100 iterations
    num_update_epochs = 0
    start_time = time.time()
    for iteration in range(start_iteration, args.num_iterations + 1):
        path_ids = torch.randint(0, len(demo_path_list), (args.num_envs,))
        current_demos = []
        current_frames = []
        start_frames = torch.zeros((args.num_envs,1), device=device, dtype=torch.int32)
        for i, path_id in enumerate(path_ids):
            video_path = glob.glob(os.path.join(demo_path_list[path_id], "downsampled.mp4"))[0]
            h5_path = glob.glob(os.path.join(demo_path_list[path_id], "*.h5"))[0]
            current_demos.append(DemonstrationData(video_path, h5_path, local_rank=args.local_rank, executor=executor, 
                                                   fps_downsamp=args.fps_downsamp, n_scales=None, load_spherical=False, load_exo=args.load_exo, load_wrist=args.load_wrist))
            safety_margin_size = args.num_steps*args.fps_downsamp
            current_frames.append(random.randint(0, max(current_demos[i].robot_data["joint_and_gripper_data"].shape[0] - safety_margin_size - 2,0)))
            start_frames[i] = current_frames[-1]

        if args.anneal_lr:
            hand_scheduler.step()
        if args.load_wrist:
            wrist_images = torch.zeros((args.num_steps, args.num_envs, 3, 720, 1280), device=device, dtype = torch.bfloat16)
        if args.load_exo:
            exo_images = torch.zeros((args.num_steps, args.num_envs, 3, 720, 1280), device=device, dtype = torch.bfloat16)
        all_proprio = torch.zeros((args.num_steps, args.num_envs, 7), device=device, dtype = torch.bfloat16)
        for step in range(0, args.num_steps):
            for e, demo in enumerate(current_demos):
                if args.load_exo:
                    full_frame = demo.exo_video[current_frames[e]] / 255.0
                    crop_width = 850
                    crop_height = int(crop_width/2)
                    #crop it from the bottom
                    frame = full_frame[:,full_frame.shape[1]-crop_height:,full_frame.shape[2]//2-crop_width//2:full_frame.shape[2]//2+crop_width//2]
                    # resize back to 1280x720
                    frame = frame.to(device, dtype=torch.bfloat16)
                    frame = F.interpolate(frame.unsqueeze(0), size=(720, 1280), mode='bilinear', align_corners=False).squeeze()
                    exo_images[step, e] = frame
                if args.load_wrist:
                    wrist_images[step, e] = (demo.wrist_video[current_frames[e]] / 255.0).to(device, dtype=torch.bfloat16)
                all_proprio[step, e] = demo.robot_data["joint_and_gripper_data"][current_frames[e]]
                current_frames[e] += args.fps_downsamp
        if args.track and rank == 0 and num_update_epochs % 10 == 0:
            if args.load_exo and args.load_wrist:
                from einops import rearrange
                imgs = torch.stack([exo_images, wrist_images], dim=2)
                imgs = rearrange(imgs, 't b n_img c h w -> b t c (n_img h) w', n_img = 2)
                train_frames_np = imgs.float().cpu().numpy()*255 # Shape: (B,T,3,2*H,W)
            elif args.load_exo:
                train_frames_np = exo_images.permute(1,0,2,3,4).float().cpu().numpy()*255 # Shape: (B,T,3,H,W)
            elif args.load_wrist:
                train_frames_np = wrist_images.permute(1,0,2,3,4).float().cpu().numpy()*255 # Shape: (B,T,3,H,W)
            for i in range(min(train_frames_np.shape[0],4)):
                wandb.log(
                    {
                        "train/video": wandb.Video(
                            train_frames_np[i,:,:,::2,::2], 
                            fps=30, 
                            format="mp4",
                        ),
                        "global_step": global_step,
                    }
                )
        obs = {"proprio": all_proprio}
        if args.load_exo:
            obs["exo_image"] = exo_images
        if args.load_wrist:
            obs["wrist_image"] = wrist_images
        # Optimizing the policy and value network
        envsperbatch = args.num_envs // args.num_minibatches
        envinds = np.arange(args.num_envs)
        num_update_epochs += 1
        for epoch in range(args.update_epochs):
            np.random.shuffle(envinds)
            for start in range(0, args.num_envs, envsperbatch):
                end = start + envsperbatch
                mbenvinds = envinds[start:end]
                mb_start_frames = start_frames[mbenvinds].to(device)# N, 1
                # Transfer minibatch data to GPU
                with amp.autocast('cuda', dtype=torch.bfloat16):
                    b_obs_mb = {k: v[:args.num_steps,mbenvinds].to(device) for k, v in obs.items()}
                    b_joint_switch_actions_mb = torch.ones((args.num_steps, mbenvinds.shape[0], 1), device=device, dtype = torch.int32)
                    # Note: The agent should handle the window length internally during training
                    # since we're passing the full sequence of observations
                    joint_actions_mb = agent(
                        b_obs_mb
                    )
                    # if args.track and rank == 0 and num_update_epochs % 10 == 1 and epoch==0 and start == 0:
                    #     from eye.sim.act_visualizer import visualize_action_chunk
                    #     fps_downsamp = args.fps_downsamp
                    #     gt_chunks = current_demos[mbenvinds[0]].robot_data['joint_and_gripper_data']
                    #     og_data_length = gt_chunks.shape[0]
                    #     vis_time_indices = torch.linspace(start_frames[mbenvinds[0]].item(), start_frames[mbenvinds[0]].item() + fps_downsamp*(args.num_steps - 1), 3).long()
                    #     #pad gt chunks with zeros
                    #     gt_chunks = torch.nn.functional.pad(gt_chunks, (0, 0, 0, args.action_chunk_size - 1), mode='constant', value = torch.nan)
                    #     gt_chunks = gt_chunks.unfold(dimension=0, size=args.action_chunk_size, step=1).permute(0, 2, 1) # D A 7
                    #     vis_time_indices = torch.minimum(vis_time_indices, torch.tensor(og_data_length - 1))
                    #     gt_chunks = gt_chunks[vis_time_indices]
                    #     gt_chunks = gt_chunks.float().cpu().numpy()
                    #     save_folder = wandb.run.dir
                    #     # Once for the joint actions
                    #     output_chunks = joint_actions_mb[(vis_time_indices.cuda() - start_frames[mbenvinds[0]])//fps_downsamp, 0].detach().float().cpu().numpy()
                    #     out_file = visualize_action_chunk(DemonstrationData.denormalize_predictions(gt_chunks), DemonstrationData.denormalize_predictions(output_chunks), save_folder)
                    #     #add the mp4 to wandb
                    #     wandb.log({
                    #         "train/robot_video": wandb.Video(out_file, format="mp4"),
                    #         "global_step": global_step,
                    #     })


                    bc_loss = torch.tensor(0.0, device=device)

                    # cumsum the jointswitches to get the frames for each env and step
                    # we multiply by fps_downsamp to get the frames in the original video
                    joint_switch_cumsum = torch.cumsum(b_joint_switch_actions_mb, dim=0)*args.fps_downsamp
                    # shift it by 1 since having an index of 1 only affects the NEXT frame
                    joint_switch_cumsum = torch.clip(joint_switch_cumsum - args.fps_downsamp, min=0)
                    demo_frames = (mb_start_frames + joint_switch_cumsum).long() # T N 1
                    count = 0
                    for i, env_ind in enumerate(mbenvinds):
                        # i stores the index within this minibatch (eg 0 or 1) and env_ind stores the index of the environment
                        # we index into current_demos using env_ind but all the _mb tensors use the indices i
                        gt_robot_data = current_demos[env_ind].robot_data['joint_and_gripper_data']
                        # Add padding at the beginning to maintain output size
                        padding = args.action_chunk_size - 1
                        padded_data = torch.nn.functional.pad(gt_robot_data, (0, 0, 0, padding), mode='constant', value = torch.nan)
                        # Use unfold to create sliding windows
                        windows = padded_data.unfold(dimension=0, size=args.action_chunk_size, step=1).permute(0, 2, 1) # D A 7
                        # clamp the frames to the range of data, which accounts for terminated episodes
                        clamped_frames = demo_frames[:,i,0].clamp(0, gt_robot_data.shape[0]-1)
                        gt_chunks = windows[clamped_frames] # T A 7
                        # ignore bc loss when cumsum overflows data
                        ignore_bc_on = (joint_switch_cumsum[:,i]>=gt_robot_data.shape[0]) # T 1
                        if ignore_bc_on.all():
                            print("Ignoring uh oh")
                            continue
                        count += 1
                        b_joint_actions_new_mb = joint_actions_mb[:, i]

                        # Calculate unclipped loss
                        bc_loss_unclipped = torch.abs(gt_chunks - b_joint_actions_new_mb)
                        bc_loss = bc_loss + bc_loss_unclipped[~ignore_bc_on.squeeze(-1)].nanmean()
                            
                    if count > 0:
                        bc_loss = bc_loss / count
                    if rank == 0:
                        writer.add_scalar("charts/bc_loss", bc_loss.item(), global_step)
                bc_loss = bc_loss/args.grad_accumulation_steps
                bc_loss.backward()

                # Only perform optimization step after accumulating gradients
                if (start + envsperbatch) % (envsperbatch * args.grad_accumulation_steps) == 0:
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    hand_optimizer.step()
                    hand_optimizer.zero_grad()


        
        global_step += args.num_envs*args.num_steps*dist.get_world_size()
        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if rank == 0:
            
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
            model_state_dict = agent.module.state_dict() if is_distributed else agent.state_dict()
            
            torch.save(
                {
                    "iteration": iteration,
                    "model_state_dict": model_state_dict,
                    "hand_optimizer_state_dict": hand_optimizer.state_dict(),
                    "hand_scheduler_state_dict": hand_scheduler.state_dict(),
                },
                checkpoint_path,
            )
            
            # Save agent config as JSON
            config_path = os.path.join(checkpoint_dir, f"checkpoint_{iteration}_config.json")
            agent_config.save(config_path)
            
            print(f"Saved model checkpoint to {checkpoint_path}")
            print(f"Saved agent config to {config_path}")

    # Final checkpoint save only on rank 0
    if rank == 0:
        checkpoint_dir = f"runs/{run_name}"
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_final.pt")
        
        # Save the model state dict - if using DDP, need to access .module
        model_state_dict = agent.module.state_dict() if is_distributed else agent.state_dict()
        
        torch.save(
            {
                "iteration": iteration,
                "model_state_dict": model_state_dict,
                "hand_optimizer_state_dict": hand_optimizer.state_dict(),
                "hand_scheduler_state_dict": hand_scheduler.state_dict(),
            },
            checkpoint_path,
        )
        
        # Save agent config as JSON
        config_path = os.path.join(checkpoint_dir, "checkpoint_final_config.json")
        agent_config.save(config_path)
        
        print(f"Saved model checkpoint to {checkpoint_path}")
        print(f"Saved agent config to {config_path}")
        
        if writer:
            writer.close()

    # Clean up distributed process group
    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
