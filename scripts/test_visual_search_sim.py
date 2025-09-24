from viser import ViserServer
import torch
import numpy as np
import viser.transforms as vtf
import time
import cv2
from typing import Tuple
from eye.camera import radial_and_tangential_undistort, get_default_video_config
import math
from eye.sim.spherical_video import SphericalVideo
from eye.sim.rewards import ClipReward
import argparse
from collections import deque
from pathlib import Path
import open_clip
import torchvision
import plotly.graph_objects as go
import plotly.express as px
import torch.amp as amp

from einops import rearrange
from eye.image_utils import pad_to_aspect_ratio
from eye.transforms import SO3
from eye.agents import VisualSearchAgent
from eye.agent_configs import VisualSearchAgentConfig, AgentConfigManager
from eye.foveal_encoders import crop_sizes_from_levels


class VisualSearchSimulator:
    def __init__(self, video_path, ckpt_path=None):
        self.server = ViserServer()
        
        # Video and rendering setup
        w = 1920
        h = 1200
        K, dist_coeffs, is_fisheye = get_default_video_config(w, h)
        self.crop_sizes = crop_sizes_from_levels(4, 320, 1200)
        self.sim = SphericalVideo(video_path, K, dist_coeffs, h, w, is_fisheye, "cuda", self.crop_sizes, decoder_device="cpu", window_size=256)
        self.sim.reset(True)
        
        # Agent setup
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        
        # Initialize CLIP model for text conditioning
        self.clip_model, _, _ = open_clip.create_model_and_transforms(
            "ViT-B-16",
            pretrained="laion2b_s34b_b88k",
            device=self.device,
            precision="fp16",
        )
        self.clip_model.eval()
        self.tokenizer = open_clip.get_tokenizer("ViT-B-16")
        
        # Initialize agent
        if ckpt_path and Path(ckpt_path).exists():
            self.load_agent_from_checkpoint(ckpt_path)
        else:
            assert False, "No checkpoint provided"
        
        self.window_length = self.agent.config.window_length
        self.window_size = self.agent.config.window_size
        
        # Initialize observation history
        self.observation_history = {
            "multicrop": [],
            "eye_direction": [],
            "target_clip_vec": []
        }
        
        # Warmup counter
        self.timestep_counter = 0
        
        # No separate state - always use viser camera
        
        # Setup GUI controls
        self.setup_gui()
        
        # Initialize with default text prompt
        self.update_target_embedding()
        
        # Benchmarking
        self.render_times = deque(maxlen=100)
        self.frame_count = 0
        
    def load_agent_from_checkpoint(self, ckpt_path):
        """Load agent from checkpoint file"""
        # Try to load config from JSON file
        config_path = str(ckpt_path).replace('.pt', '_config.json')
        agent_config = None
        
        if Path(config_path).exists():
            try:
                agent_config = VisualSearchAgentConfig.from_json(config_path)
                print(f"Loaded agent config from {config_path}")
            except Exception as e:
                print(f"Warning: Could not load config from {config_path}: {e}")
        
        # If no config file, try loading from checkpoint
        if agent_config is None:
            agent_config = AgentConfigManager.load_config_from_checkpoint(ckpt_path)
            if agent_config:
                print(f"Loaded agent config from checkpoint {ckpt_path}")
        
        # If still no config, use defaults
        if agent_config is None:
            raise ValueError("No agent config found")
        else:
            agent_config.device = self.device
            
        print(f"Initializing agent with config: {agent_config}")
        self.agent = VisualSearchAgent(config=agent_config)
        self.agent.to(self.device)
        
        # Load state dict
        state_dict = torch.load(ckpt_path, map_location=self.device)
        model_state_dict = state_dict["model_state_dict"]
        
        # Handle _orig_mod keys from torch.compile()
        if any("_orig_mod" in key for key in model_state_dict.keys()):
            print("Detected _orig_mod keys, removing prefix...")
            new_state_dict = {}
            for key, value in model_state_dict.items():
                new_key = key.replace("_orig_mod.", "")
                new_state_dict[new_key] = value
            model_state_dict = new_state_dict
        self.agent.load_state_dict(model_state_dict, strict=True)
        self.agent.eval()
        print(f"Successfully loaded agent from {ckpt_path}")
            
    
    def setup_gui(self):
        """Setup viser GUI controls"""
        # Video info
        self.video_name = self.server.gui.add_text("Video", str(self.sim.video_path.name))
        
        # Separate pause controls
        self.video_playing = self.server.gui.add_checkbox("Video Playing", False)
        self.policy_playing = self.server.gui.add_checkbox("Policy Playing", False)
        
        # Deterministic inference control
        self.deterministic_inference = self.server.gui.add_checkbox("Deterministic Inference", False)
        
        # Text input for CLIP conditioning
        self.text_input = self.server.gui.add_text(
            "Search Target", 
            initial_value="expo eraser"
        )
        self.update_button = self.server.gui.add_button("Update Target")
        self.idx_viser = self.server.gui.add_slider("Layer for attn",0,2,1,2)
        self.head_viser = self.server.gui.add_slider("Head for attn",0,11,1,11)
        self.viser_plotly = self.server.gui.add_plotly(px.imshow(np.zeros((10,10))))
        # Preset dropdown for convenience
        self.preset_dropdown = self.server.gui.add_dropdown(
            "Presets",
            options=[
                "a red button",
                "a brown bear wearing an orange shirt",
                "a human face",
                "a person's feet",
                "the floor of a room",
                "the ceiling of a room",
                "a towel",
                "a plastic tray",
            ],
            initial_value="a red button"
        )
        
        # Manual control info (when policy is paused, use mouse to control camera)
        self.control_info = self.server.gui.add_text(
            "Manual Control", 
            "When policy paused: use mouse to control camera viewpoint"
        )
        
        # Stats display
        self.stats_text = self.server.gui.add_text("Stats", "FPS: 0.0")
        
        # Setup callbacks
        @self.update_button.on_click
        def _(_):
            self.update_target_embedding()
            
        @self.preset_dropdown.on_update
        def _(_):
            self.text_input.value = self.preset_dropdown.value
            self.update_target_embedding()
    
    def update_target_embedding(self):
        """Update CLIP embedding for the target text"""
        text = self.text_input.value
        if not text.strip():
            return
            
        with torch.no_grad():
            tokens = self.tokenizer([text]).to(self.device)
            embedding = self.clip_model.encode_text(tokens)
            embedding /= embedding.norm(dim=-1, keepdim=True)
            self.target_embedding = embedding.float()
            print(f"Updated target: '{text}'")
    
    def get_camera_rotation(self, clients):
        """Always get rotation from viser camera"""
        if len(clients) > 0:
            client = clients[max(clients.keys())]
            return SO3(torch.from_numpy(client.camera.wxyz).float().to(self.device))
        else:
            # Fallback to identity if no clients
            return SO3.identity(device=self.device, dtype=torch.float32)
    
    def get_current_observation(self, rotation):
        """Get current observation for the agent"""
        # Render multicrop from current viewpoint
        multicrop = self.sim.render_multicrop(rotation)
        
        # Get eye direction (forward vector from rotation)
        eye_direction = rotation.as_matrix()[:, 2]
        
        return {
            "multicrop": multicrop,
            "eye_direction": eye_direction,
            "target_clip_vec": self.target_embedding,
        }
    
    def update_observation_history(self, obs):
        """Update observation history buffer"""
        for key, value in obs.items():
            self.observation_history[key].append(value)
            # Keep only the most recent window_length observations
            if len(self.observation_history[key]) > self.window_length:
                self.observation_history[key].pop(0)
    
    def create_window_observation(self):
        """Create windowed observation for the agent"""
        if len(self.observation_history["multicrop"]) < self.window_length:
            return None
            
        window_obs = {}
        for key in self.observation_history:
            if key == "multicrop":
                # Stack to (T, n_scales, 3, H, W) then add batch dim
                stacked = torch.stack(self.observation_history[key], dim=0)
                window_obs[key] = stacked.unsqueeze(1)  # (T, 1, n_scales, 3, H, W)
            elif key == "eye_direction":
                # Stack to (T, 3) then add batch dim
                stacked = torch.stack(self.observation_history[key], dim=0)
                window_obs[key] = stacked.unsqueeze(1)  # (T, 1, 3)
            elif key == 'target_clip_vec':
                # Repeat across time dimension
                window_obs[key] = self.observation_history[key][0].repeat(len(self.observation_history[key]), 1, 1)
                
        return window_obs
    
    def get_agent_action(self):
        """Get action from the agent"""
        # Always run inference for JIT compilation, but only apply action after warmup
        window_obs = self.create_window_observation()
        if window_obs is None:
            return np.array([0.0, 0.0])
        attn_hook = []
        with torch.no_grad():
            with amp.autocast('cuda', dtype=torch.bfloat16):
                action, _, _, _ = self.agent.get_action_and_value(
                    window_obs, deterministic=self.deterministic_inference.value, attention_hook=attn_hook
                )
            # Take action from last timestep
            action = action[-1].squeeze().float().cpu().numpy()
        all_attn = torch.stack(attn_hook, dim=0) # shape is N_layers, B, n_head, L, L (here B is 1)
        l_idx = self.idx_viser.value
        h_idx = self.head_viser.value
        attn_weights = all_attn[l_idx,0,h_idx] # shape is L, L
        T = 0
        actor_to_img_attn = attn_weights[:1,(256+3)*T+3:(256+3)*T+3+256]
        actor_to_img_attn = rearrange(actor_to_img_attn, '1 (n h w) -> n h w', n=4,h=8,w=8)
        top_row = torch.concatenate([actor_to_img_attn[3], actor_to_img_attn[2]], axis=1)
        bottom_row = torch.concatenate([actor_to_img_attn[1], actor_to_img_attn[0]], axis=1)
        actor_to_img_attn = torch.concatenate([top_row, bottom_row], axis=0)

        self.viser_plotly.figure = px.imshow(actor_to_img_attn.cpu().numpy())
        
        # Only return action if we have enough history, otherwise return zero
        if self.timestep_counter >= self.window_length:
            return action
        else:
            return np.array([0.0, 0.0])
    
    def apply_action(self, action, clients):
        """Apply action to update viewpoint by updating viser camera directly"""
        if len(clients) == 0:
            return
        
        # Get current camera rotation
        client = clients[max(clients.keys())]
        current_rotation = SO3(torch.from_numpy(client.camera.wxyz).float().to(self.device))
        
        # Apply action directly using SO3 transforms like live_eval.py
        delta_azimuth = torch.tensor(action[0], dtype=torch.float32, device=self.device)
        delta_elevation = torch.tensor(action[1], dtype=torch.float32, device=self.device)
        
        # Update rotation: new_R = SO3.from_z_radians(delta_azimuth) @ R @ SO3.from_x_radians(delta_elevation)
        new_rotation = SO3.from_z_radians(delta_azimuth) @ current_rotation @ SO3.from_x_radians(delta_elevation)
        
        # Update viser camera directly
        client.camera.wxyz = new_rotation.wxyz.cpu().numpy()
    
    def run(self):
        """Main loop"""
        print("Starting visual search simulator...")
        print("Controls:")
        print("- Video Playing: Controls video playback")
        print("- Policy Playing: Controls agent policy")
        print("- When policy is paused, use mouse to control camera viewpoint")
        print("- Use text input or presets to change search target")
        
        while True:
            clients = self.server.get_clients()
            if len(clients) == 0:
                time.sleep(0.01)
                continue
            time.sleep(1/30)
            
            # Get current rotation (from camera or policy)
            current_rotation = self.get_camera_rotation(clients)
            
            # Get current observation
            start_time = time.perf_counter()
            current_obs = self.get_current_observation(current_rotation)
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            
            # Track render time
            render_time = end_time - start_time
            self.render_times.append(render_time)
            self.frame_count += 1
            
            # Update observation history
            self.update_observation_history(current_obs)
            
            # Increment timestep counter
            self.timestep_counter += 1
            
            # Get agent action if policy is playing
            action = self.get_agent_action()
            if self.policy_playing.value:
                self.apply_action(action, clients)
            
            # Advance video if video is playing
            if self.video_playing.value:
                if not self.sim.advance():
                    print("Video ended, resetting to beginning")
                    self.sim.reset()
            
            # Render visualization
            multicrop = current_obs["multicrop"].squeeze(0)  # Remove batch dimension
            n_crops = multicrop.shape[0]
            
            # Create 2x2 tiled visualization
            if n_crops >= 4:
                top_row = torch.concatenate([multicrop[3], multicrop[2]], axis=2)
                bottom_row = torch.concatenate([multicrop[1], multicrop[0]], axis=2)
                tiled_multicrop = torch.concatenate([top_row, bottom_row], axis=1)
            elif n_crops == 2:
                tiled_multicrop = torch.concatenate([multicrop[1], multicrop[0]], axis=2)
            else:
                tiled_multicrop = multicrop[0] if n_crops > 0 else torch.zeros(3, 224, 224)
            
            # Convert to image
            view = (255 * tiled_multicrop).byte().cpu().numpy().transpose(1, 2, 0)
            
            # Pad to client aspect ratio
            client = clients[max(clients.keys())]
            aspect = client.camera.aspect
            view = pad_to_aspect_ratio(view, aspect)
            
            # Set background
            self.server.scene.set_background_image(view, jpeg_quality=99)
            
            # Update stats every 30 frames
            if self.frame_count % 30 == 0:
                avg_time = sum(self.render_times) / len(self.render_times)
                fps = 1.0 / avg_time if avg_time > 0 else 0
                self.stats_text.value = f"FPS: {fps:.1f} | Render: {avg_time*1000:.1f}ms"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visual search simulator with RL agent')
    parser.add_argument('--video', type=Path, help='Path to video file or YouTube ID', 
                       default=Path("downloads/boris_tracker.mp4"))
    parser.add_argument('--checkpoint', type=Path, help='Path to agent checkpoint', 
                       default=None)
    args = parser.parse_args()
    
    # Check if it's a YouTube ID
    if args.video.suffix == "":
        print(f"Using YouTube video ID: {args.video}")
    else:
        print(f"Using local video: {args.video}")
    
    if args.checkpoint:
        print(f"Loading agent from: {args.checkpoint}")
    else:
        print("Using default agent (no checkpoint provided)")
    
    simulator = VisualSearchSimulator(args.video, args.checkpoint)
    
    try:
        simulator.run()
    except KeyboardInterrupt:
        print("\nShutting down...")