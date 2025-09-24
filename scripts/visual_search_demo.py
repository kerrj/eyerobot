import threading
from eye.eyeball import Eyeball
from eye.zmq_tools import Subscriber
import numpy as np
import time
import torch
import os
from eye.foveal_encoders import create_foveated_batch
from eye.agents import VisualSearchAgent
from eye.agent_configs import VisualSearchAgentConfig, AgentConfigManager
from eye.transforms import SO3
import matplotlib.pyplot as plt
from eye.foveal_encoders import crop_sizes_from_levels
from viser import ViserServer
import open_clip
import torchvision
import torch.amp as amp
import os
from pathlib import Path

# Set PyTorch Dynamo cache size to prevent memory issues
torch._dynamo.config.cache_size_limit = 64

class EyeController:
    def __init__(self, ckpt_paths_dict):
        self.ckpt_paths_dict = ckpt_paths_dict
        # Determine the initial model name (e.g., the first one in the dict)
        if not self.ckpt_paths_dict:
             raise ValueError("ckpt_paths_dict cannot be empty")
        self.current_model_name = list(self.ckpt_paths_dict.keys())[0]
        initial_ckpt_path = self.ckpt_paths_dict[self.current_model_name] # We still need the path for the print statement
        print(f"Initializing with model: {self.current_model_name} from {initial_ckpt_path}")


        self.eye = Eyeball(P_speed_1=1500, I_speed_1=30, P_speed_2=1500, I_speed_2=30)
        self.frame_subscriber = Subscriber("ipc:///tmp/eye_frame")
        self.last_available_frame = None
        self.frame_lock = threading.Lock()
        self.is_running = True
        self.count = 0
        self.frame_subscriber.on_recv_bytes(self.on_frame)
        self.eye_elev_bound = (-50, 85)
        self.eye_azim_bound = (-180, 50)
        # Initialize timestep counter for warmup period
        self.timestep_counter = 0
        self.eye_elev_bound = (-20, 20)
        self.eye_azim_bound = (50, 310)
        # Initialize EMA smoothing for motor commands
        self.ema_alpha = 0.5  # Smoothing factor (0-1): higher = more weight to current action
        self.prev_azimuth_cmd = 0
        self.prev_elev_cmd = 0

        # Add a model loading lock to prevent concurrent access
        self.model_loading_lock = threading.Lock()

        # Initialize the RL agent parameters
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.window_size = 224
        self.crop_sizes = crop_sizes_from_levels(4, 224, 1920)
        self.agent = None # Agent will be initialized in _load_model
        self.window_length = 10 # Default, will be updated in _load_model if agent needs different


        # Initialize observation history buffer
        # Length is dynamic based on self.window_length
        self.observation_history = {
            "multicrop": [],
            "eye_direction": [],
            "target_clip_vec": [],
            "foveal_tokens": [] # Add buffer for foveal tokens
        }

        self.viser_server = ViserServer()

        # Initialize CLIP model
        self.model, _, _ = open_clip.create_model_and_transforms(
            "ViT-B-16",
            pretrained="laion2b_s34b_b88k",
            device=self.device,
            precision="fp16",
        )
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer("ViT-B-16")

        # --- GUI Controls ---
        # Model selection dropdown
        self.model_selector = self.viser_server.gui.add_dropdown(
            "Select Model",
            options=list(self.ckpt_paths_dict.keys()),
            initial_value=self.current_model_name
        )
        self.model_selector.on_update(self._on_model_select) # Register callback

        # Add GUI controls for CLIP prompting
        self.prompt_box = self.viser_server.gui.add_dropdown(
            "Prompt Presets",
            options=[
                "a brown bear wearing an orange shirt;stuffed bear;boris bear character",
                "a human;a face;a person;a human head",
                "a person's feet;a person's legs",
                "the floor of a room;the floor",
                "the ceiling of a room",
                "a towel",
                "a plastic tray",
                "A red button"
            ],
            initial_value="A red button"
        )
        self.change_prompt = self.viser_server.gui.add_button("Update Prompt")
        self.pause = self.viser_server.gui.add_checkbox("Pause", False)
        # --- End GUI Controls ---

        print("SETTING TARGET TO 0")
        self.target_embedding = torch.zeros((1,512), device=self.device)
        self.get_prompt_embedding(
            self.prompt_box.value.split(";")
        )
        # from eye.sim.spherical_video import ClipReward
        # openclip_model, _, _ = open_clip.create_model_and_transforms(
        #     "ViT-B-16", pretrained="laion2b_s34b_b88k", device='cuda', precision='fp16'
        # )
        # openclip_model.eval()
        # preprocess = torchvision.transforms.Normalize(
        #     mean=[0.48145466, 0.4578275, 0.40821073],
        #     std=[0.26862954, 0.26130258, 0.27577711],
        # )
        # tokenizer = open_clip.get_tokenizer("ViT-B-16")
        # import glob
        # # stores a dictionary mapping the video to a list of clip rewards, one for each prompt
        # self.clip_reward = ClipReward(glob.glob("data/crops/estop/*"), glob.glob("data/crops/negatives/*"), openclip_model, preprocess, tokenizer, 'cuda')
        # self.target_embedding = self.clip_reward.target_clip_vec


        @self.change_prompt.on_click
        def _(_):
            self.target_embedding = self.get_prompt_embedding(
                self.prompt_box.value.split(";")
            )

        # Load initial model state using the helper method AFTER agent and viser are initialized
        self._load_model(self.current_model_name) # Loads state dict and compiles


    def _load_model(self, model_name):
        """Loads the state dict for the selected model and compiles."""
        if model_name not in self.ckpt_paths_dict:
            print(f"Error: Model name '{model_name}' not found in ckpt_paths_dict.")
            if hasattr(self, 'model_selector'):
                 self.model_selector.value = self.current_model_name # Revert dropdown
            return

        # Acquire the model loading lock to prevent concurrent access
        with self.model_loading_lock:
            ckpt_path = self.ckpt_paths_dict[model_name]
            print(f"Loading model '{model_name}' from {ckpt_path}...")
            try:
                # Clear dynamo cache before loading/compiling new model
                torch._dynamo.reset()

                # --- Try to load agent config from checkpoint ---
                config_path = ckpt_path.replace('.pt', '_config.json')
                agent_config = None
                
                # Try loading config from JSON file
                if os.path.exists(config_path):
                    try:
                        agent_config = VisualSearchAgentConfig.from_json(config_path)
                        print(f"Loaded agent config from {config_path}")
                    except Exception as e:
                        print(f"Warning: Could not load config from {config_path}: {e}")
                        agent_config = None
                
                # If no config file, try loading from checkpoint
                if agent_config is None:
                    agent_config = AgentConfigManager.load_config_from_checkpoint(ckpt_path)
                    if agent_config:
                        print(f"Loaded agent config from checkpoint {ckpt_path}")
                
                # If still no config, use defaults for backwards compatibility
                if agent_config is None:
                    print("No config found, using defaults for backwards compatibility")
                    agent_config = VisualSearchAgentConfig(
                        action_type='discrete',
                        decouple_direction=False,
                        device=self.device,
                        freeze_encoder=True,
                        crop_sizes=self.crop_sizes,
                        window_length=self.window_length,
                        window_size=self.window_size,
                        apply_score_mod=False,
                    )
                else:
                    # Use loaded config as-is
                    agent_config.device = self.device  # Ensure device is correct
                
                print(f"Initializing agent with config: window_length={agent_config.window_length}, action_type={agent_config.action_type}")
                self.agent = VisualSearchAgent(config=agent_config)
                self.agent.to(self.device) # Move to device before loading state dict
                # --- Agent Initialized ---

                # Load the state dict
                state_dict = torch.load(ckpt_path, map_location=self.device)
                self.agent.load_state_dict(state_dict["model_state_dict"], strict=True)
                self.agent.eval() # Set to evaluation mode
                self.current_model_name = model_name
                # Update window length if needed based on loaded model, though it's fixed here
                # self.window_length = WIN

                # Reset relevant states
                print("Resetting observation history and counters...")
                self.observation_history = {key: [] for key in self.observation_history} # Clear history
                self.timestep_counter = 0 # Reset warmup counter
                self.prev_azimuth_cmd = 0 # Reset EMA state
                self.prev_elev_cmd = 0
                self.start_time = None # Reset start time for FPS calculation

                print(f"Successfully switched to model: {model_name}")
            except FileNotFoundError:
                print(f"Error: Checkpoint file not found at {ckpt_path}")
                if hasattr(self, 'model_selector'):
                     self.model_selector.value = self.current_model_name # Revert dropdown
            except KeyError as e:
                 print(f"Error loading state dict from {ckpt_path}: Missing key {e}")
                 if hasattr(self, 'model_selector'):
                      self.model_selector.value = self.current_model_name # Revert dropdown
            except Exception as e:
                print(f"An unexpected error occurred while loading {ckpt_path}: {e}")
                import traceback
                traceback.print_exc() # Print full traceback for debugging
                if hasattr(self, 'model_selector'):
                     self.model_selector.value = self.current_model_name # Revert dropdown


    def _on_model_select(self, dropdown_handle):
        """Callback triggered when the model dropdown selection changes."""
        new_model_name = self.model_selector.value
        if new_model_name != self.current_model_name:
            print(f"Dropdown selection changed to: {new_model_name}")
            # Add a small delay or ensure Viser processes updates before heavy load?
            # time.sleep(0.1) # Optional small delay
            self._load_model(new_model_name)
        else:
            print(f"Model {new_model_name} already selected.")

    def on_frame(self, frame):
        buf = (
            torch.frombuffer(frame, dtype=torch.uint8)
            .to(self.device, torch.bfloat16)
            .reshape(1200, 1920, 3)
            .permute(2, 0, 1)
            / 255.0
        )
        with self.frame_lock:
            self.last_available_frame = buf

    def get_prompt_embedding(self, prompts):
        with torch.no_grad():
            tokens = self.tokenizer(prompts).to(self.device)
            embedding = self.model.encode_text(tokens)
            embedding /= embedding.norm(dim=-1, keepdim=True)
            embedding = embedding.mean(dim=0, keepdim=True)
            embedding /= embedding.norm(dim=-1, keepdim=True)
            return embedding.float()

    def pad_to_aspect_ratio(self, img, target_aspect, pad_value=0):
        """
        Pad an image to match a target aspect ratio (width/height)
        """
        h, w = img.shape[:2]
        current_aspect = w / h

        if current_aspect < target_aspect:
            # Need to add width padding
            new_w = int(h * target_aspect)
            pad_w = new_w - w
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            padded_img = np.pad(
                img,
                ((0, 0), (pad_left, pad_right), (0, 0)),
                mode="constant",
                constant_values=pad_value,
            )
        else:
            # Need to add height padding
            new_h = int(w / target_aspect)
            pad_h = new_h - h
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            padded_img = np.pad(
                img,
                ((pad_top, pad_bottom), (0, 0), (0, 0)),
                mode="constant",
                constant_values=pad_value,
            )

        return padded_img

    def loop_forever(self):
        self.ts = []
        time.sleep(0.1)

        self.start_time = time.time()
        while self.is_running:
            # Skip if model is being loaded
            if self.model_loading_lock.locked():
                time.sleep(0.1)  # Short sleep to avoid busy waiting
                continue

            with self.frame_lock:
                current_frame = self.last_available_frame
                self.last_available_frame = None
            if self.pause.value:
                # stop motors
                self.eye.azimuth_motor.set_speed(0)
                self.eye.elev_motor.set_speed(0)
            if current_frame is None:
                continue
            multicrop = create_foveated_batch(
                current_frame[None], crop_sizes=self.crop_sizes, window_size=self.window_size
            )
            if not self.pause.value:
                # Query the RL agent for the next action
                with torch.no_grad():
                    # Create foveated input
                    a, e = self.eye.azimuth_elev()
                    BASE_AZIMUTH_OFFSET = -np.deg2rad(119.5) + np.pi/2
                    so3 = SO3.from_z_radians(
                        torch.tensor(a + BASE_AZIMUTH_OFFSET, dtype=torch.float32)
                    ) @ SO3.from_x_radians(
                        torch.tensor(e - np.pi / 2, dtype=torch.float32)
                    )
                    # was -np.pi/2
                    observation_so3 = SO3.from_z_radians(torch.tensor(np.pi, dtype=torch.float32))@so3
                    # Current observation
                    current_obs = {
                        "multicrop": multicrop,  # Shape: (n_scales, 3, 224, 224)
                        "eye_direction": observation_so3.as_matrix()[:, 2].to(self.device),
                        "target_clip_vec": self.target_embedding,
                    }
                    # Update observation history
                    for key, value in current_obs.items():
                        self.observation_history[key].append(value)
                        # Keep only the most recent window_length observations
                        if len(self.observation_history[key]) > self.window_length:
                            self.observation_history[key].pop(0)
                    
                    # Create window observation for context agent
                    window_obs = {}
                    for key in current_obs:
                        if key == "multicrop":
                            # For multicrop, we need shape (T, B, n_scales, 3, H, W)
                            # Our history has entries with shape (n_scales, 3, H, W)
                            # Stack to (T, n_scales, 3, H, W) then add batch dim
                            stacked = torch.stack(self.observation_history[key], dim=0)  # (T, n_scales, 3, H, W)
                            window_obs[key] = stacked.unsqueeze(1)  # (T, 1, n_scales, 3, H, W)
                        elif key == "eye_direction":
                            # For eye_direction, stack to (T, 3) then add batch dim
                            stacked = torch.stack(self.observation_history[key], dim=0)  # (T, 3)
                            window_obs[key] = stacked.unsqueeze(1)  # (T, 1, 3)
                        elif key == 'target_clip_vec':  # target_clip_vec
                            # For target embedding, repeat across time dimension
                            # It's the same target for all time steps
                            window_obs[key] = self.observation_history[key][0].repeat(len(self.observation_history[key]), 1, 1)  # (T, 1, 512)
                    # next cache the foveal tokens
                    if len(self.observation_history['foveal_tokens']) > 0:
                        window_obs['foveal_tokens'] = torch.stack(self.observation_history['foveal_tokens'], dim=0) 
                    # Get action from agent using context window (always run for JIT compilation)
                    with amp.autocast('cuda', dtype=torch.bfloat16):
                        action, _, _, value = self.agent.get_action_and_value(
                            window_obs, deterministic=False
                        )
                        
                    # Store foveal_tokens if they were computed and returned in window_obs
                    if 'foveal_tokens' in window_obs:
                        # Store the most recent foveal tokens (last time step)
                        self.observation_history['foveal_tokens'].append(window_obs['foveal_tokens'][-1].clone())
                        # Keep only the most recent window_length tokens
                        if len(self.observation_history['foveal_tokens']) >= self.window_length:
                            self.observation_history['foveal_tokens'].pop(0)
                    
                    # Take the action from the last time step
                    action = action[-1].squeeze().float().cpu().numpy()

                    # Only set motor speeds if we're past the warmup period
                    if self.timestep_counter >= self.window_length:
                        # Apply EMA smoothing to motor commands
                        raw_azimuth_cmd = np.rad2deg(action[0])*15
                        raw_elev_cmd = np.rad2deg(action[1])*15
                        
                        # Calculate smoothed commands using EMA
                        smoothed_azimuth = self.ema_alpha * raw_azimuth_cmd + (1 - self.ema_alpha) * self.prev_azimuth_cmd
                        smoothed_elev = self.ema_alpha * raw_elev_cmd + (1 - self.ema_alpha) * self.prev_elev_cmd
                        
                        # Update previous commands for next iteration
                        self.prev_azimuth_cmd = smoothed_azimuth
                        self.prev_elev_cmd = smoothed_elev
                        
                        # Get current position
                        cam_elev = np.rad2deg(e)#np.mod(self.eye.elev_motor.get_angle(), 360) - Eyeball.zero_elev
                        
                        # Get current position
                        BASE_AZIMUTH_OFFSET = -np.deg2rad(119.5) + np.pi/2

                        # Some ugly offsets, should clean later...
                        cam_azim = np.mod(np.rad2deg(a + BASE_AZIMUTH_OFFSET + np.pi/2), 360)
                        # Check each boundary separately and clip commands accordingly
                        if cam_elev < self.eye_elev_bound[0]:
                            print(f"Safety limit triggered: elev={cam_elev:.1f}° < {self.eye_elev_bound[0]}°")
                            smoothed_elev = max(0, smoothed_elev)  # Only allow positive elevation commands
                        elif cam_elev > self.eye_elev_bound[1]:
                            print(f"Safety limit triggered: elev={cam_elev:.1f}° > {self.eye_elev_bound[1]}°")
                            smoothed_elev = min(0, smoothed_elev)  # Only allow negative elevation commands
                            
                        if cam_azim < self.eye_azim_bound[0]:
                            print(f"Safety limit triggered: azim={cam_azim:.1f}° < {self.eye_azim_bound[0]}°")
                            smoothed_azimuth = max(0, smoothed_azimuth)  # Only allow positive azimuth commands
                        elif cam_azim > self.eye_azim_bound[1]:
                            print(f"Safety limit triggered: azim={cam_azim:.1f}° > {self.eye_azim_bound[1]}°")
                            smoothed_azimuth = min(0, smoothed_azimuth)  # Only allow negative azimuth commands
                        
                        # Position is within bounds, send motor commands
                        self.eye.azimuth_motor.set_speed(smoothed_azimuth)
                        self.eye.elev_motor.set_speed(smoothed_elev)

                        # Check if current position is outside the bounds, considering wraparound
                        # For azimuth, normalize to -180 to 180 range for easier comparison
                        # norm_azim = (cam_azim + 180) % 360 - 180
                        
                        # if False:# (cam_elev < self.eye_elev_bound[0] or cam_elev > self.eye_elev_bound[1] or 
                        #     pass
                        #     # norm_azim < self.eye_azim_bound[0] or norm_azim > self.eye_azim_bound[1]):
                        #     # Position is outside bounds, activate pause
                        #     print(f"Safety limit triggered: elev={cam_elev:.1f}°, azim={norm_azim:.1f}°")
                        #     print(f"Bounds: elev=[{self.eye_elev_bound[0]},{self.eye_elev_bound[1]}], azim=[{self.eye_azim_bound[0]},{self.eye_azim_bound[1]}]")
                        #     self.pause.value = True
                        #     # Stop motors immediately
                        #     self.eye.azimuth_motor.set_speed(0)
                        #     self.eye.elev_motor.set_speed(0)
                        # else:
                        #     # Position is within bounds, send motor commands
                        #     self.eye.azimuth_motor.set_speed(smoothed_azimuth)
                        #     self.eye.elev_motor.set_speed(smoothed_elev)
                    else:
                        # During warmup, set speeds to 0
                        self.eye.azimuth_motor.set_speed(0)
                        self.eye.elev_motor.set_speed(0)
                        self.timestep_counter += 1
            # Display multicrop visualization
            multicrop = multicrop.squeeze(0)  # Remove batch dimension
            n_crops = len(self.crop_sizes)

            # Handle different numbers of crops
            if n_crops == 1:
                tiled_multicrop = multicrop
            elif n_crops == 2:
                tiled_multicrop = torch.concatenate(
                    [multicrop[1], multicrop[0]], axis=2
                )
            elif n_crops == 3:
                # Create a 2x2 grid with empty top-left quadrant
                top_row = torch.concatenate(
                    [torch.zeros_like(multicrop[2]), multicrop[2]], axis=2
                )
                bottom_row = torch.concatenate([multicrop[1], multicrop[0]], axis=2)
                tiled_multicrop = torch.concatenate([top_row, bottom_row], axis=1)
            else:
                multicrop = multicrop[:4]
                # 2x2 grid layout
                top_row = torch.concatenate([multicrop[3], multicrop[2]], axis=2)
                bottom_row = torch.concatenate([multicrop[1], multicrop[0]], axis=2)
                tiled_multicrop = torch.concatenate([top_row, bottom_row], axis=1)

            view = (255 * tiled_multicrop).byte().cpu().numpy().transpose(1, 2, 0)
            # Using a fixed aspect ratio since we don't have client camera info
            # view = self.pad_to_aspect_ratio(view, 16 / 9)
            self.viser_server.scene.set_background_image(view, format="jpeg")
            
            self.count += 1

    def stop(self):
        elapsed_time = time.time() - self.start_time
        self.is_running = False
        self.eye.teardown()
        print(f"Average FPS: {self.count / elapsed_time:.2f}")


if __name__ == "__main__":
    import signal
    import sys
    torch._dynamo.config.cache_size_limit = 128

    def signal_handler(signum, frame):
        print("\nCtrl+C received. Stopping the controller...")
        if 'controller' in globals() and controller.is_running: # Check if controller exists and is running
             controller.stop()
        sys.exit(0)

    # --- Define your models here ---
    CKPT_PATHS = {
        "TODO: checkpoint description": "checkpoint_path"
    }
    # Ensure at least one model is defined
    if not CKPT_PATHS:
        print("Error: No checkpoints defined in CKPT_PATHS dictionary.")
        sys.exit(1)

    # Check if paths exist (optional but recommended)
    existing_paths = {}
    for name, path in CKPT_PATHS.items():
        if os.path.exists(path):
            existing_paths[name] = path
        else:
            print(f"Warning: Checkpoint path for '{name}' does not exist: {path}. Skipping this model.")

    if not existing_paths:
         print("Error: None of the defined checkpoint paths exist.")
         sys.exit(1)

    controller = EyeController(existing_paths) # Pass the dictionary of existing paths

    signal.signal(signal.SIGINT, signal_handler)

    print("Starting EyeController. Press Ctrl+C to stop.")
    try:
        controller.loop_forever()
    except Exception as e:
        print(f"Error in main loop: {e}")

    finally:
        controller.stop()
