import threading
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "dependencies"))
from eye.eyeball import Eyeball
from eye.zmq_tools import Subscriber
import numpy as np
import time
import torch
from eye.foveal_encoders import create_foveated_batch
from eye.agents import EyeRobotAgent
from eye.agent_configs import EyeRobotAgentConfig, AgentConfigManager
from eye.transforms import SO3
import matplotlib.pyplot as plt
from eye.foveal_encoders import crop_sizes_from_levels
from viser import ViserServer
from viser.extras import ViserUrdf
import open_clip
import torchvision
import torch.amp as amp
from gello.robots.ur import URRobot
import os
from eye.zmq_tools import Publisher
from pathlib import Path
from gello.robots.robotiq_gripper import RobotiqGripper
import rtde_receive
from pynput import keyboard
from eye.eyeball import angle_direction
from eye.sim.demo_data import DemonstrationData
# JAX-based IK utilities
import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls
import numpy as onp
import pyroki as pk
# Set PyTorch Dynamo cache size to prevent memory issues
torch._dynamo.config.cache_size_limit = 128

def solve_ik(
    robot: pk.Robot,
    target_link_name: str,
    target_wxyz: onp.ndarray,
    target_position: onp.ndarray,
    q_init: onp.ndarray | None = None,
) -> onp.ndarray:
    """
    Solves the basic IK problem for a robot.

    Args:
        robot: PyRoKi Robot.
        target_link_name: String name of the link to be controlled.
        target_wxyz: onp.ndarray. Target orientation as quaternion wxyz.
        target_position: onp.ndarray. Target position xyz.
        q_init: Optional initial joint configuration for warm-start.

    Returns:
        cfg: onp.ndarray. Shape: (robot.joints.num_actuated_joints,).
    """
    assert target_position.shape == (3,) and target_wxyz.shape == (4,)
    target_link_index = robot.links.names.index(target_link_name)
    cfg = _solve_ik_jax(
        robot,
        jnp.array(target_link_index),
        jnp.array(target_wxyz),
        jnp.array(target_position),
        None if q_init is None else jnp.asarray(q_init),
    )
    assert cfg.shape == (robot.joints.num_actuated_joints,)
    return onp.array(cfg)


@jdc.jit
def _solve_ik_jax(
    robot: pk.Robot,
    target_link_index: jax.Array,
    target_wxyz: jax.Array,
    target_position: jax.Array,
    q_init: jax.Array | None,
) -> jax.Array:
    # If q_init provided, use it to initialize joint variable; else start from zeros
    joint_var = robot.joint_var_cls(0)  
    factors = [
        pk.costs.pose_cost_analytic_jac(
            robot,
            joint_var,
            jaxlie.SE3.from_rotation_and_translation(
                jaxlie.SO3(target_wxyz), target_position
            ),
            target_link_index,
            pos_weight=50.0,
            ori_weight=10.0,
        ),
        pk.costs.limit_cost(
            robot,
            joint_var,
            weight=100.0,
        ),
    ]
    sol = (
        jaxls.LeastSquaresProblem(factors, [joint_var])
        .analyze()
        .solve(
            initial_vals=jaxls.VarValues.make(
                [joint_var.with_value(q_init)]
            ),
            verbose=False,
            linear_solver="dense_cholesky",
            trust_region=jaxls.TrustRegionConfig(lambda_initial=1.0),
        )
    )
    return sol[joint_var]

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
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.robot_ip="172.22.22.2"
        self.r_inter = rtde_receive.RTDEReceiveInterface(self.robot_ip)
        self.use_gripper = True
        if self.use_gripper:
            self.gripper = RobotiqGripper()
            self.gripper.connect(hostname=self.robot_ip, port=63352)
        self.last_available_frame = None
        self.frame_lock = threading.Lock()
        self.is_running = True
        self.count = 0
        self.eye_elev_bound = (-30, 30)
        self.eye_azim_bound = (50, 310)
        # reset for eraser
        # self.reset_joints = torch.tensor([ 3.3279, -1.5542, -1.9253, -1.2189,  1.5642,  1.7711,  0.0], device=self.device).unsqueeze(0).unsqueeze(0)
        self.reset_joints = torch.tensor([ 3.3690, -1.5778, -1.6276, -1.4800,  1.5781,  1.8183,  0.0], device=self.device).unsqueeze(0).unsqueeze(0)
        #towel
        # self.reset_joints = torch.tensor([ 3.3690, -1.5778, -1.6276, -1.4800,  1.5781,  1.8183-np.pi/2,  0.0], device=self.device).unsqueeze(0).unsqueeze(0)
        #brush
        # self.reset_joints = torch.tensor([ 3.308755874633789, -1.5426496186158438, -1.9840549230575562, -1.1958702963641663, 1.5839810371398926,  0.0, 0.0], device=self.device).unsqueeze(0).unsqueeze(0)
        #self.reset_joints = torch.tensor([ 3.308755874633789, -1.5426496186158438, -1.9840549230575562, -1.1958702963641663, 1.5839810371398926,  0.0, 0.0], device=self.device).unsqueeze(0).unsqueeze(0)
        # Initialize timestep counter for warmup period
        self.timestep_counter = 0

        # Add a model loading lock to prevent concurrent access
        self.model_loading_lock = threading.Lock()

        # Initialize EMA smoothing for motor commands
        self.ema_alpha = 0.5 # Smoothing factor (0-1): higher = more weight to current action
        self.prev_azimuth_cmd = 0
        self.prev_elev_cmd = 0

        # Initialize the RL agent parameters
        self.agent = None # Agent will be initialized in _load_model
        self.window_length = 3 # Default, will be updated in _load_model
        self.act_chunk_size = 30 # Keep constant for now, or make GUI element later if needed


        # Initialize observation history buffer
        # Length is dynamic based on self.window_length set in _load_model
        self.observation_history = {
            "multicrop": [],
            "eye_direction": [],
            "target_clip_vec": [],
            "foveal_tokens": [],
            "proprio": [],
        }

        self.viser_server = ViserServer()
        # --- GUI Controls ---
        self.deterministic_actions = self.viser_server.gui.add_checkbox("Deterministic Actions", True)
        self.pause_eye = self.viser_server.gui.add_checkbox("Pause Eye", True)
        self.pause_robot = self.viser_server.gui.add_checkbox("Pause Robot", True)
        self.reset_joints_check = self.viser_server.gui.add_checkbox("Reset Joints", False)

        # Agent configuration will be loaded from saved configs
        # GUI elements removed - configs now loaded from checkpoint files
        self.value_function_text = self.viser_server.gui.add_text(label='Value Function',initial_value='N/A',disabled=True)

        # Model selection dropdown
        print(self.ckpt_paths_dict.keys())
        self.model_selector = self.viser_server.gui.add_dropdown(
            "Select Model",
            options=list(self.ckpt_paths_dict.keys()),
            initial_value=self.current_model_name
        )
        self.model_selector.on_update(self._on_model_select) # Register callback
        # --- End GUI Controls ---

        self.viser_server.scene.enable_default_lights()
        # Add URDF visualization
        urdf_path = os.path.join(os.path.dirname(__file__),'../urdf/ur5e_with_robotiq_gripper.urdf')
        self.gt_ur5 = ViserUrdf(
            self.viser_server,
            urdf_or_path=Path(urdf_path),
            root_node_name='/OG'
        )
        self.act_vis_subsample = 15
        # Adjust ACT_SIZE reference if it's dynamic later
        self.act_urdfs = [ViserUrdf(self.viser_server, urdf_or_path=Path(urdf_path), root_node_name=f'/OG_{i}',mesh_color_override=(0.8-0.7*i/(self.act_chunk_size//self.act_vis_subsample-1), 0.1+0.7*i/(self.act_chunk_size//self.act_vis_subsample-1), 0.1)) for i in range(self.act_chunk_size//self.act_vis_subsample)]
        # Add grid
        self.viser_server.scene.add_grid(
            "/grid",
            width=2,
            height=2,
            position=(
                0.0,
                0.0,
                # Get the minimum z value of the trimesh scene
                self.gt_ur5._urdf.scene.bounds[0, 2],
            ),
        )



        self.target_embedding = torch.zeros((1,512), device=self.device, dtype=torch.float32)
        self.frame_subscriber = Subscriber("ipc:///tmp/eye_frame")
        self.frame_subscriber.on_recv_bytes(self.on_frame)
        self.action_chunk_publisher = Publisher("ipc:///tmp/eye_action_chunk",conflate=False)

        # Lazy-initialized PyRoki robot for IK (filled when needed)
        self.pk_robot_obj = None
        self.ee_link_name = "tool0"

        # --- Keyboard Listener Setup ---
        self.keyboard_listener = keyboard.Listener(on_press=self._on_key_press)
        self.keyboard_listener.start()
        # ------------------------------

        # Load initial model state using the helper method AFTER agent and viser are initialized
        self._load_model(self.current_model_name) # Loads state dict and compiles


    def _load_model(self, model_name):
        """Loads the state dict for the selected model and recompiles."""
        if model_name not in self.ckpt_paths_dict:
            print(f"Error: Model name '{model_name}' not found in ckpt_paths_dict.")
            if hasattr(self, 'model_selector'):
                 self.model_selector.value = self.current_model_name
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
                        agent_config = EyeRobotAgentConfig.from_json(config_path)
                        print(f"Loaded agent config from {config_path}")
                    except Exception as e:
                        print(f"Warning: Could not load config from {config_path}: {e}")
                        agent_config = None
                
                # If no config file, try loading from checkpoint
                if agent_config is None:
                    agent_config = AgentConfigManager.load_config_from_checkpoint(ckpt_path)
                    if agent_config:
                        print(f"Loaded agent config from checkpoint {ckpt_path}")
                # If still no config, use defaults with GUI overrides for backwards compatibility
                if agent_config is None:
                    print("No config found, using defaults for backwards compatibility")
                    agent_config = EyeRobotAgentConfig(
                        device=self.device
                    )
                else:
                    # Use loaded config as-is
                    agent_config.device = self.device  # Ensure device is correct
                
                self.crop_sizes = crop_sizes_from_levels(agent_config.n_levels, agent_config.fovea_size, agent_config.sphere_size)
                self.window_size = agent_config.window_size
                print(f"Initializing agent with config: {agent_config}")

                self.agent = EyeRobotAgent(config=agent_config)
                self.agent.to(self.device) # Move to device before loading state dict
                self.agent.feature_extractor.compile()
                # --- Agent Initialized ---
                # Load the state dict
                state_dict = torch.load(ckpt_path, map_location=self.device)
                print("Loaded state dict")
                self.agent.load_state_dict(state_dict["model_state_dict"], strict=True)
                self.agent.eval() # Set to evaluation mode
                self.current_model_name = model_name
                self.window_length = agent_config.window_length # Update the class window_length attribute

                # Reset relevant states
                print("Resetting observation history and counters...")
                self.observation_history = {key: [] for key in self.observation_history} # Clear history
                # Ensure history buffers respect the new window length dynamically in loop_forever
                self.timestep_counter = 0 # Reset warmup counter
                self.prev_azimuth_cmd = 0 # Reset EMA state
                self.prev_elev_cmd = 0
                self.start_time = None # Reset start time for FPS calculation


                print(f"Successfully switched to model: {model_name}")
            except FileNotFoundError:
                print(f"Error: Checkpoint file not found at {ckpt_path}")
                # Optionally revert dropdown or handle error appropriately
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
            self._load_model(new_model_name)
        else:
            print(f"Model {new_model_name} already selected.")

    def on_frame(self, frame):
        buf = (
            torch.frombuffer(frame, dtype=torch.uint8)
            .to(self.device, torch.float32)
            .reshape(1200, 1920, 3)
            .permute(2, 0, 1)
            / 255.0
        )
        with self.frame_lock:
            self.last_available_frame = buf

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
        self.start_time = None
        act_pred = None
        while self.is_running:
            # Skip if model is being loaded
            if self.model_loading_lock.locked():
                time.sleep(0.1)  # Short sleep to avoid busy waiting
                continue
                
            with self.frame_lock:
                current_frame = self.last_available_frame
                self.last_available_frame = None
            if self.pause_eye.value:
                # stop motors
                self.eye.azimuth_motor.set_speed(0)
                self.eye.elev_motor.set_speed(0)
            if current_frame is None:
                continue
            multicrop = create_foveated_batch(
                current_frame[None], crop_sizes=self.crop_sizes, window_size=self.window_size
            )
            # Query the RL agent for the next action
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
            cur_joints = torch.tensor(self.r_inter.getActualQ()).to(self.device,dtype=torch.float32)[None,None]
            current_frame_time = time.perf_counter()
            if not self.use_gripper:
                # append a dummy of 0s for the gripper position
                cur_joints = torch.cat([cur_joints, torch.zeros((1,1,1), dtype=torch.float32, device=self.device)], dim=-1)
            else:
                pos = float(self.gripper.get_current_position())/255.0
                cur_joints = torch.cat([cur_joints, torch.tensor(pos, dtype=torch.float32, device=self.device)[None,None,None]], dim=-1)
            current_obs = {
                "multicrop": multicrop,  # Shape: (n_scales, 3, 224, 224)
                "eye_direction": observation_so3.as_matrix()[:, 2].to(self.device),
                "target_clip_vec": self.target_embedding,
                'proprio': cur_joints,
            }
            if self.reset_joints_check.value:
                reset_joint_dir = (self.reset_joints - cur_joints)
                steps = 30
                start_vec = cur_joints  # Shape (1, 1, 7)
                end_vec = cur_joints + reset_joint_dir * 1.0  # Shape (1, 1, 7)
                # Generate interpolation factors t, shape (steps, 1, 1) for broadcasting
                t = torch.linspace(0, 1, steps, device=cur_joints.device).reshape(steps, 1, 1)
                # Perform linear interpolation
                reset_traj = start_vec + t * (end_vec - start_vec) # Result shape (steps, 1, 7)
                reset_traj = reset_traj.permute(1,0,2).unsqueeze(0)
                self.action_chunk_publisher.send_action_chunk(reset_traj,current_frame_time)
                continue
            
            # Update observation history
            for key, value in current_obs.items():
                self.observation_history[key].append(value)
                # Keep only the most recent window_length observations
                while len(self.observation_history[key]) > self.window_length:
                    self.observation_history[key].pop(0)
                    
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
                elif key == 'proprio':
                    #stack the last WIN observations
                    window_obs[key] = torch.cat(self.observation_history[key], dim=0)  # (T, 1, 7)
            # next cache the foveal tokens
            if len(self.observation_history['foveal_tokens']) > 0:
                window_obs['foveal_tokens'] = torch.stack(self.observation_history['foveal_tokens'], dim=0) 
            # Get action from agent using context window (always run for JIT compilation)
            with torch.inference_mode(),amp.autocast('cuda', dtype=torch.bfloat16):
                action, _, _, value, act_pred = self.agent.get_action_and_value(
                    window_obs, deterministic=self.deterministic_actions.value, inference=True
                )
            self.value_function_text.value = f"{value[-1,:,:].item():.2f}"
            if self.agent.config.use_se3_actions:
                pass #TODO eventually need to convert to global se3 (not needed yet, network outputs global already)
            elif self.agent.config.relative_act:
                act_pred = DemonstrationData.denormalize_relative_data(act_pred.float(), method="mean_std") + window_obs['proprio'][-1].unsqueeze(1)
            else:
                act_pred = DemonstrationData.denormalize_predictions(act_pred.float())

            
            if self.agent.config.use_se3_actions:
                # Convert SE3 poses to joint angles with PyRoki IK and send action chunk
                #nidhya 9/3/25: in progress, probably not working yet
                import yourdfpy
                from pyroki import Robot
                # Lazy init robot model for IK
                if self.pk_robot_obj is None:
                    urdf_path = os.path.join(os.path.dirname(__file__), '../urdf/ur5e_with_robotiq_gripper.urdf')
                    urdf = yourdfpy.URDF.load(Path(urdf_path), load_meshes=False)
                    self.pk_robot_obj = Robot.from_urdf(urdf)
                    # Cache end-effector link name
                    self.ee_link_name = 'tool0'
                # act_pred last step shape: (1, 1, A, 10) where 10 = xyz(3) + R cols (6) + gripper(1)
                se3_chunk = act_pred[-1, 0]  # A, 10
                A = se3_chunk.shape[0]
                # Build rotation matrices from first two columns using Gram-Schmidt orthonormalization
                pos_xyz = se3_chunk[:, 0:3]
                col1 = se3_chunk[:, 3:6]
                col2 = se3_chunk[:, 6:9]
                
                # Normalize first column
                col1_n = torch.nn.functional.normalize(col1, dim=-1)
                # Remove projection of col2 on col1 and normalize
                proj = (col2 * col1_n).sum(dim=-1, keepdim=True) * col1_n
                col2_orth = col2 - proj
                col2_n = torch.nn.functional.normalize(col2_orth, dim=-1)
                # Derive third column as right-handed cross of the orthonormalized cols
                col3_n = torch.cross(col1_n, col2_n, dim=-1)
                R = torch.stack([col1_n, col2_n, col3_n], dim=-1).to(torch.float32)  # (A, 3, 3)

                # Convert to wxyz quaternion for IK target
                wxyz_list = []
                for i in range(A):
                    wxyz_list.append(SO3.from_matrix(R[i]).wxyz)
                wxyz = torch.stack(wxyz_list, dim=0)  # (A,4)

                # Current joints for warm start (arm only, exclude gripper)
                cur_q = cur_joints[0, 0, :7].detach().float().cpu().numpy()

                # Solve IK per waypoint
                q_list = []
                for i in range(A):
                    target_xyz = pos_xyz[i].detach().float().cpu().numpy()
                    target_wxyz = wxyz[i].detach().float().cpu().numpy()
                    ik_sol = solve_ik(
                        robot=self.pk_robot_obj,
                        target_link_name=self.ee_link_name,
                        target_position=target_xyz,
                        target_wxyz=target_wxyz,
                        q_init=cur_q,
                    )
                    # Ensure arm-only (6) vector for warm-start
                    q_list.append(ik_sol)
                    cur_q = ik_sol[:7]
                
                # Stack arm solutions (A, 6), then append gripper as 7th joint
                arm_chunk = np.stack([q[:6] for q in q_list], axis=0) # (A,6)
                grip = se3_chunk[:, 9].clamp(0, 1).cpu().numpy()[:, None]  # (A,1)
                act_joint_chunk = np.concatenate([arm_chunk, grip], axis=1)  # (A,7)
                
                # g = act_joint_chunk[:, 6]
                # g[g > 0.5] = torch.clamp(g[g > 0.5] + 0.1, 0, 1)
                # act_joint_chunk[:, 6] = g
                # Publish with leading dims (1,1,A,D) to match consumers
                act_pred = act_joint_chunk[None,None]
                # Adjust ACT_SIZE reference if it becomes dynamic
                if not self.pause_robot.value:
                    self.action_chunk_publisher.send_action_chunk(torch.from_numpy(act_pred), current_frame_time)
                
            else:
                # offset the gripper action a bit if its >50% closed
                act_pred[...,6][act_pred[...,6] > 0.5] = torch.clamp(act_pred[...,6][act_pred[...,6] > 0.5] + 0.1, 0, 1)
                if not self.pause_robot.value:
                    self.action_chunk_publisher.send_action_chunk(act_pred[-1:],current_frame_time)
                act_pred = act_pred.float().cpu().numpy()
            if self.start_time is None:
                self.start_time = time.time()
            # Store foveal_tokens if they were computed and returned in window_obs
            if 'foveal_tokens' in window_obs:
                # Store the most recent foveal tokens (last time step)
                self.observation_history['foveal_tokens'].append(window_obs['foveal_tokens'][-1].clone())
                # Keep only the most recent window_length tokens
                while len(self.observation_history['foveal_tokens']) >= self.window_length:
                    self.observation_history['foveal_tokens'].pop(0)
            
            # Take the action from the last time step
            action = action[-1].squeeze().float().cpu().numpy()

            # Only set motor speeds if we're past the warmup period
            # Check against self.window_length which reflects the current agent's setting
            if self.timestep_counter >= self.window_length:
                # Apply EMA smoothing to motor commands
                raw_azimuth_cmd = np.rad2deg(action[0])*15
                raw_elev_cmd = np.rad2deg(action[1])*15
                if self.pause_eye.value:
                    raw_azimuth_cmd = 0
                    raw_elev_cmd = 0
                
                # Calculate smoothed commands using EMA
                smoothed_azimuth_vel = self.ema_alpha * raw_azimuth_cmd + (1 - self.ema_alpha) * self.prev_azimuth_cmd
                smoothed_elev_vel = self.ema_alpha * raw_elev_cmd + (1 - self.ema_alpha) * self.prev_elev_cmd

                # Update previous commands for next iteration
                self.prev_azimuth_cmd = smoothed_azimuth_vel
                self.prev_elev_cmd = smoothed_elev_vel
                
                # Get current position
                cam_elev = np.rad2deg(e)
                # Some ugly offsets, should clean later...
                cam_azim = np.mod(np.rad2deg(a + BASE_AZIMUTH_OFFSET + np.pi/2), 360)
                # # Check each boundary separately and clip commands accordingly
                if cam_elev < self.eye_elev_bound[0]:
                    print(f"Safety limit triggered: elev={cam_elev:.1f}° < {self.eye_elev_bound[0]}°")
                    smoothed_elev_vel = max(0, smoothed_elev_vel)  # Only allow positive elevation commands
                elif cam_elev > self.eye_elev_bound[1]:
                    print(f"Safety limit triggered: elev={cam_elev:.1f}° > {self.eye_elev_bound[1]}°")
                    smoothed_elev_vel = min(0, smoothed_elev_vel)  # Only allow negative elevation commands
                    
                if cam_azim < self.eye_azim_bound[0]:
                    print(f"Safety limit triggered: azim={cam_azim:.1f}° < {self.eye_azim_bound[0]}°")
                    smoothed_azimuth_vel = max(0, smoothed_azimuth_vel)  # Only allow positive azimuth commands
                elif cam_azim > self.eye_azim_bound[1]:
                    print(f"Safety limit triggered: azim={cam_azim:.1f}° > {self.eye_azim_bound[1]}°")
                    smoothed_azimuth_vel = min(0, smoothed_azimuth_vel)  # Only allow negative azimuth commands
                self.eye.azimuth_motor.set_speed(smoothed_azimuth_vel)
                self.eye.elev_motor.set_speed(smoothed_elev_vel)

            else:
                # During warmup, set speeds to 0
                self.eye.azimuth_motor.set_speed(0)
                self.eye.elev_motor.set_speed(0)
                self.timestep_counter += 1 # Increment counter only when history is not full
            multicrop = multicrop.squeeze(0)  # Remove batch dimension
            # Handle different numbers of crops
            #update the act_urdfs if we have actions available
            
            if act_pred is not None:
                if self.agent.config.use_se3_actions:
                    for i in range(0,30,6):
                        xyz=se3_chunk[i,:3].cpu().numpy()
                        xcol_ycol = se3_chunk[i,3:9].reshape(2,3).permute(1,0) # 2 (3,1) col vectors
                        z = torch.cross(xcol_ycol[:,0:1],xcol_ycol[:,1:2], dim=0)
                        wxyz = SO3.from_matrix(torch.cat([xcol_ycol, z], dim=1)).wxyz.cpu().numpy()
                        self.viser_server.scene.add_frame(f"pred_{i}", position=xyz, wxyz=wxyz, origin_radius=.01, axes_length=.07, axes_radius=.005)
                    
                # Adjust ACT_SIZE reference if it becomes dynamic
                for i in range(min(len(self.act_urdfs), act_pred.shape[2]//self.act_vis_subsample)):
                    self.act_urdfs[i].update_cfg(act_pred[-1,0,i*self.act_vis_subsample,:].squeeze())
            self.gt_ur5.update_cfg(cur_joints.squeeze().float().cpu().numpy())
            if len(multicrop.shape)>3:
                view = multicrop[-1]
            else:
                view = multicrop
            big_view = (255*view).byte().permute(1, 2, 0).cpu().numpy()
            self.viser_server.scene.add_camera_frustum("eye",fov=120,scale=.15,aspect=big_view.shape[1]/big_view.shape[0], image = big_view,
                                                    wxyz=so3.wxyz.float(),position=np.array([-.1,0,-.1]))

            self.count += 1

    def stop(self):
        elapsed_time = time.time() - self.start_time
        self.is_running = False
        self.eye.teardown()
        print(f"Average FPS: {self.count / elapsed_time:.2f}")
        # --- Stop Keyboard Listener ---
        if hasattr(self, 'keyboard_listener') and self.keyboard_listener.is_alive():
            self.keyboard_listener.stop()
            try:
                self.keyboard_listener.join() # Wait for listener thread to finish
            except RuntimeError:
                 # Ignore if the thread hasn't started yet or already stopped
                 pass
        # ------------------------------

    def _on_key_press(self, key):
        """Callback function for keyboard listener."""
        try:
            if key.char == 'o': # Toggle eye pause on 'o' key press
                 self.pause_eye.value = not self.pause_eye.value
        except AttributeError:
            # Ignore special keys
            pass

if __name__ == "__main__":
    import signal
    import sys

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

    controller = EyeController(existing_paths) # Pass only existing paths

    signal.signal(signal.SIGINT, signal_handler)

    print("Starting EyeController. Press Ctrl+C to stop.")
    try:
        controller.loop_forever()
    except Exception as e:
        print(f"An error occurred in the main loop: {e}") # Print error before stopping
        controller.stop()
        raise e

