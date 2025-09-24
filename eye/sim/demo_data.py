import torch
import numpy as np
from pathlib import Path
from typing import List, Optional
import h5py
import jax
import jax.numpy as jnp
import jaxlie
import numpy as onp

from eye.camera import get_default_video_config
from eye.foveal_encoders import crop_sizes_from_levels
from .spherical_video import SphericalVideo


class DemonstrationData:
    """
    Shallow class which holds the video as well as the robot data
    """
    def __init__(self, 
                 video_filename: str, 
                 h5_filename: Optional[str],
                 video_text_prompts: Optional[List[str]] = None, 
                 negative_prompts: Optional[List[str]] = None, 
                 w: int = 1920,
                 h: int = 1200,
                 local_rank: int = -1,
                 n_scales: int = 4,
                 data_dtype = torch.float,
                 fps_downsamp: int = 1,
                 load_spherical = True,
                 load_exo = False,
                 load_wrist = False,
                 fovea_res: int = 224,
                 window_size: int = 224,
                 sphere_size: int = 1920,
                 decoder_device: str = 'cuda'
                 ):
        self.video_filename = video_filename
        self.video_text_prompts = video_text_prompts
        self.negative_prompts = negative_prompts
        self.h5_filename = h5_filename
        self.w = w
        self.h = h
        self.local_rank = local_rank
        self.n_scales = n_scales
        self.K, self.dist_coeffs, self.is_fisheye = get_default_video_config(w, h)
        self.device = f"cuda:{self.local_rank}" if self.local_rank != -1 else "cuda"
        if load_spherical:
            self.video = SphericalVideo(
                Path(self.video_filename),
                self.K,
                self.dist_coeffs,
                self.h, 
                self.w,
                self.is_fisheye,
                self.device,
                crop_sizes=crop_sizes_from_levels(self.n_scales, fovea_res, sphere_size),
                decoder_device=decoder_device,
                fps_downsamp=fps_downsamp,
                window_size = window_size
            )
        self.robot_data = {}
        with h5py.File(self.h5_filename, "r") as f:
            for key in f.keys():
                self.robot_data[key] = torch.from_numpy(f[key][:]).to(self.device, dtype=data_dtype)
        
        # normalize the gripper data from 0 to 1
        grip_key = "interpolated_gripper_data" if "interpolated_gripper_data" in self.robot_data else "gripper_data"
        self.robot_data[grip_key] = self.robot_data[grip_key] / 255.0
        # concatenate the joint data and gripper data
        self.robot_data["joint_and_gripper_data"] = torch.cat([self.robot_data["joint_data"], self.robot_data[grip_key].unsqueeze(-1)], dim=1)
        # Load in wrist/exo videos if available
        assert self.video.total_frames == self.robot_data["joint_and_gripper_data"].shape[0], f"Video total frames {self.video.total_frames} does not match robot data {self.robot_data['joint_and_gripper_data'].shape[0]}"
        if load_exo:
            from torchcodec.decoders import VideoDecoder
            # exo name is the same but /zed.mp4 
            exo_file = Path(self.video_filename).parent / "zed_trimmed.mp4"
            self.exo_video = VideoDecoder(exo_file)

        if load_wrist:
            from torchcodec.decoders import VideoDecoder
            wrist_file = Path(self.video_filename).parent / "wrist_trimmed.mp4"
            self.wrist_video = VideoDecoder(wrist_file)
        

    @staticmethod
    def normalize_relative_data(relative_chunks, method="min_max"):
        is_torch = isinstance(relative_chunks, torch.Tensor)
        
        if method == "min_max":
            if is_torch:
                device = relative_chunks.device
                dtype = relative_chunks.dtype
                delta_min = torch.tensor([[-1.11377112, -0.65093874, -0.58465486, -0.88135759, -0.58820788, -1.08334295, -1.0]], device=device, dtype=dtype)
                delta_max = torch.tensor([[1.35768321, 0.72514558, 0.86634573, 0.76663546, 0.54602947, 1.25068096, 1.0]], device=device, dtype=dtype)
            else:
                import numpy as np
                dtype = relative_chunks.dtype
                delta_min = np.array([[-1.11377112, -0.65093874, -0.58465486, -0.88135759, -0.58820788, -1.08334295, -1.0]], dtype=dtype)
                delta_max = np.array([[1.35768321, 0.72514558, 0.86634573, 0.76663546, 0.54602947, 1.25068096, 1.0]], dtype=dtype)
            delta_min_expanded = delta_min.expand_as(relative_chunks)
            delta_max_expanded = delta_max.expand_as(relative_chunks)
            return 2.0 * ((relative_chunks - delta_min_expanded) / (delta_max_expanded - delta_min_expanded)) - 1.0
        
        elif method == "mean_std":
            if is_torch:
                device = relative_chunks.device
                dtype = relative_chunks.dtype
                delta_mean = torch.tensor([[0.00923639, -0.01492183, 0.016087, 0.00066093, -0.00083917, -0.0017091, 0.0]], device=device, dtype=dtype)
                delta_std = torch.tensor([[0.16677107, 0.12087255, 0.0956811, 0.12022196, 0.03015316, 0.09090457, 1.0]], device=device, dtype=dtype)
            else:
                import numpy as np
                dtype = relative_chunks.dtype
                delta_mean = np.array([[0.00923639, -0.01492183, 0.016087, 0.00066093, -0.00083917, -0.0017091, 0.0]], dtype=dtype)
                delta_std = np.array([[0.16677107, 0.12087255, 0.0956811, 0.12022196, 0.03015316, 0.09090457, 1.0]], dtype=dtype)
            delta_mean_expanded = delta_mean.expand_as(relative_chunks)
            delta_std_expanded = delta_std.expand_as(relative_chunks)
            return (relative_chunks - delta_mean_expanded) / delta_std_expanded
        
        else:
            raise ValueError(f"Unknown normalization method: {method}. Use 'min_max' or 'mean_std'.")

    @staticmethod
    def denormalize_relative_data(relative_chunks, method="min_max"):
        is_torch = isinstance(relative_chunks, torch.Tensor)
        
        if method == "min_max":
            if is_torch:
                device = relative_chunks.device
                dtype = relative_chunks.dtype
                delta_min = torch.tensor([[-1.11377112, -0.65093874, -0.58465486, -0.88135759, -0.58820788, -1.08334295, -1.0]], device=device, dtype=dtype)
                delta_max = torch.tensor([[1.35768321, 0.72514558, 0.86634573, 0.76663546, 0.54602947, 1.25068096, 1.0]], device=device, dtype=dtype)
            else:
                import numpy as np
                dtype = relative_chunks.dtype
                delta_min = np.array([[-1.11377112, -0.65093874, -0.58465486, -0.88135759, -0.58820788, -1.08334295, -1.0]], dtype=dtype)
                delta_max = np.array([[1.35768321, 0.72514558, 0.86634573, 0.76663546, 0.54602947, 1.25068096, 1.0]], dtype=dtype)
            delta_min_expanded = delta_min.expand_as(relative_chunks)
            delta_max_expanded = delta_max.expand_as(relative_chunks)
            return ((relative_chunks + 1.0) / 2.0) * (delta_max_expanded - delta_min_expanded) + delta_min_expanded
        
        elif method == "mean_std":
            if is_torch:
                device = relative_chunks.device
                dtype = relative_chunks.dtype
                delta_mean = torch.tensor([[0.00923639, -0.01492183, 0.016087, 0.00066093, -0.00083917, -0.0017091, 0.0]], device=device, dtype=dtype)
                delta_std = torch.tensor([[0.16677107, 0.12087255, 0.0956811, 0.12022196, 0.03015316, 0.09090457, 1.0]], device=device, dtype=dtype)
            else:
                import numpy as np
                dtype = relative_chunks.dtype
                delta_mean = np.array([[0.00923639, -0.01492183, 0.016087, 0.00066093, -0.00083917, -0.0017091, 0.0]], dtype=dtype)
                delta_std = np.array([[0.16677107, 0.12087255, 0.0956811, 0.12022196, 0.03015316, 0.09090457, 1.0]], dtype=dtype)
            delta_mean_expanded = delta_mean.expand_as(relative_chunks)
            delta_std_expanded = delta_std.expand_as(relative_chunks)
            return relative_chunks * delta_std_expanded + delta_mean_expanded
        
        else:
            raise ValueError(f"Unknown normalization method: {method}. Use 'min_max' or 'mean_std'.")

    @staticmethod
    def normalize_joint_data(joint_data):
        is_torch = isinstance(joint_data, torch.Tensor)
        
        if is_torch:
            device = joint_data.device
            dtype = joint_data.dtype
            joint_min = torch.tensor([1.66405463, -3.01445669, -2.53272627, -1.86111035, 0.99094322, -1.64613761, 0.0], device=device, dtype=dtype)
            joint_max = torch.tensor([5.61606252, -1.1168563, -0.40348044, 0.11331638, 2.14451, 2.16595739, 1.0], device=device, dtype=dtype)
        else:
            import numpy as np
            joint_min = np.array([1.66405463, -3.01445669, -2.53272627, -1.86111035, 0.99094322, -1.64613761, 0.0], dtype=joint_data.dtype)
            joint_max = np.array([5.61606252, -1.1168563, -0.40348044, 0.11331638, 2.14451, 2.16595739, 1.0], dtype=joint_data.dtype)
            
        return 2.0 * ((joint_data - joint_min) / (joint_max - joint_min)) - 1.0

    @staticmethod
    def denormalize_predictions(predictions):
        is_torch = isinstance(predictions, torch.Tensor)
        
        if is_torch:
            device = predictions.device
            dtype = predictions.dtype
            joint_min = torch.tensor([1.66405463, -3.01445669, -2.53272627, -1.86111035, 0.99094322, -1.64613761, 0.0], device=device, dtype=dtype)
            joint_max = torch.tensor([5.61606252, -1.1168563, -0.40348044, 0.11331638, 2.14451, 2.16595739, 1.0], device=device, dtype=dtype)
        else:
            import numpy as np
            joint_min = np.array([1.66405463, -3.01445669, -2.53272627, -1.86111035, 0.99094322, -1.64613761, 0.0], dtype=predictions.dtype)
            joint_max = np.array([5.61606252, -1.1168563, -0.40348044, 0.11331638, 2.14451, 2.16595739, 1.0], dtype=predictions.dtype)
            
        return ((predictions + 1.0) / 2.0) * (joint_max - joint_min) + joint_min


def line_distance(pred, gt):
    """
    Calculates a one-sided distance from pred to gt (partial Chamfer distance).
    For each point in pred, finds the minimum Euclidean distance to any point in gt,
    and sums these minimum distances.

    Args:
        pred: Tensor or array-like of shape (N, D) representing the predicted curve points.
        gt: Tensor or array-like of shape (M, D) representing the ground truth curve points.

    Returns:
        Tensor: The total summed minimum distance.
    """
    # Ensure inputs are torch tensors
    if not isinstance(pred, torch.Tensor):
        # Make sure to handle potential device placement if using GPU
        pred = torch.tensor(pred, dtype=torch.float32)
    if not isinstance(gt, torch.Tensor):
        gt = torch.tensor(gt, dtype=torch.float32)

    # Calculate pairwise Euclidean distances using torch.cdist
    # dist_matrix[i, j] is the distance between pred[i] and gt[j]
    # Shape: (N, M)
    dist_matrix = torch.cdist(pred, gt, p=2)

    # For each point in pred (dim=0), find the minimum distance to any point in gt (dim=1)
    # min_dists will have shape (N,)
    min_dists, _ = torch.min(dist_matrix, dim=1)

    # Average up these minimum distances
    total_distance = torch.mean(min_dists)

    return total_distance


@jax.jit
def compute_fk_batch(joint_data, robot_obj, ee_link_index):
    batched_fk = jax.vmap(robot_obj.forward_kinematics)
    # Input shape: (num_frames, num_joints)
    # Output shape: (num_frames, num_links, 4, 4) 
    all_link_poses_wxyz_xyz = batched_fk(joint_data)
    # Convert matrices to jaxlie SE3 objects
    all_link_poses_se3 = jaxlie.SE3(all_link_poses_wxyz_xyz)
    
    # Assume the last link is the end-effector
    ee_poses_wxyz_xyz = all_link_poses_se3.wxyz_xyz[:, ee_link_index] # Shape: (num_frames,) of SE3 objects
    # Convert SE3 poses to xyz_wxyz format
    translations = ee_poses_wxyz_xyz[:, 4:] # Shape: (num_frames, 3)
    rotations_wxyz = ee_poses_wxyz_xyz[:, :4] # Shape: (num_frames, 4)
    # Concatenate translation (xyz) and rotation (wxyz)
    ee_poses_wxyz_xyz = jnp.concatenate([rotations_wxyz, translations], axis=-1) # Shape: (num_frames, 7)
    return ee_poses_wxyz_xyz