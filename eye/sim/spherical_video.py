import torch
import numpy as np
import torch.nn.functional as F
from eye.camera import radial_and_tangential_undistort
import math
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from eye.transforms import SO3
from typing import List, Optional
from torchcodec.decoders import VideoDecoder


def one_time_cache(func):
    """Decorator that caches the first result of a function and returns it for all subsequent calls."""
    cache = {}
    def wrapper(*args, **kwargs):
        if not cache:
            cache['result'] = func(*args, **kwargs)
        return cache['result']
    return wrapper


@one_time_cache
def _get_local_cam_rays(
    K: torch.Tensor,
    dist_coeffs: torch.Tensor,
    rend_H: int,
    rend_W: int,
    is_fisheye: bool,
):
    # Create pixel coordinate grid
    device = K.device
    x = torch.linspace(0, rend_W - 1, rend_W, device=device)
    y = torch.linspace(0, rend_H - 1, rend_H, device=device)
    X, Y = torch.meshgrid(x, y, indexing="xy")
    pixels = torch.stack([X.flatten(), Y.flatten()], dim=1)

    # Ensure inputs are on the correct device and type
    pixels = pixels.float()

    # Get focal lengths and principal point from K matrix
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    # Normalize pixel coordinates
    normalized_pixels = torch.empty_like(pixels)
    normalized_pixels[..., 0] = (pixels[..., 0] - cx) / fx
    normalized_pixels[..., 1] = (pixels[..., 1] - cy) / fy

    # Undistort the normalized coordinates
    undistorted_points = radial_and_tangential_undistort(normalized_pixels, dist_coeffs)

    # Fisheye math
    if is_fisheye:
        # coord_stack is (N,2) containing normalized pixel coordinates
        theta = torch.sqrt(torch.sum(undistorted_points**2, dim=-1))  # (N,)
        theta = torch.clip(theta, 0.0, math.pi)

        sin_theta = torch.sin(theta)  # (N,)
        # Numerically stable computation of sin(theta)/theta
        # Use Taylor series: sin(x)/x ≈ 1 - x²/6 + x⁴/120 - ... for small x
        # For theta near 0, use the approximation, otherwise use the exact formula
        eps = 1e-8
        small_theta_mask = theta < eps
        
        # For small theta, use Taylor series approximation: 1 - theta²/6
        factor_small = 1.0 - (theta**2) / 6.0
        
        # For regular theta, use exact formula
        factor_regular = sin_theta / theta
        
        # Combine using where to avoid division by zero
        factor = torch.where(small_theta_mask, factor_small, factor_regular)

        # Build the ray directions (N,3)
        local_rays = torch.zeros(
            (undistorted_points.shape[0], 3),
            dtype=undistorted_points.dtype,
            device=undistorted_points.device,
        )
        local_rays[..., 0] = undistorted_points[..., 0] * factor  # x component
        local_rays[..., 1] = undistorted_points[..., 1] * factor  # y component
        local_rays[..., 2] = torch.cos(theta)  # z component
        local_rays /= local_rays.norm(dim=-1, keepdim=True)  # normalize
    else:
        # Convert undistorted points directly to unit vectors
        x = undistorted_points[..., 0]
        y = undistorted_points[..., 1]
        z = torch.ones_like(x)
        norm = torch.sqrt(x * x + y * y + 1)
        local_rays = torch.stack([x / norm, y / norm, z / norm], dim=-1)
    return local_rays


def _sample_from_equiv(
    equi_image: torch.Tensor, R: torch.Tensor, camera_local_rays: torch.Tensor
) -> torch.Tensor:
    """
    Optimized pinhole camera simulation using PyTorch with unified batched/single processing.

    Args:
        equi_image (torch.Tensor): Equirectangular image tensor with shape (H, W, 3)
        R (torch.Tensor): Rotation matrix with shape (3, 3) or (B, 3, 3) for batched
        camera_local_rays (torch.Tensor): Local camera rays tensor with shape (N, 3)

    Returns:
        pinhole_image (torch.Tensor): Pinhole image tensor with shape (C, N) or (B, C, N) for batched
    """
    # Normalize input to batched format for unified processing
    if len(R.shape) == 2:
        R = R.unsqueeze(0)  # (1, 3, 3)
        squeeze_output = True
    else:
        squeeze_output = False
    
    batch_size = R.shape[0]
    
    # Optimized batched matrix multiplication
    # Expand rays once and reuse: (B, 3, N)
    rays_expanded = camera_local_rays.T.unsqueeze(0).expand(batch_size, -1, -1)
    global_rays = torch.bmm(R, rays_expanded).transpose(-2, -1)  # (B, N, 3)
    
    # Extract coordinates - avoid redundant indexing
    x, y, z = global_rays.unbind(dim=-1)  # Each: (B, N)
    
    # Optimized coordinate calculations
    # Use faster hypot for radius calculation
    xy_radius = torch.hypot(x, y)  # More efficient than sqrt(x*x + y*y)
    
    # Calculate equirectangular coordinates
    u = torch.atan2(-y, x) * (1.0 / np.pi)  # Avoid division, use multiplication
    v = torch.atan2(xy_radius, z) * (1.0 / np.pi)
    
    # Prepare grid more efficiently - avoid intermediate view operations
    grid_coords = torch.stack([u, 2 * v - 1], dim=-1)  # (B, N, 2)
    grid = grid_coords.unsqueeze(1)  # (B, 1, N, 2)
    
    # Optimized image expansion - avoid memory copying when possible
    if batch_size == 1:
        equi_input = equi_image.unsqueeze(0)  # (1, C, H, W)
    else:
        # Use repeat instead of expand to avoid memory aliasing issues
        equi_input = equi_image.unsqueeze(0).repeat(batch_size, 1, 1, 1)  # (B, C, H, W)
    
    # Sample from equirectangular image
    pinhole_image = torch.nn.functional.grid_sample(
        equi_input, grid.to(equi_input.dtype), mode="bilinear", padding_mode="zeros", align_corners=True
    )
    
    # Remove spatial dimensions and optionally squeeze batch dimension
    result = pinhole_image.squeeze(2)  # (B, C, N)
    
    if squeeze_output:
        result = result.squeeze(0)  # (C, N)
    
    return result


@one_time_cache
def _multicrop_rays(
    rays: torch.Tensor, crop_sizes: List[int], win_size: int, h: int, w: int
) -> torch.Tensor:
    """
    Takes in a Nx3 set of rays and generates a pyramid of crops of size win_size.
    Rays are generated from meshgrid with indexing='xy' and should be reshaped accordingly.

    Args:
        rays: Input rays tensor of shape (N, 3) from meshgrid with indexing='xy'
        crop_sizes: List of crop sizes
        win_size: Window size for each crop
        h: Height of the original ray grid
        w: Width of the original ray grid

    Returns:
        Tensor of shape (-1, 3) that can be reshaped to (n_levels, 3, win_size, win_size)
        using: output.reshape(n_levels, 3, win_size, win_size)
    """
    # Reshape rays to 2D grid - since rays were created with meshgrid(x, y, indexing='xy'),
    # we need to reshape to (h, w, 3) directly
    rays_2d = rays.reshape(h, w, 3)

    # Convert to (1, 3, H, W) format for padding and interpolation
    rays_2d = rays_2d.permute(2, 0, 1).unsqueeze(0)

    # Calculate crop sizes for each level
    max_crop_size = max(crop_sizes)

    # Calculate padding
    pad_h = max(0, max_crop_size - h)
    pad_w = max(0, max_crop_size - w)
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    # Pad the rays tensor (already in NCHW format)
    rays_padded = F.pad(
        rays_2d, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0
    )

    cropped_rays = []

    for crop_size in crop_sizes:
        # Calculate start indices for cropping
        start_h = (rays_padded.shape[2] - crop_size) // 2
        start_w = (rays_padded.shape[3] - crop_size) // 2

        # Crop the central region (maintains NCHW format)
        cropped = rays_padded[
            :, :, start_h : start_h + crop_size, start_w : start_w + crop_size
        ]

        # Resize to win_size x win_size
        resized = F.interpolate(
            cropped, size=(win_size, win_size), mode="bilinear", align_corners=False
        )

        # Reshape to (3, win_size, win_size) and normalize
        resized = resized.squeeze(0)  # Remove batch dimension
        # Normalize rays to unit vectors along channel dimension
        resized = resized / torch.norm(resized, dim=0, keepdim=True)

        # Reshape to (-1, 3) for this level
        resized = resized.permute(1, 2, 0).reshape(-1, 3)
        cropped_rays.append(resized)

    # Concatenate all levels into a single (-1, 3) tensor
    return torch.cat(cropped_rays, dim=0)


class SphericalVideo:
    """
    Class which controls a video stream and renders frames from a given view using torchcodec
    with threaded frame prefetching
    """
    def __init__(
        self,
        video_path: Path,
        K: torch.Tensor,
        dist_coeffs: torch.Tensor,
        rend_H: int,
        rend_W: int,
        is_fisheye: bool,
        device: torch.device,
        crop_sizes: List[int],
        window_size: int = 224,
        decoder_device: str = 'cuda',
        downsample_factor: float = 1.0,
        fps_downsamp: Optional[int] = 1
    ):
        assert video_path.exists(), f"Video file {video_path} does not exist"
        self.video_path = video_path
        self.K = torch.tensor(K).to(device).float()
        self.dist_coeffs = dist_coeffs.to(device).float()
        self.rend_H = rend_H
        self.rend_W = rend_W
        self.crop_sizes = crop_sizes
        self.crop_size = window_size
        self.is_fisheye = is_fisheye
        self.device = device
        self.downsample_factor = downsample_factor
        self.fps_downsamp = fps_downsamp
        self._local_rays_cam = _get_local_cam_rays(
            self.K, self.dist_coeffs, self.rend_H, self.rend_W, self.is_fisheye
        )
        # Get the rays associated with the multi-crop
        self._local_rays_multicrop = _multicrop_rays(
            self._local_rays_cam,
            self.crop_sizes,
            self.crop_size,
            self.rend_H,
            self.rend_W,
        )
        self._local_rays_cam = self._local_rays_cam
        if 'cuda' in decoder_device:
            device_index = int(decoder_device.split(':')[1]) if ':' in decoder_device else 0
            self.decoder = VideoDecoder(str(self.video_path), device = f'cuda:{device_index}')
        else:
            self.decoder = VideoDecoder(self.video_path, device = 'cpu')
        self.total_frames = len(self.decoder)
        self.current_frame_idx = 0
        self._frame = None
        # Pipeline state for async decoding
        self._pipeline_executor = ThreadPoolExecutor(max_workers=1)
        self._next_frame_future = None

    @property
    def fps(self):
        return self.decoder.metadata.average_fps
    
    def get_time_of_frame(self, frame_idx: int) -> float:
        return frame_idx / self.fps

    def render_image(self, rot: SO3) -> torch.Tensor:
        """
        Render from the current time-step
        """
        return torch.nan_to_num(_sample_from_equiv(
            self._frame, rot.as_matrix().squeeze(0), self._local_rays_cam
        ).reshape(3, self.rend_H, self.rend_W), nan=0.0)

    @torch.inference_mode()
    def render_multicrop(self, rot: SO3, pad_value: float = 0.0) -> torch.Tensor:
        """
        Render the current timestep in a foveated way
        
        Args:
            rot: SO3 rotation(s) - can be single rotation or batched (B,)
            pad_value: Value to replace NaN pixels with
            
        Returns:
            Rendered crops - shape (n_crops, 3, crop_size, crop_size) for single rotation
                           or (B, n_crops, 3, crop_size, crop_size) for batched rotations
        """
        rot_matrix = rot.as_matrix()
        # Single rotation: squeeze and process as before
        renders = (
            _sample_from_equiv(
                self._frame, rot_matrix, self._local_rays_multicrop
            )
            .reshape(3, len(self.crop_sizes), self.crop_size, self.crop_size)
            .permute(1, 0, 2, 3)
        )
        
        # Pad the nan values with the pad_value
        renders = torch.nan_to_num(renders, nan=pad_value)
        return renders
    
    def nothread_set_frame(self, frame_idx: int):
        """
        Don't use this function unless you know what you're doing (this means not you)
        """
        frame = self.decoder[frame_idx].to(dtype=torch.float16, device=self.device).div_(255.0)
        self._frame = frame

    def _decode_frame(self, frame_idx: int) -> torch.Tensor:
        """Decode a single frame"""
        if frame_idx >= self.total_frames:
            return None
        frame = self.decoder[frame_idx].to(dtype=torch.float16, device=self.device).div_(255.0)
        
        if self.downsample_factor != 1.0:
            frame = F.interpolate(
                frame.unsqueeze(0),
                scale_factor=self.downsample_factor,
                mode='bilinear',
                align_corners=False
            )[0]
        return frame

    def advance(self) -> bool:
        """
        Pipelined version that decodes the next frame in a future while processing 
        the current frame. This overlaps video decoding with other computation.
        Returns False if the video stream is at the end
        """
        # Get the current frame from the pipeline
        if self._next_frame_future is not None:
            # Wait for the next frame to be ready and set it as current
            self._frame = self._next_frame_future.result()
        else:
            # First call - decode current frame synchronously
            self._frame = self._decode_frame(self.current_frame_idx)
        
        # Update frame index
        self.current_frame_idx += self.fps_downsamp
        
        # Check if we've reached the end
        if self.current_frame_idx >= self.total_frames:
            self._next_frame_future = None
            return self._frame is not None
        
        # Start decoding the next frame asynchronously
        self._next_frame_future = self._pipeline_executor.submit(
            self._decode_frame, self.current_frame_idx
        )
        
        return self._frame is not None

    def reset(self, random_time: bool = False, frame_idx: Optional[int] = None):
        """
        Reset video to beginning or random position
        """
        if random_time:
            self.current_frame_idx = torch.randint(0, self.total_frames - 1, (1,)).item()
        else:
            if frame_idx is not None:
                self.current_frame_idx = frame_idx
            else:
                self.current_frame_idx = 0

        # Reset pipeline state
        self._next_frame_future = None
        
        # Decode initial frame
        self._frame = self._decode_frame(self.current_frame_idx)

    def close(self):
        """
        Clean up resources. 
        """
        # Cancel any pending future and shutdown executor
        if self._next_frame_future is not None:
            self._next_frame_future.cancel()
            self._next_frame_future = None
        self._pipeline_executor.shutdown(wait=False)