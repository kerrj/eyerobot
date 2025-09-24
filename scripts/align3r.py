"""UR5 Robot URDF visualizer with H5 motion playback

This script loads the UR5 robot model from robot_descriptions and
visualizes the joint trajectories from h5 files created by data_collect.py.
"""

from __future__ import annotations

import time
import viser
from eye.ur5_visualizer import UR5Visualizer
import tyro
from viser import ViserServer
import torch
import numpy as np
import viser.transforms as vtf
import time
import cv2
from typing import Tuple, Optional
from eye.camera import radial_and_tangential_undistort, get_default_video_config
import math
from eye.sim.spherical_video import SphericalVideo
from eye.sim.rewards import ClipReward
import argparse
from pathlib import Path
from eye.transforms import SO3
import os
import plotly.graph_objects as go
from torchcodec.decoders import VideoDecoder
import sys
def pad_to_aspect_ratio(img, target_aspect, pad_value=0):
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

class WristVideo:
    """Class to handle loading and decoding of wrist camera video"""
    def __init__(
        self,
        video_path: Path,
        device: torch.device = torch.device("cuda"),
    ):
        self.video_path = video_path
        self.device = device
        self.decoder = VideoDecoder(str(self.video_path), device="cpu", num_ffmpeg_threads=2)
        self.total_frames = len(self.decoder)
        self.current_frame_idx = 0
        self._frame = None
        
    @property
    def fps(self):
        return self.decoder.metadata.average_fps
    
    def get_time_of_frame(self, frame_idx: int) -> float:
        return frame_idx / self.fps
    
    def set_frame(self, frame_idx: int):
        """Set the current frame"""
        if frame_idx >= self.total_frames:
            return
        
        try:
            frame = self.decoder[frame_idx].to(torch.float16).div_(255.0)
            self._frame = frame
            self.current_frame_idx = frame_idx
        except Exception as e:
            print(f"Error setting wrist video frame: {e}")
    
    def get_frame_as_numpy(self) -> np.ndarray:
        """Get current frame as numpy array for plotting"""
        if self._frame is None:
            return None
        
        # Convert frame to numpy, scale to [0, 255] and convert to uint8
        frame_np = (self._frame.cpu().numpy() * 255).astype(np.uint8)
        # Convert from CHW to HWC format
        frame_np = np.transpose(frame_np, (1, 2, 0))
        return frame_np

class ZedVideo:
    """Class to handle loading and decoding of Zed video (now using mp4 and torchcodec like WristVideo)"""
    def __init__(
        self,
        video_path: Path,
        device: torch.device = torch.device("cuda"),
    ):
        self.video_path = video_path
        self.device = device
        self.decoder = VideoDecoder(str(self.video_path), device="cpu", num_ffmpeg_threads=2)
        self.total_frames = len(self.decoder)
        self.current_frame_idx = 0
        self._frame = None
    
    @property
    def fps(self):
        return self.decoder.metadata.average_fps
    
    def get_time_of_frame(self, frame_idx: int) -> float:
        return frame_idx / self.fps
    
    def set_frame(self, frame_idx: int):
        """Set the current frame"""
        if frame_idx >= self.total_frames:
            return
        try:
            frame = self.decoder[frame_idx].to(torch.float16).div_(255.0)
            self._frame = frame
            self.current_frame_idx = frame_idx
        except Exception as e:
            print(f"Error setting zed video frame: {e}")
    
    def get_frame_as_numpy(self) -> np.ndarray:
        """Get current frame as numpy array for plotting"""
        if self._frame is None:
            return None
        frame_np = (self._frame.cpu().numpy() * 255).astype(np.uint8)
        frame_np = np.transpose(frame_np, (1, 2, 0))
        return frame_np
    
    def close(self):
        pass  # No-op for mp4

def main(
    port: int = 8080,
    insta360_path: Path = Path("data/insta360"),
):
    
    # TODO: make this a parameter
    look_at = (0.0, -0.25, 0.0)
    
    # Start viser server
    server = viser.ViserServer(port=port)
    
    # Create visualizer
    visualizer = UR5Visualizer(server)
    
    # Setup variables to track video changes
    current_video_name = None
    sim: Optional[SphericalVideo] = None
    wrist_sim: Optional[WristVideo] = None
    zed_sim: Optional[ZedVideo] = None
    last_folder = None
    
    # Add video controls
    video_info = server.gui.add_text("Video", "No video loaded")
    wrist_video_info = server.gui.add_text("Wrist Video", "No wrist video loaded")
    zed_video_info = server.gui.add_text("Zed Video", "No zed video loaded")
    
    # Add frame index displays and controls
    # This is the time relative to "peak" time which is like the entire global sequence origin.
    global_time = server.gui.add_number("Global Time (s)", 0.0, step = 1/60.0)
    prev_demo_time = 0.0
    video_peak_frame = server.gui.add_number("Video Peak Frame", 0)
    prev_video_peak_frame = 0
    
    # Add separate peak frame for wrist video
    wrist_peak_frame = server.gui.add_number("Wrist Video Peak Frame", 0)
    prev_wrist_peak_frame = 0
    
    # Add separate peak frame for zed video
    zed_peak_frame = server.gui.add_number("Zed Video Peak Frame", 0)
    prev_zed_peak_frame = 0
    
    # Add wrist video display using plotly
    wrist_plot = server.gui.add_plotly(figure=go.Figure())
    
    # Add zed video display using plotly
    zed_plot = server.gui.add_plotly(figure=go.Figure())

    @video_peak_frame.on_update
    def _(_):
        # save the video_peak_frame.value to a file
        folder = f"{visualizer.demos_folder.value}/{visualizer.folder_select.value}"
        peak_frame_path = f"{folder}/video_peak_frame.txt"
        with open(peak_frame_path, "w") as f:
            f.write(str(video_peak_frame.value))
            
    @wrist_peak_frame.on_update
    def _(_):
        # save the wrist_peak_frame.value to a file
        folder = f"{visualizer.demos_folder.value}/{visualizer.folder_select.value}"
        wrist_peak_frame_path = f"{folder}/wrist_peak_frame.txt"
        with open(wrist_peak_frame_path, "w") as f:
            f.write(str(wrist_peak_frame.value))
            
    @zed_peak_frame.on_update
    def _(_):
        # save the zed_peak_frame.value to a file
        folder = f"{visualizer.demos_folder.value}/{visualizer.folder_select.value}"
        zed_peak_frame_path = f"{folder}/zed_peak_frame.txt"
        with open(zed_peak_frame_path, "w") as f:
            f.write(str(zed_peak_frame.value))
            
    # Add rotation offset controls
    rot_x = server.gui.add_slider("Rotation X", min=-180.0, max=180.0, initial_value=0.0, step=5.0)
    rot_y = server.gui.add_slider("Rotation Y", min=-180.0, max=180.0, initial_value=0.0, step=5.0)
    rot_z = server.gui.add_slider("Rotation Z", min=-180.0, max=180.0, initial_value=90.0, step=5.0)

    def update_video(guihandle=None):
        nonlocal sim, wrist_sim, zed_sim, current_video_name, last_folder
        # Get the video peak frame
        folder = f"{visualizer.demos_folder.value}/{visualizer.folder_select.value}"
        peak_frame_path = f"{folder}/video_peak_frame.txt"
        # check if it exists
        if os.path.exists(peak_frame_path):
            with open(peak_frame_path, "r") as f:
                video_peak_frame.value = int(f.read())
                
        # Get the wrist video peak frame if it exists
        wrist_peak_frame_path = f"{folder}/wrist_peak_frame.txt"
        if os.path.exists(wrist_peak_frame_path):
            with open(wrist_peak_frame_path, "r") as f:
                wrist_peak_frame.value = int(f.read())
                
        # Get the zed video peak frame if it exists
        zed_peak_frame_path = f"{folder}/zed_peak_frame.txt"
        if os.path.exists(zed_peak_frame_path):
            with open(zed_peak_frame_path, "r") as f:
                zed_peak_frame.value = int(f.read())
                
        print(f"update_video called, folder: {visualizer.folder_select.value}")
        
        # Skip update if folder hasn't changed
        if visualizer.folder_select.value == last_folder and sim is not None:
            print(f"Folder unchanged ({last_folder}), skipping update")
            return
            
        # Update last_folder
        last_folder = visualizer.folder_select.value
        
        try:
            # Get the video name from the current folder selection
            new_video_name = visualizer.get_video_name()
                
            print(f"Got video name: {new_video_name}")
            
            # Check if video name has changed
            if new_video_name == current_video_name and sim is not None:
                print("Video name unchanged, skipping update")
                return
            
            video_path = Path(insta360_path, new_video_name)
            print(f"Looking for video at: {video_path}")
            
            # Check if video exists
            if not video_path.exists():
                error_msg = f"Error: Video {new_video_name} not found at {video_path}"
                video_info.value = error_msg
                print(error_msg)
                return
                
            # Update video path and create new SphericalVideo
            w = 1920//4
            h = 1200//4
            f_scale = 0.25
            downsample_factor = 0.5
            K, dist_coeffs, is_fisheye = get_default_video_config(w, h)
            K[0, 0] *= f_scale
            K[1, 1] *= f_scale
            
            # Create new SphericalVideo with proper torch device
            device = torch.device("cuda")
            print(f"Creating new SphericalVideo for {video_path}")
            

            # Clear CUDA cache to help with memory issues
            torch.cuda.empty_cache()

            try:
                sim = SphericalVideo(video_path, torch.tensor(K).to(device), torch.tensor(dist_coeffs).to(device), 
                                    h, w, is_fisheye, device, crop_sizes=[224,],
                                    downsample_factor=downsample_factor)
                sim.nothread_set_frame(round(0*sim.fps))
                
                # Update current video name and info display
                current_video_name = new_video_name
                video_info.value = f"Video: {video_path.name}"
                
                print(f"Successfully updated video to: {video_path}")
                
                # Check for wrist video and load if exists
                wrist_video_path = Path(folder, "wrist_video.mp4")
                if wrist_video_path.exists():
                    print(f"Found wrist video at {wrist_video_path}")
                    try:
                        wrist_sim = WristVideo(wrist_video_path, device)
                        wrist_sim.set_frame(0)
                        wrist_video_info.value = f"Wrist Video: {wrist_video_path.name}"
                        print(f"Successfully loaded wrist video from {wrist_video_path}")
                        
                        # Update the plotly figure with the first frame
                        update_wrist_video_display()
                    except Exception as e:
                        error_msg = f"Error loading wrist video: {str(e)}"
                        wrist_video_info.value = error_msg
                        print(error_msg)
                        wrist_sim = None
                else:
                    print(f"No wrist video found at {wrist_video_path}")
                    wrist_video_info.value = "No wrist video available"
                    wrist_sim = None
                    
                    # Clear the plot when no wrist video is available
                    empty_fig = go.Figure()
                    empty_fig.update_layout(
                        margin=dict(l=0, r=0, t=0, b=0),
                        xaxis=dict(showticklabels=False, showgrid=False),
                        yaxis=dict(showticklabels=False, showgrid=False)
                    )
                    wrist_plot.figure = empty_fig
                    
                # Check for zed video and load if exists
                zed_video_path = Path(folder, "zed.mp4")
                if zed_video_path.exists():
                    print(f"Found zed video at {zed_video_path}")
                    try:
                        zed_sim = ZedVideo(zed_video_path, device)
                        zed_sim.set_frame(0)
                        zed_video_info.value = f"Zed Video: {zed_video_path.name}"
                        print(f"Successfully loaded zed video from {zed_video_path}")
                        # Update the plotly figure with the first frame
                        update_zed_video_display()
                    except Exception as e:
                        error_msg = f"Error loading zed video: {str(e)}"
                        zed_video_info.value = error_msg
                        print(error_msg)
                        zed_sim = None
                else:
                    print(f"No zed video found at {zed_video_path}")
                    zed_video_info.value = "No zed video available"
                    zed_sim = None
                    # Clear the plot when no zed video is available
                    empty_fig = go.Figure()
                    empty_fig.update_layout(
                        margin=dict(l=0, r=0, t=0, b=0),
                        xaxis=dict(showticklabels=False, showgrid=False),
                        yaxis=dict(showticklabels=False, showgrid=False)
                    )
                    zed_plot.figure = empty_fig
            except Exception as e:
                error_msg = f"Error initializing video: {str(e)}"
                video_info.value = error_msg
                print(error_msg)
                sim = None
            
        except Exception as e:
            error_msg = f"Error updating video: {str(e)}"
            video_info.value = error_msg
            print(f"Error in update_video: {e}")
    
    def update_wrist_video_display():
        """Update the plotly figure with the current wrist camera frame"""
        if wrist_sim is None or wrist_sim._frame is None:
            return
            
        # Get the current frame as numpy array
        frame = wrist_sim.get_frame_as_numpy()
        if frame is None:
            return
            
        # Create plotly figure with the frame
        # fig = go.Figure(data=go.Image(z=frame))
        
        # # Update layout to minimize margins and axis markings
        # fig.update_layout(
        #     margin=dict(l=0, r=0, t=0, b=0),
        #     xaxis=dict(showticklabels=False, showgrid=False),
        #     yaxis=dict(showticklabels=False, showgrid=False, scaleanchor="x", scaleratio=1)
        # )
        
        # # Update the plot
        # wrist_plot.figure = fig
        server.scene.add_image("wrist", frame, 500,500)
    
    def update_zed_video_display():
        """Update the plotly figure with the current zed camera frame"""
        if zed_sim is None or zed_sim._frame is None:
            return
        
        # Get the current frame as numpy array
        frame = zed_sim.get_frame_as_numpy()
        if frame is None:
            return
            
        # Create plotly figure with the frame   
        fig = go.Figure(data=go.Image(z=frame))
        
        # Update layout to minimize margins and axis markings
        fig.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis=dict(showticklabels=False, showgrid=False),   
            yaxis=dict(showticklabels=False, showgrid=False, scaleanchor="x", scaleratio=1)
        )
        
        # Update the plot
        zed_plot.figure = fig
        
        
    # Override the visualizer's on_folder_selected to also trigger our video update
    original_on_folder_selected = visualizer.on_folder_selected
    def enhanced_on_folder_selected(guihandle):
        # Call the original handler first
        original_on_folder_selected(guihandle)
        # Then update our video
        update_video()
    visualizer.on_folder_selected = enhanced_on_folder_selected
    visualizer.folder_select.on_update(visualizer.on_folder_selected)

    original_on_sequence_selected = visualizer.on_sequence_selected
    def enhanced_on_sequence_selected(guihandle):
        original_on_sequence_selected(guihandle)
        global_time.value = visualizer.current_timestamps[0] - visualizer.peak_time
    visualizer.on_sequence_selected = enhanced_on_sequence_selected
    visualizer.sequence_select.on_update(visualizer.on_sequence_selected)
    
    # Initial video load attempt - give UI time to initialize first
    time.sleep(0.2)
    
    # Add a periodic update check (every 5 seconds)
    last_update_time = time.time()
    update_interval = 5  # seconds
    
    while True:
        clients = server.get_clients()
        if len(clients) == 0:
            time.sleep(0.01)
            continue
        
        # Periodic update check
        current_time = time.time()
        if current_time - last_update_time > update_interval:
            if last_folder != visualizer.folder_select.value or sim is None:
                print("Periodic check: folder changed or sim is None, updating video")
                update_video()
            last_update_time = current_time
        
        client = clients[max(clients.keys())]
        # client.camera.look_at = look_at
        
        # Create base rotation from camera
        R = SO3(torch.from_numpy(client.camera.wxyz).float().cuda())
        
        # Create rotation offset from slider values (convert degrees to radians)
        rx = math.radians(rot_x.value)
        ry = math.radians(rot_y.value)
        rz = math.radians(rot_z.value)
        
        # Create rotation matrices for each axis
        Rx = SO3.from_x_radians(torch.tensor([rx]).cuda())
        Ry = SO3.from_y_radians(torch.tensor([ry]).cuda())
        Rz = SO3.from_z_radians(torch.tensor([rz]).cuda())
        
        # Apply the rotation offset (order: Z, Y, X)
        R_offset = Rz @ Ry @ Rx @ R
        
        # Only render if sim is available
        if sim is not None and sim._frame is not None:
            frame = sim.render_image(R_offset)
            view = (255 * frame).byte().cpu().numpy().transpose(1, 2, 0)
            aspect = client.camera.aspect
            view = pad_to_aspect_ratio(view, aspect)
            server.scene.set_background_image(view, jpeg_quality=99)
            time.sleep(0.03)
        
        
        if (global_time.value != prev_demo_time or video_peak_frame.value != prev_video_peak_frame or wrist_peak_frame.value != prev_wrist_peak_frame ) and sim is not None:
            prev_demo_time = global_time.value
            prev_video_peak_frame = video_peak_frame.value
            prev_wrist_peak_frame = wrist_peak_frame.value
            robot_time = global_time.value + visualizer.peak_time
            # Update robot configuration
            visualizer.advance_to_time(robot_time)
            
            # Calculate separate time for each video based on their respective peak frames
            video_time = global_time.value + sim.get_time_of_frame(video_peak_frame.value)
                
            # Update video if sim is available
            if sim is not None:
                # Advance the video
                try:
                    sim.nothread_set_frame(round(video_time*sim.fps))
                except Exception as e:
                    print(f"Error setting video frame: {e}")
                    
            # Update wrist video if available with its own timing
            if wrist_sim is not None:
                try:
                    wrist_time = global_time.value + wrist_sim.get_time_of_frame(wrist_peak_frame.value)
                    wrist_sim.set_frame(round(wrist_time*wrist_sim.fps))
                    update_wrist_video_display()
                except Exception as e:
                    print(f"Error setting wrist video frame: {e}")



if __name__ == "__main__":
    tyro.cli(main)