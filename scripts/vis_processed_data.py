"""Process data visualizer with synchronized video playback

This script loads processed data and visualizes both the robot motion 
and the corresponding trimmed video in viser.
The data is already synchronized so no peak time alignment is needed.
"""

from __future__ import annotations

import time
import viser
import tyro
from viser import ViserServer
import torch
import numpy as np
import h5py
import cv2
from typing import Tuple, Optional, List
from pathlib import Path
import os
import glob
from robot_descriptions.loaders.yourdfpy import load_robot_description
from viser.extras import ViserUrdf
from eye.image_utils import pad_to_aspect_ratio
from torchcodec.decoders import VideoDecoder

def main(
    processed_dir: Path,
    port: int = 8080,
):
    # Start viser server
    server = viser.ViserServer(port=port)
    server.scene.enable_default_lights()
    
    # Create UR5 visualizer directly
    urdf_path = os.path.join(os.path.dirname(__file__),'../urdf/ur5e_with_robotiq_gripper.urdf')
    viser_urdf = ViserUrdf(
        server,
        urdf_or_path=Path(urdf_path),
    )
    
    # Create grid
    server.scene.add_grid(
        "/grid",
        width=2,
        height=2,
        position=(
            0.0,
            0.0,
            # Get the minimum z value of the trimesh scene
            viser_urdf._urdf.scene.bounds[0, 2],
        ),
    )
    
    # Setup variables to track data
    current_video_path = None
    current_h5_path = None
    frame_index = 0
    joint_data = None
    timestamps = None
    gripper_joint_data = None
    # GUI controls
    video_info = server.gui.add_text("Video", "No video loaded")
    
    # Add frame index displays and controls
    frame_slider = server.gui.add_slider("Frame", 0, 1, step=1, initial_value=0)
    frame_slider.on_update(lambda _: show_frame(frame_slider.value))
    video = None
    
    def load_sequence():
        """Load the data and video from the processed directory"""
        nonlocal current_video_path, current_h5_path, joint_data, timestamps, frame_index, gripper_joint_data
        
        # Find the h5 file and video file
        h5_files = glob.glob(f"{processed_dir}/downsampled_*.h5")
        video_file = f"{processed_dir}/trimmed.mp4"
        if not h5_files or not os.path.exists(video_file):
            video_info.value = f"Error: Missing files in {processed_dir}"
            return
        
        current_h5_path = h5_files[0]
        current_video_path = video_file
        
        # Load h5 data
        with h5py.File(current_h5_path, 'r') as f:
            joint_data = np.array(f['joint_data'])
            gripper_joint_data = (np.array(f['interpolated_gripper_data']) / 255) * 0.9
            timestamps = np.array(f['timestamps'])
        # Open video
        nonlocal video
        video = VideoDecoder(current_video_path, device='cpu')
        # set slider min and max
        frame_slider.min = 0
        frame_slider.max = len(video) - 1
        frame_slider.value = 0
    
    def show_frame(index: int):
        """Show a specific frame of data and video"""
        if video is None or joint_data is None:
            return
        
        
        # Set robot configuration
        if index < len(joint_data):
            gd = gripper_joint_data[index]
            # Update robot configuration with the specified frame

        
        # Get video frame
        
        # Get the aspect ratio of the client's viewport
        clients = server.get_clients()
        aspect = None
        if clients:
            frame = video[index].permute(1, 2, 0)[::3,::3].cpu().numpy()
            client = clients[max(clients.keys())]
            aspect = client.camera.aspect
                
            # If we have a valid aspect ratio, pad the image
            if aspect:
                frame = pad_to_aspect_ratio(frame, aspect)
            
            # Set as background image
            with client.atomic():   
                server.scene.set_background_image(frame, jpeg_quality=85)
                viser_urdf.update_cfg(np.append(joint_data[index],gd))
    
    
    # Load the sequence on startup
    load_sequence()
    import matplotlib.pyplot as plt
    if gripper_joint_data is not None:
        plt.plot(gripper_joint_data)
        plt.savefig("gripper_joint_data.png")
    playing = server.gui.add_checkbox("Playing", initial_value=True)
    while True:
        time.sleep(0.05)
        if playing.value and video is not None:
            frame_slider.value = (frame_slider.value + 1) % len(video)

if __name__ == "__main__":
    tyro.cli(main)
