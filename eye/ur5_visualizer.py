"""UR5 Robot URDF visualizer with H5 motion playback

Loads the UR5 robot model from robot_descriptions and
visualizes the joint trajectories from h5 files created by data_collect.py.
"""

from __future__ import annotations

import glob
import os
import time
from typing import Optional

import h5py
import numpy as np
from robot_descriptions.loaders.yourdfpy import load_robot_description

import viser
from viser.extras import ViserUrdf
from pathlib import Path

class UR5Visualizer:
    def __init__(self, server: viser.ViserServer):
        self.server = server
        server.scene.enable_default_lights()
        
        urdf_path = os.path.join(os.path.dirname(__file__),'../urdf/ur5e_with_robotiq_gripper.urdf')
        # Load UR5 URDF
        self.viser_urdf = ViserUrdf(
            server,
            urdf_or_path=Path(urdf_path)#load_robot_description("ur5_description"),
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
                self.viser_urdf._urdf.scene.bounds[0, 2],
            ),
        )
        
        # Save the initial zero configuration but don't apply it yet
        self.initial_config = np.zeros(len(self.viser_urdf.get_actuated_joint_limits()))
        
        # State for visualization
        self.peak_time = None
        self.peak_marker = None
        self._playing = False
        
        # Cached trajectory data
        self.current_joint_data = None
        self.current_timestamps = None
        self.current_time_diffs = None
        self.current_gripper_data = None
        
        # Setup GUI controls
        self.setup_gui()
        
        # Attempt to load a trajectory and show first frame
        # Only fall back to initial config if no trajectory is available
        self.refresh_demos(None)
        
    def setup_gui(self):
        """Set up the GUI controls for selecting and playing h5 files"""
        self.demos_folder = self.server.gui.add_text(
            "Demos folder", 
            initial_value="./data/demos",
        )
        self.refresh_button = self.server.gui.add_button("Refresh demos")
        self.refresh_button.on_click(self.refresh_demos)
        
        # Add a folder dropdown
        self.folder_select = self.server.gui.add_dropdown(
            "Select folder",
            options=[""],
        )
        self.folder_select.on_update(self.on_folder_selected)
        
        # Add a sequence dropdown (initially empty)
        self.sequence_select = self.server.gui.add_dropdown(
            "Select sequence", 
            options=[""],
            disabled=True
        )
        self.sequence_select.on_update(self.on_sequence_selected)
        
    def refresh_demos(self, _):
        """Refresh the list of demo folders"""
        demo_dirs = sorted(glob.glob(f"{self.demos_folder.value}/*"))
        assert len(demo_dirs) > 0
        demo_dirs = [os.path.basename(d) for d in demo_dirs if os.path.isdir(d)]
        self.folder_select.options = demo_dirs
        self.folder_select.value = demo_dirs[-1]
        print(self.folder_select.value)

    def get_video_name(self) -> str:
        """Get the insta360 video name from the insta360_filenames.txt file"""
        video_name_path = f"{self.demos_folder.value}/{self.folder_select.value}/insta360_filenames.txt"
        # the file contains this... ['/DCIM/Camera01/VID_20250321_003911_00_070.insv', '/DCIM/Camera01/LRV_20250321_003911_01_070.lrv']
        # I want a video of the form VID_20250321_003911_00_070.mp4
        with open(video_name_path, 'r') as f:
            content = f.read()
            # Extract the first filename from the list
            insv_name = eval(content)[0]
            # Get just the VID_*.insv part
            base_name = os.path.basename(insv_name)
            # Replace .insv with .mp4
            return base_name.replace('.insv', '.mp4')
        
    def on_folder_selected(self, guihandle):
        """Handle folder selection change"""
        if not self.folder_select.value:
            return
            
        folder_path = f"{self.demos_folder.value}/{self.folder_select.value}/sequences"
        print(folder_path)
        if not os.path.exists(folder_path):
            print("folder does not exist", folder_path)
            self.sequence_select.options = []
            self.sequence_select.disabled = True
            return
            
        # Get h5 files in the selected folder
        h5_files = sorted(glob.glob(f"{folder_path}/*.h5"))
        h5_files = [os.path.basename(f) for f in h5_files]
        
        self.sequence_select.options = h5_files
        self.sequence_select.disabled = False
        
        # Try to load peak time from the selected folder
        self.load_peak_time()
        
        if h5_files:
            self.sequence_select.value = h5_files[0]
    
    def on_sequence_selected(self, guihandle):
        """Handle sequence selection change and show first frame"""
        if not self.sequence_select.value:
            return
            
        # Get full path to the h5 file
        h5_path = f"{self.demos_folder.value}/{self.folder_select.value}/sequences/{self.sequence_select.value}"
        
        # Load trajectory data and cache it
        self.current_joint_data, self.current_timestamps, self.current_time_diffs, self.current_gripper_data = self.load_trajectory(h5_path)
        gripper_joint_data = (self.current_gripper_data[0] / 255) * 0.8
        # Set robot to the first frame
        self.viser_urdf.update_cfg(np.append(self.current_joint_data[0],gripper_joint_data))

    
    def load_peak_time(self):
        """Load peak time from peak_time.txt if it exists"""
        peak_time_path = f"{self.demos_folder.value}/{self.folder_select.value}/peak_time.txt"
        if os.path.exists(peak_time_path):
            with open(peak_time_path, 'r') as f:
                self.peak_time = float(f.read().strip())
            print(f"Loaded peak time: {self.peak_time}")
        else:
            self.peak_time = None
            print("No peak time file found")
    
    def load_trajectory(self, h5_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Load joint trajectory from h5 file"""
        print(f"Loading trajectory from {h5_path}")
        with h5py.File(h5_path, 'r') as f:
            # Extract joint data and timestamps
            joint_data = np.array(f['joint_data'])
            timestamps = np.array(f['timestamps'])
            
            # Get gripper data if available
            gripper_data = None
            if 'gripper_data' in f:
                gripper_data = np.array(f['gripper_data'])
            
            # Calculate time differences for playback
            time_diffs = np.diff(timestamps, prepend=timestamps[0])
            
            return joint_data, timestamps, time_diffs, gripper_data

    def advance_to_frame(self, frame_idx):
        """
        Advance the visualization to a specific frame
        
        Args:
            frame_idx: The frame index to advance to
        """
        if not self.sequence_select.value or self.current_joint_data is None:
            return
            
        # Ensure the frame index is valid
        frame_idx = max(0, min(frame_idx, self.get_trajectory_length() - 1))
        gripper_joint_data = (self.current_gripper_data[frame_idx] / 255) * 0.8
        # Update robot configuration with the specified frame
        self.viser_urdf.update_cfg(np.append(self.current_joint_data[frame_idx],gripper_joint_data))

    def advance_to_time(self, time):
        """
        Advance the visualization to a specific time
        
        Args:
            time: The time to advance to
        """
        if not self.sequence_select.value or self.current_joint_data is None:
            return

        # Find the frame index for the specified time
        frame_idx = np.searchsorted(self.current_timestamps, time)
        self.advance_to_frame(frame_idx)

    def get_trajectory_length(self) -> int:
        """
        Get the length of the currently selected trajectory
        
        Returns:
            The number of frames in the current trajectory
        """
        if self.current_joint_data is None:
            return 0
            
        return len(self.current_joint_data)