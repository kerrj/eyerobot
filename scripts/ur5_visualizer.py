"""UR5 Robot URDF visualizer with H5 motion playback

This script loads the UR5 robot model from robot_descriptions and
visualizes the joint trajectories from h5 files created by data_collect.py.
"""

from __future__ import annotations

import glob
import os
import time
from typing import List, Dict, Any, Optional

import h5py
import numpy as np
from robot_descriptions.loaders.yourdfpy import load_robot_description

import viser
from viser.extras import ViserUrdf
from pathlib import Path
import os

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
        
        # Initialize with zero joint configuration
        self.initial_config = np.zeros(len(self.viser_urdf.get_actuated_joint_limits()))
        self.viser_urdf.update_cfg(self.initial_config)
        
        # State for visualization
        self.peak_time = None
        self.peak_marker = None
        
        # Setup GUI controls
        self.setup_gui()
        
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
        
        # Add playback controls
        with self.server.gui.add_folder("Playback controls"):
            self.play_button = self.server.gui.add_button("Play", disabled=True)
            self.play_button.on_click(self.play_trajectory)
            
            self.stop_button = self.server.gui.add_button("Stop", disabled=True, visible=False)
            self.stop_button.on_click(self.stop_playback)
            
            self.playback_speed = self.server.gui.add_slider(
                "Playback speed", 
                min=0.1, 
                max=2.0, 
                step=0.1, 
                initial_value=1.0
            )
            
            self.loop_playback = self.server.gui.add_checkbox(
                "Loop playback", 
                initial_value=False
            )
        
        # Refresh demos on startup
        self.refresh_demos(None)
    
    def refresh_demos(self, _):
        """Refresh the list of demo folders"""
        # try:
        demo_dirs = sorted(glob.glob(f"{self.demos_folder.value}/*"))
        assert len(demo_dirs) > 0
        demo_dirs = [os.path.basename(d) for d in demo_dirs if os.path.isdir(d)]
        self.folder_select.options = demo_dirs
        self.folder_select.value = demo_dirs[-1]
        print(self.folder_select.value)
    
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
            self.play_button.disabled = True
            return
            
        # Get h5 files in the selected folder
        h5_files = sorted(glob.glob(f"{folder_path}/*.h5"))
        h5_files = [os.path.basename(f) for f in h5_files]
        print(h5_files)
        
        self.sequence_select.options = h5_files
        self.sequence_select.disabled = False
        
        # Try to load peak time from the selected folder
        self.load_peak_time()
        
        if h5_files:
            self.sequence_select.value = h5_files[0]
            self.play_button.disabled = False
        else:
            self.play_button.disabled = True
    
    def load_peak_time(self):
        """Load peak time from peak_time.txt if it exists"""
        peak_time_path = f"{self.demos_folder.value}/{self.folder_select.value}/peak_time.txt"
        try:
            if os.path.exists(peak_time_path):
                with open(peak_time_path, 'r') as f:
                    self.peak_time = float(f.read().strip())
                print(f"Loaded peak time: {self.peak_time}")
            else:
                self.peak_time = None
                print("No peak time file found")
        except Exception as e:
            print(f"Error loading peak time: {e}")
            self.peak_time = None
    
    def load_trajectory(self, h5_path: str) -> np.ndarray:
        """Load joint trajectory from h5 file"""
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
            
            return joint_data, time_diffs, gripper_data
    
    def play_trajectory(self, _):
        """Play the selected trajectory"""
        if not self.sequence_select.value:
            return
            
        self.play_button.visible = False
        self.stop_button.visible = True
        self.stop_button.disabled = False
        
        # Get full path to the h5 file
        h5_path = f"{self.demos_folder.value}/{self.folder_select.value}/sequences/{self.sequence_select.value}"
        
        try:
            # Load trajectory data
            joint_data, time_diffs, gripper_data = self.load_trajectory(h5_path)
            
            # Start playback thread
            self._playing = True
            self._play_trajectory_loop(joint_data, time_diffs, gripper_data)
        except Exception as e:
            print(f"Error playing trajectory: {e}")
            self.stop_playback(None)
    
    def _play_trajectory_loop(self, joint_data: np.ndarray, time_diffs: np.ndarray, gripper_data: Optional[np.ndarray] = None):
        """Play the trajectory in a loop"""
        i = 0            
        current_time = 0
        replay_time = time.perf_counter() 
        while self._playing:
            gripper_joint_data = (gripper_data[i] / 255) * 0.8
            # Update robot configuration with current joint values
            self.viser_urdf.update_cfg(np.append(joint_data[i],gripper_joint_data))
            time.sleep(0.001)

                
            # Update current time
            current_time += time_diffs[i]
            
            # Calculate time to advance based on playback speed
            real_elapsed_time = (time.perf_counter() - replay_time) * self.playback_speed.value
            replay_time = time.perf_counter()
            advance_time = 0
            frames_to_skip = 0
            
            # Skip ahead to the next meaningful frame
            while advance_time < real_elapsed_time and (i + frames_to_skip + 1) < len(time_diffs):
                frames_to_skip += 1
                advance_time += time_diffs[i + frames_to_skip]
            
            # Actually skip ahead in time rather than sleeping
            if frames_to_skip > 0:
                i += frames_to_skip
                current_time += advance_time
            else:
                i += 1
            
            # Check for end of data
            if i >= len(joint_data):
                if self.loop_playback.value:
                    i = 0  # Loop back to start
                    current_time = 0  # Reset current time
                else:
                    break  # End playback
        
        # If playback ended naturally (not stopped), reset controls
        if self._playing:
            self.stop_playback(None)
    
    def stop_playback(self, _):
        """Stop the current playback"""
        self._playing = False
        self.play_button.visible = True
        self.stop_button.visible = False
        self.play_button.disabled = False


def main():
    # Start viser server
    server = viser.ViserServer()
    
    # Create visualizer
    visualizer = UR5Visualizer(server)
    
    # Sleep to keep the server running
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("Shutting down...")


if __name__ == "__main__":
    main() 