import h5py
import numpy as np
from pathlib import Path
import os
from typing import Dict
from dm_control import mjcf
import mujoco
import time
from moviepy import ImageSequenceClip
from datetime import datetime

from typing import Optional
from gello.robots.sim_robot import attach_hand_to_arm
def build_scene(robot_xml_path: str, arena, name, gripper_xml_path: Optional[str] = None, color: Optional[list] = None):
    # assert robot_xml_path.endswith(".xml")

    
    with open(robot_xml_path, 'r') as f:
        robot_xml_string = f.read()
    # Convert XML string to XML object using lxml's etree
    from lxml import etree
    xml_root = etree.fromstring(robot_xml_string)
    xml_root.set('model',name)
    robot_xml_string = etree.tostring(xml_root, pretty_print=True, encoding='unicode')
    arm_simulate = mjcf.from_xml_string(robot_xml_string,model_dir=os.path.split(robot_xml_path)[0])
    # arm_copy = mjcf.from_path(robot_xml_path)

    # Turn off collisions for all geometries in this robot
    for geom in arm_simulate.find_all('geom'):
        geom.contype = 0
        geom.conaffinity = 0

    if gripper_xml_path is not None:
        gripper_simulate = mjcf.from_path(gripper_xml_path)
        # Turn off collisions for gripper geometries too
        for geom in gripper_simulate.find_all('geom'):
            geom.contype = 0
            geom.conaffinity = 0
        attach_hand_to_arm(arm_simulate, gripper_simulate)
    
    # Get the number of DoFs in the new robot before attaching
    physics = mjcf.Physics.from_mjcf_model(arm_simulate)
    new_nq = physics.model.nq  # Number of position coordinates
    new_nu = physics.model.nu  # Number of velocity coordinates

    # Handle keyframes in the arena
    for key in arena.find_all('key'):
        if key.qpos is not None:
            key.qpos = np.concatenate([key.qpos, np.zeros(new_nq)])
        if hasattr(key, 'ctrl') and key.ctrl is not None:
            key.ctrl = np.concatenate([key.ctrl, np.zeros(new_nu)])

    # Handle keyframes in the robot itself
    for key in arm_simulate.find_all('key'):
        if key.qpos is not None:
            # For the first robot, expand with zeros
            # For subsequent robots, prepend zeros for the previous robots' DoFs
            if len(arena.find_all('joint')) > 0:  # If there are already joints in the arena
                key.qpos = np.concatenate([np.zeros(len(arena.find_all('joint'))), key.qpos])
        if hasattr(key, 'ctrl') and key.ctrl is not None:
            if len(arena.find_all('actuator')) > 0:  # If there are already actuators in the arena
                key.ctrl = np.concatenate([np.zeros(len(arena.find_all('actuator'))), key.ctrl])

    # Now attach the robot
    arena.worldbody.attach(arm_simulate)
    # arena.worldbody.attach(arm_copy)

    # Apply color to the robot's material if specified
    if color is not None:
        for material in arm_simulate.asset.find_all('material'):
            material.rgba = color

    return arena
def visualize_action_chunk(gt_action_chunk, pred_action_chunk, save_dir):
    assert gt_action_chunk.shape == pred_action_chunk.shape
    # Get directory of this file
    current_dir = Path(__file__).parent.absolute()
    mujoco_root_folder = os.path.join(current_dir,'../../dependencies/gello_software/third_party/mujoco_menagerie')
    robot_xml_path = os.path.join(mujoco_root_folder,"universal_robots_ur5e/ur5e.xml")
    gripper_xml_path = os.path.join(mujoco_root_folder,"robotiq_2f85/2f85.xml")
    arena = mjcf.RootElement()
    # First robot (ground truth) will be red
    arena = build_scene(robot_xml_path, arena, "ur5e_1", gripper_xml_path, color=[0.2, 0.8, 0.2, 1.0])
    # Second robot (prediction) will be green
    arena = build_scene(robot_xml_path, arena, "ur5e_2", gripper_xml_path, color=[0.8, 0.2, 0.2, 1.0])
    
    assets: Dict[str, str] = {}
    for asset in arena.asset.all_children():
        if asset.tag == "mesh":
            f = asset.file
            assets[f.get_vfs_filename()] = asset.file.contents
    xml_string = arena.to_xml_string()
    # save xml_string to file
    with open("arena.xml", "w") as f:
        f.write(xml_string)

    model = mujoco.MjModel.from_xml_string(xml_string, assets)
    data = mujoco.MjData(model)

    # Setup renderer for offscreen rendering
    width = 640
    height = 480
    renderer = mujoco.Renderer(model, height=height, width=width)
    
    # Configure camera
    camera = mujoco.MjvCamera()  # Get the first camera from the tuple
    camera.distance = 2.5
    camera.azimuth = -90.0 # Look from the front

    num_joints = model.nu
    num_trajectories, trajectory_length, _ = gt_action_chunk.shape

    # Calculate grid dimensions
    max_cols = 5
    num_cols = min(num_trajectories, max_cols)
    num_rows = (num_trajectories + num_cols - 1) // num_cols  # Ceiling division

    # Initialize video sequences
    video_sequences = []
    
    # Process all trajectories
    for i in range(num_trajectories):
        gt_trajectory = gt_action_chunk[i]
        pred_trajectory = pred_action_chunk[i]
        video_sequence = []
        
        for step in range(trajectory_length):
            # Set joint positions for both robots
            if not np.isnan(gt_trajectory[step]).any() and not np.isnan(pred_trajectory[step]).any():
                data.qpos[:num_joints //2] = gt_trajectory[step]
                data.qpos[num_joints:num_joints + num_joints//2] = pred_trajectory[step]
            
            # Render frame
            mujoco.mj_step(model, data)
            renderer.update_scene(data, camera)
            rgb_array = renderer.render()
            
            video_sequence.append(rgb_array)
        
        video_sequences.append(video_sequence)
    
    # Convert to numpy array and reshape for grid layout
    video_sequences = np.array(video_sequences)
    video_sequences = np.transpose(video_sequences, (1, 0, 2, 3, 4))
    
    # Pad the sequences if needed to make a complete grid
    if num_trajectories < num_rows * num_cols:
        padding = num_rows * num_cols - num_trajectories
        padding_shape = list(video_sequences.shape)
        padding_shape[1] = padding
        padding_shape[2:] = [height, width, 3]
        padding_array = np.zeros(padding_shape, dtype=video_sequences.dtype)
        video_sequences = np.concatenate([video_sequences, padding_array], axis=1)
    
    # Reshape to grid layout
    video_sequences = video_sequences.reshape(video_sequences.shape[0], num_rows, num_cols, height, width, 3)
    
    # Create grid layout
    grid_rows = []
    for row in range(num_rows):
        row_frames = []
        for col in range(num_cols):
            row_frames.append(video_sequences[:, row, col])
        grid_rows.append(np.concatenate(row_frames, axis=2))
    
    grid_video = np.concatenate(grid_rows, axis=1)
    
    # Save video
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(save_dir)
    output_dir.mkdir(exist_ok=True)
    video_path = output_dir / f"simulation_{timestamp}.mp4"
    
    # Create clip from frames
    clip = ImageSequenceClip(list(grid_video), fps=24)
    # Write video file
    clip.write_videofile(str(video_path), codec='libx264', fps=30)
    return str(video_path)

if __name__ == "__main__":
    # Load joint data from HDF5 file
    joint_data_path = Path("/home/eyeball/eyeball/data/demos/boris_test/processed/20250323_170927_joint_data/downsampled_20250323_170927_joint_data.h5")
    with h5py.File(joint_data_path, 'r') as f:
        joint_positions = np.array(f['joint_data'])
        joint_positions = np.hstack((joint_positions, np.zeros((joint_positions.shape[0], 1))))

    # Create a noisy version of joint positions by adding random noise
    noise_scale = 0.02  # Scale of the noise (radians)
    joint_positions_noise = joint_positions.copy()
    joint_positions_noise[:, 2] += 1  # Add 0.05 radians offset to first joint

    # Reshape joint positions from (N,7) to (num_trajectories,N,7) by repeating the same trajectory
    num_trajectories = 4  # Example with 12 trajectories
    joint_positions = np.tile(joint_positions[np.newaxis, :, :], (num_trajectories, 1, 1))
    joint_positions_noise = np.tile(joint_positions_noise[np.newaxis, :, :], (num_trajectories, 1, 1))

    visualize_action_chunk(joint_positions, joint_positions_noise, Path("action_chunk_videos"))