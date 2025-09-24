import glob
import h5py
import torch
import numpy as np
import viser
import viser.extras
import time
import os
from pathlib import Path
import tyro
import random

# --- Global variables to hold state ---
trajectories = []             # List to hold currently loaded trajectory data
urdfs = []                    # List to hold the fixed pool of ViserUrdf objects
all_sequence_dirs_full = []   # Store all found sequence directory paths
max_len = 0                   # Max length of currently loaded trajectories
server = None                 # Viser server instance
time_slider = None            # Viser slider for time control
urdf_path_global = None       # Path to the URDF file
num_urdfs_to_display_global = 0 # Max number of URDFs to manage
# ---

def load_and_display_subset(sequence_dirs_subset):
    """Loads data for a subset of directories and updates Viser URDF visibility."""
    global trajectories, urdfs, max_len, server, time_slider, num_urdfs_to_display_global

    # Clear previous trajectory data ONLY
    trajectories.clear()
    max_len = 0

    print(f"\nLoading subset of {len(sequence_dirs_subset)} sequences...")

    # 2. Load Data for the subset
    for i, seq_dir in enumerate(sequence_dirs_subset):
        # Limit loading to the number of URDFs we have placeholders for
        if len(trajectories) >= num_urdfs_to_display_global:
            print(f"Reached URDF display limit ({num_urdfs_to_display_global}). Not loading more.")
            break
        try:
            h5_files = glob.glob(os.path.join(seq_dir, "*.h5"))
            if not h5_files:
                # print(f"Warning: No h5 file found in {seq_dir}") # Less verbose
                continue
            h5_path = h5_files[0]

            with h5py.File(h5_path, "r") as f:
                joint_data = torch.from_numpy(f["joint_data"][:])
                grip_key = "interpolated_gripper_data" if "interpolated_gripper_data" in f else "gripper_data"
                if grip_key not in f:
                    gripper_data = torch.zeros_like(joint_data[..., :1])
                else:
                    gripper_data = torch.from_numpy(f[grip_key][:])/255.0
                    gripper_data = gripper_data.float() / 255.0
                    if gripper_data.ndim == 1:
                        gripper_data = gripper_data.unsqueeze(-1)

                trajectory = torch.cat([joint_data, gripper_data], dim=-1)
                trajectories.append(trajectory.float())
                max_len = max(max_len, trajectory.shape[0])

        except Exception as e:
            print(f"Error loading data from {seq_dir}: {e}")
            continue

    if not trajectories:
        print("No trajectories loaded successfully for this subset.")
        # Hide all URDFs if no trajectories loaded
        if time_slider is not None:
            with server.atomic():
                time_slider.max = 0
                time_slider.value = 0
        return

    print(f"Loaded {len(trajectories)} trajectories. New max length: {max_len}")

    # Update Time Slider max value
    if time_slider is not None:
        with server.atomic():
            time_slider.max = max(0, max_len - 1) # Ensure max is non-negative
            time_slider.value = min(time_slider.value, time_slider.max)

    # Trigger update for the new poses at the current slider value
    update_poses()
    print("Subset loaded and displayed using existing URDFs.")


def update_poses():
    """Callback function to update URDF poses based on slider."""
    global trajectories, urdfs, time_slider
    if time_slider is None or not trajectories:
        return

    timestep = time_slider.value
    # Iterate only up to the number of currently loaded trajectories
    for i in range(len(trajectories)):
        traj = trajectories[i]
        urdf = urdfs[i] # Assumes urdfs[i] corresponds to trajectories[i]

        current_step = min(timestep, traj.shape[0] - 1)
        if current_step < 0: continue

        pose = traj[current_step].cpu().numpy()
        urdf.update_cfg(pose)


@tyro.cli
def main(
    data_glob: str = "data/demos/clean_brush_pick_only/*/processed/*",
    urdf_path: Path = Path(os.path.dirname(__file__)) / "../urdf/ur5e_with_robotiq_gripper.urdf",
    num_urdfs_to_display: int = 50, # Max number of URDFs to create and manage
):
    """Visualize robot trajectories from H5 files using Viser."""
    global all_sequence_dirs_full, server, time_slider, urdf_path_global, num_urdfs_to_display_global, urdfs

    urdf_path_global = urdf_path
    num_urdfs_to_display_global = num_urdfs_to_display

    # 1. Find All Data initially
    all_sequence_dirs_full = sorted(glob.glob(data_glob))
    if not all_sequence_dirs_full:
        print(f"No sequences found matching glob: {data_glob}")
        return
    print(f"Found {len(all_sequence_dirs_full)} total sequences.")

    # 3. Setup Viser
    server = viser.ViserServer()
    server.scene.enable_default_lights()
    server.scene.add_grid(
        "/grid", width=2, height=2, position=(0.0, 0.0, 0.0)
    )

    # 4. Create the fixed pool of URDFs ONCE
    print(f"Creating {num_urdfs_to_display_global} URDF objects...")
    for i in range(num_urdfs_to_display_global):
        # Generate distinct colors
        urdf = viser.extras.ViserUrdf(
            server,
            urdf_or_path=urdf_path_global,
            root_node_name=f'/traj_{i}', # Unique names are good practice
        )
        urdfs.append(urdf)
    print("URDF objects created.")

    # 5. Add Time Slider
    time_slider = server.gui.add_slider(
        "Timestep", min=0, max=1, step=1, initial_value=0
    )
    @time_slider.on_update
    def _(event: viser.GuiEvent):
        update_poses()

    # -- Randomization Button --
    randomize_button = server.gui.add_button("Randomize Trajectories")
    @randomize_button.on_click
    def _(event: viser.GuiEvent):
        print("Randomizing trajectories...")
        if not all_sequence_dirs_full:
            print("No sequences available to randomize.")
            return

        random.shuffle(all_sequence_dirs_full)
        # Select subset based on display limit, even if fewer sequences are available
        num_to_take = min(num_urdfs_to_display_global, len(all_sequence_dirs_full))
        new_subset = all_sequence_dirs_full[:num_to_take]
        load_and_display_subset(new_subset)
    # -------------------------

    # Initial Load
    num_initial_load = min(num_urdfs_to_display_global, len(all_sequence_dirs_full))
    initial_subset = all_sequence_dirs_full[:num_initial_load]
    load_and_display_subset(initial_subset)


    print("\n Viser server running. Open your browser to the printed address.")
    while True:
        time.sleep(0.1)

if __name__ == "__main__":
    main()