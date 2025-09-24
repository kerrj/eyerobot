import glob
import h5py
import numpy as np

files = glob.glob("data/demos/towel_*/processed/*/*.h5") + glob.glob("data/demos/clean_brush/*/processed/*/*.h5") + glob.glob("data/demos/eraser/*/processed/*/*.h5") + glob.glob("data/demos/estop*/processed/*/*.h5")
print(f"Number of files: {len(files)}")
if not files:
    print("No files found matching the pattern.")
else:
    all_joint_data = []
    for filename in files:
        print(f"Processing file: {filename}")
        try:
            with h5py.File(filename, "r") as f:
                if "joint_data" in f:
                    joint_data = f["joint_data"][:] # Load data into memory as numpy array
                    all_joint_data.append(joint_data) # Collect data for overall stats

                else:
                    print(f"  'joint_data' key not found in {filename}")
        except Exception as e:
            print(f"  Error processing file {filename}: {e}")
        print("-" * 20)

    if all_joint_data:
        # Calculate overall statistics across all files
        combined_joint_data = np.concatenate(all_joint_data, axis=0)
        print("Overall Statistics Across All Files:")
        print(f"  Total Shape: {combined_joint_data.shape}")
        print(f"  Overall Min: {np.min(combined_joint_data):.4f}")
        print(f"  Overall Max: {np.max(combined_joint_data):.4f}")
        print(f"  Overall Mean: {np.mean(combined_joint_data):.4f}")
        print(f"  Overall Std Dev: {np.std(combined_joint_data):.4f}")

        # Overall Per-joint statistics
        print("  Overall Per-joint statistics:")
        print(f"    Min: {np.min(combined_joint_data, axis=0)}")
        print(f"    Max: {np.max(combined_joint_data, axis=0)}")
        print(f"    Mean: {np.mean(combined_joint_data, axis=0)}")
        print(f"    Std Dev: {np.std(combined_joint_data, axis=0)}")
        
        # all_joint_data contains a list of the episodes so a list of T, 6 arrays.
        # Calculate relative chunk delta min/max for action chunk of 30
        print("\nRelative Action Statistics (30-frame chunks):")
        all_deltas = []
        
        for episode_data in all_joint_data:
            T, num_joints = episode_data.shape
            if T < 30:
                continue
                
            # Create sliding windows of size 30
            for start_idx in range(T - 29):
                window = episode_data[start_idx:start_idx + 30]  # Shape: (30, 6)
                
                # Calculate relative changes within the window (T=1 to T=29 relative to T=0)
                reference = window[0]  # T=0 as reference
                deltas = window[1:] - reference  # Shape: (29, 6)
                all_deltas.append(deltas)
        
        if all_deltas:
            # Combine all deltas and calculate statistics
            all_deltas = np.concatenate(all_deltas, axis=0)  # Shape: (N*29, 6)
            
            min_deltas = np.min(all_deltas, axis=0)
            max_deltas = np.max(all_deltas, axis=0)
            mean_deltas = np.mean(all_deltas, axis=0)
            std_deltas = np.std(all_deltas, axis=0)
            
            print(f"  Relative action delta shape: {all_deltas.shape}")
            print(f"  Min deltas per joint: {min_deltas}")
            print(f"  Max deltas per joint: {max_deltas}")
            print(f"  Mean deltas per joint: {mean_deltas}")
            print(f"  Std deltas per joint: {std_deltas}")
        else:
            print("  No episodes with sufficient length (>= 30 frames) found.")
    else:
        print("No 'joint_data' found in any processed files.") 