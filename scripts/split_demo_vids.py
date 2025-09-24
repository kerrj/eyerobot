import glob
from pathlib import Path
import numpy as np
import os
import threading
from concurrent.futures import ThreadPoolExecutor
import subprocess
import tyro
import torch
def get_interpolated_gripper_data(f):
    gripper_data = f['gripper_data'][:]
    
    # Calculate the derivative to identify flat regions
    derivative = np.diff(gripper_data)
    threshold = 0.01  # Adjust this threshold as needed
    
    # Identify flat regions (where derivative is close to zero)
    flat_regions = np.abs(derivative) < threshold
    
    # Find continuous flat regions
    flat_regions = np.concatenate(([False], flat_regions, [False]))
    flat_starts = np.where(np.diff(flat_regions.astype(int)) == 1)[0]
    flat_ends = np.where(np.diff(flat_regions.astype(int)) == -1)[0]
    
    # Initialize interpolated data array
    interpolated_data = np.copy(gripper_data)
    
    # Identify long and short regions
    long_regions = []
    short_regions = []
    
    for start, end in zip(flat_starts, flat_ends):
        region_length = end - start
        if region_length >= 1000:
            long_regions.append((start, end))
        else:
            short_regions.append((start, end))
    for i in range(len(long_regions) - 1):
        current_long_end = long_regions[i][1]
        next_long_start = long_regions[i + 1][0]
        
        # Find all short regions between these two long regions
        between_short_regions = [
            (start, end) for start, end in short_regions 
            if current_long_end < start < next_long_start
        ]
        
        # Interpolate from long region end to first short region
        if between_short_regions:
            x_interp = np.arange(current_long_end, between_short_regions[0][0]+1)
            y_interp = np.round(np.interp(x_interp, [current_long_end, between_short_regions[0][0]], 
                                        [gripper_data[current_long_end], gripper_data[between_short_regions[0][0]]])).astype(np.uint8)
            interpolated_data[x_interp] = y_interp
            
            # Interpolate between short regions
            for j in range(0, len(between_short_regions)-1, 1):
                first_region = between_short_regions[j]
                second_region = between_short_regions[j+1] if j+1 < len(between_short_regions) else None
                if second_region:
                    x_interp = np.arange(first_region[0], second_region[0] + 1)
                    y_interp = np.round(np.interp(x_interp, [first_region[0], second_region[0]], 
                                               [gripper_data[first_region[0]], gripper_data[second_region[0]]])).astype(np.uint8)
                    interpolated_data[x_interp] = y_interp
            
            # Interpolate from last short region to next long region
            x_interp = np.arange(between_short_regions[-1][0], next_long_start + 1)
            y_interp = np.round(np.interp(x_interp, [between_short_regions[-1][0], next_long_start], 
                                        [gripper_data[between_short_regions[-1][0]], gripper_data[next_long_start]])).astype(np.uint8)
            interpolated_data[x_interp] = y_interp
    
    return interpolated_data

def process_sequence(h5_path, demos_folder, processed_dir, video_filename, video_peak_frame, peak_time, wrist_peak_frame, zed_peak_frame, device_id):
    """Process a single sequence file."""
    try:
        import h5py
        import scipy.interpolate as interp
        
        with h5py.File(h5_path, 'r+') as f:
            # Add interpolated gripper data processing
            interpolated_gripper_data = get_interpolated_gripper_data(f)
            try:
                f.create_dataset('interpolated_gripper_data', data=interpolated_gripper_data)
            except Exception as e:
                print(f"Already created interpolated gripper data")
            
            # Extract timestamps from h5 file
            timestamps = f['timestamps'][:]
            start_time = timestamps[0]
            end_time = timestamps[-1]
            
            # Calculate corresponding video frames relative to peak
            fps = 29.97  # Assuming 30fps, adjust if needed
            start_frame = int((start_time - peak_time) * fps) + video_peak_frame
            end_frame = int((end_time - peak_time) * fps) + video_peak_frame
            
            # Ensure positive frame numbers
            start_frame = max(0, start_frame)
            end_frame = max(start_frame + 1, end_frame)
            
            print(f"  Frame range: {start_frame} - {end_frame}")
            
            # Trim the video using ffmpeg
            seq_processed_dir = processed_dir / h5_path.stem
            seq_processed_dir.mkdir(exist_ok=True)
            video_output_path = seq_processed_dir / f"trimmed.mp4"
            
            cmd = [
                "ffmpeg", "-y",
                # Hardware acceleration flags
                "-hwaccel", "cuda",
                "-hwaccel_device", str(device_id),
                # Input file
                "-i", str(Path("data/insta360") / video_filename),
                # Accurate seeking requires -ss and -to after -i when re-encoding
                "-ss", str(start_frame / fps),
                "-to", str(end_frame / fps),
                # Video codec and options
                "-c:v", "hevc_nvenc",      # NVIDIA HEVC encoder
                "-preset", "p4",           # Encoding speed/quality preset (p1-p7)
                "-crf", "23",              # Constant Rate Factor (quality, lower=better)
                # You might need to handle audio separately if present/needed:
                # "-c:a", "aac", "-b:a", "128k", # Example: Re-encode audio to AAC
                # Or just drop audio if not needed:
                "-an",                     # No audio in the output
                str(video_output_path)
            ]
            try:
                subprocess.run(cmd, check=True, capture_output=True)
                print(f"  Trimmed video saved to {video_output_path}")
            except subprocess.CalledProcessError as e:
                print(f"  Error trimming video: {e}")
                print(f"  STDOUT: {e.stdout.decode()}")
                print(f"  STDERR: {e.stderr.decode()}")
                return
            
            # Count frames in output video
            probe_cmd = [
                "ffprobe", "-v", "error", 
                "-select_streams", "v:0", 
                "-count_packets",
                "-show_entries", "stream=nb_read_packets", 
                "-of", "csv=p=0", 
                str(video_output_path)
            ]
            
            try:
                result = subprocess.run(probe_cmd, check=True, capture_output=True, text=True)
                frame_count = int(result.stdout.strip())
                print(f"  Output video has {frame_count} frames")
            except (subprocess.CalledProcessError, ValueError) as e:
                print(f"  Error getting frame count: {e}")
                # Estimate frame count
                frame_count = end_frame - start_frame
                print(f"  Using estimated frame count: {frame_count}")
            
            # --- Wrist Video Trimming ---
            if wrist_peak_frame is not None:
                wrist_video_path = demos_folder / "wrist_video.mp4"
                if wrist_video_path.exists():
                    print(f"  Found wrist video: {wrist_video_path}")
                    
                    # Calculate corresponding wrist video frames relative to wrist peak, using the shared peak_time
                    fps = 29.97 
                    wrist_start_frame = int((start_time - peak_time) * fps) + wrist_peak_frame
                    wrist_end_frame = int((end_time - peak_time) * fps) + wrist_peak_frame
                    
                    # Ensure positive frame numbers
                    wrist_start_frame = max(0, wrist_start_frame)
                    wrist_end_frame = max(wrist_start_frame + 1, wrist_end_frame)
                    
                    print(f"  Wrist frame range: {wrist_start_frame} - {wrist_end_frame}")
                    
                    # Trim the wrist video using ffmpeg
                    wrist_video_output_path = seq_processed_dir / f"wrist_trimmed.mp4"
                    
                    wrist_cmd = [
                        "ffmpeg", "-y",
                        # Hardware acceleration flags (can potentially use the same device?)
                        "-hwaccel", "cuda",
                        "-hwaccel_device", str(device_id), 
                        # Input file
                        "-i", str(wrist_video_path),
                        # Accurate seeking
                        "-ss", str(wrist_start_frame / fps),
                        "-to", str(wrist_end_frame / fps),
                        # Video codec and options (match main video settings)
                        "-c:v", "hevc_nvenc",      
                        "-preset", "p4",           
                        "-crf", "23",              
                        "-an",                     # No audio
                        str(wrist_video_output_path)
                    ]
                    try:
                        subprocess.run(wrist_cmd, check=True, capture_output=True)
                        print(f"  Trimmed wrist video saved to {wrist_video_output_path}")
                    except subprocess.CalledProcessError as e:
                        print(f"  Error trimming wrist video: {e}")
                        print(f"  STDOUT: {e.stdout.decode()}")
                        print(f"  STDERR: {e.stderr.decode()}")
                        # Don't return here, main processing might still succeed
                else:
                    print(f"  Wrist video not found at {wrist_video_path}. Skipping wrist trim.")

            # --- ZED Video Trimming ---
            if zed_peak_frame is not None:
                zed_video_path = demos_folder / "zed.mp4"
                if zed_video_path.exists():
                    print(f"  Found zed video: {zed_video_path}")

                    # Calculate corresponding zed video frames relative to zed peak, using the shared peak_time
                    fps = 30
                    zed_start_frame = int((start_time - peak_time) * fps) + zed_peak_frame
                    zed_end_frame = int((end_time - peak_time) * fps) + zed_peak_frame

                    # Ensure positive frame numbers
                    zed_start_frame = max(0, zed_start_frame)
                    zed_end_frame = max(zed_start_frame + 1, zed_end_frame)

                    print(f"  ZED frame range: {zed_start_frame} - {zed_end_frame}")

                    # Trim the zed video using ffmpeg
                    zed_video_output_path = seq_processed_dir / f"zed_trimmed.mp4"

                    zed_cmd = [
                        "ffmpeg", "-y",
                        "-hwaccel", "cuda",
                        "-hwaccel_device", str(device_id),
                        "-i", str(zed_video_path),
                        "-ss", str(zed_start_frame / fps),
                        "-to", str(zed_end_frame / fps),
                        "-c:v", "hevc_nvenc",
                        "-preset", "p4",
                        "-crf", "23",
                        "-an",
                        str(zed_video_output_path)
                    ]
                    try:
                        subprocess.run(zed_cmd, check=True, capture_output=True)
                        print(f"  Trimmed zed video saved to {zed_video_output_path}")
                    except subprocess.CalledProcessError as e:
                        print(f"  Error trimming zed video: {e}")
                        print(f"  STDOUT: {e.stdout.decode()}")
                        print(f"  STDERR: {e.stderr.decode()}")
                else:
                    print(f"  ZED video not found at {zed_video_path}. Skipping zed trim.")

            # Downsample h5 data to match video frame count
            h5_output_path = seq_processed_dir / f"downsampled_{h5_path.name}"
            
            # Create downsampled h5 file
            with h5py.File(h5_output_path, 'w') as out_f:
                # Create new timestamps linearly spaced to match video frames
                new_timestamps = np.linspace(start_time, end_time, frame_count)
                out_f.create_dataset('timestamps', data=new_timestamps)
                
                # Downsample all datasets containing joint data using spline interpolation
                for key in f.keys():
                    if key == 'timestamps':
                        continue
                        
                    data = f[key][:]
                    
                    # Check if dataset is compatible with interpolation
                    if len(data) == len(timestamps):
                        # For each dimension in the data
                        if len(data.shape) > 1:
                            # Multi-dimensional data (like joint positions)
                            downsampled = np.zeros((frame_count, *data.shape[1:]))
                            for i in range(data.shape[1]):
                                spline = interp.splrep(timestamps, data[:, i])
                                downsampled[:, i] = interp.splev(new_timestamps, spline)
                        else:
                            # One-dimensional data
                            spline = interp.splrep(timestamps, data)
                            downsampled = interp.splev(new_timestamps, spline)
                            
                        out_f.create_dataset(key, data=downsampled)
                    else:
                        # Just copy the dataset if it doesn't match timestamps
                        out_f.create_dataset(key, data=data)
            
            print(f"  Downsampled motion data saved to {h5_output_path}")
            
    except Exception as e:
        print(f"  Error processing sequence {h5_path.name}: {e}")

def export_video(demos_folder: Path):
    """Export processed videos and downsampled motion data from demo folders.
    
    Args:
        demos_folder: Path to the folder containing demonstration recordings
    """
    # Create output directory
    processed_dir = demos_folder / "processed"
    processed_dir.mkdir(exist_ok=True)
    
    # Load the master video 
    # Read main video peak frame and peak time
    peak_frame_path = demos_folder / "video_peak_frame.txt"
    if peak_frame_path.exists():
        with open(peak_frame_path, "r") as f:
            video_peak_frame = int(f.read())
    else:
        print(f"Warning: {peak_frame_path} not found")
        video_peak_frame = 0
        
    peak_time_path = demos_folder / "peak_time.txt"
    if peak_time_path.exists():
        with open(peak_time_path, "r") as f:
            peak_time = float(f.read())
    else:
        print(f"Warning: {peak_time_path} not found")
        peak_time = 0.0

    """Get the insta360 video name from the insta360_filenames.txt file"""
    video_name_path = demos_folder / "insta360_filenames.txt"
    # the file contains this... ['/DCIM/Camera01/VID_20250321_003911_00_070.insv', '/DCIM/Camera01/LRV_20250321_003911_01_070.lrv']
    # I want a video of the form VID_20250321_003911_00_070.mp4
    with open(video_name_path, 'r') as f:
        content = f.read()
        # Extract the first filename from the list
        insv_name = eval(content)[0]
        # Get just the VID_*.insv part
        base_name = os.path.basename(insv_name)
        # Replace .insv with .mp4
        video_filename = base_name.replace('.insv', '.mp4')
    
    # Read wrist video peak frame
    wrist_peak_frame_path = demos_folder / "wrist_peak_frame.txt"
    if wrist_peak_frame_path.exists():
        with open(wrist_peak_frame_path, "r") as f:
            wrist_peak_frame = int(f.read())
    else:
        print(f"Info: Optional {wrist_peak_frame_path} not found. Wrist video processing will be skipped.")
        wrist_peak_frame = None

    # Read zed video peak frame
    zed_peak_frame_path = demos_folder / "zed_peak_frame.txt"
    if zed_peak_frame_path.exists():
        with open(zed_peak_frame_path, "r") as f:
            zed_peak_frame = int(f.read())
    else:
        print(f"Info: Optional {zed_peak_frame_path} not found. ZED video processing will be skipped.")
        zed_peak_frame = None
        
    # Process each sequence folder
    sequences_dir = demos_folder / "sequences"
    if not sequences_dir.exists() or not sequences_dir.is_dir():
        print(f"No sequences directory found at {sequences_dir}")
        return
    
    # Use a thread pool to process sequences in parallel
    # Limit the number of concurrent tasks to avoid overloading the system
    max_workers = min(torch.cuda.device_count()*2, 32)
    print(f"Processing with up to {max_workers} parallel workers")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all sequence processing tasks to the thread pool
        futures = []
        for i, h5_path in enumerate(sequences_dir.iterdir()):
            future = executor.submit(
                process_sequence, 
                h5_path, 
                demos_folder, 
                processed_dir, 
                video_filename, 
                video_peak_frame, 
                peak_time,
                wrist_peak_frame,
                zed_peak_frame,
                i % torch.cuda.device_count()
            )
            futures.append((h5_path, future))
        
        # Wait for all tasks to complete
        for h5_path, future in futures:
            try:
                future.result()
            except Exception as e:
                print(f"Error processing {h5_path}: {e}")
    
    print(f"Processed {demos_folder}")

def main(folder_path: str = None):
    if folder_path is None:
        print("No folder path provided")
        return
    import glob
    for seq_name in glob.glob(folder_path):
        print(f"Processing {seq_name}")
        export_video(Path(seq_name))

if __name__ == "__main__":
    tyro.cli(main)