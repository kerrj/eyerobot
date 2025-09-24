import time
from gello.robots.robotiq_gripper import RobotiqGripper
import rtde_receive
import numpy as np
import matplotlib
matplotlib.use('Agg') # Use a non-interactive backend
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import viser
from insta360 import Camera, VideoResolution
import os
import h5py
from pynput import keyboard
import threading
import cv2
import subprocess
import signal
import tyro
import pyzed.sl as sl
import sys
import signal
import time
import threading # Added threading


WRIST_CAM_DEVICE = "/dev/video2" 
# Global flags accessible by both threads
exit_signal_received = False
frames_recorded = 0 # Make global for thread access
record_thread = None

def find_main_peak(timestamps, joint_data, folder_name):
    print("Data collection complete. Analyzing and plotting...")
    
    # Convert to numpy arrays for easier handling
    joint_data = np.array(joint_data)
    timestamps = np.array(timestamps)
    joint5_data = joint_data[:, 4]
    
    # Find peaks (local maxima)
    peaks, _ = find_peaks(joint5_data)
    # Find valleys (local minima) by inverting the signal
    valleys, _ = find_peaks(-joint5_data)
    
    # Find the most extreme outlier among peaks and valleys
    peak_values = joint5_data[peaks]
    valley_values = joint5_data[valleys]
    
    # Calculate how far each peak/valley is from the mean
    mean_value = np.mean(joint5_data)
    peak_distances = np.abs(peak_values - mean_value)
    valley_distances = np.abs(valley_values - mean_value)
    
    # Find the most extreme outlier
    if len(peak_distances) > 0 and len(valley_distances) > 0:
        max_peak_dist = np.max(peak_distances)
        max_valley_dist = np.max(valley_distances)
        
        if max_peak_dist > max_valley_dist:
            extreme_idx = peaks[np.argmax(peak_distances)]
            extreme_type = "peak"
        else:
            extreme_idx = valleys[np.argmax(valley_distances)]
            extreme_type = "valley"
    elif len(peak_distances) > 0:
        extreme_idx = peaks[np.argmax(peak_distances)]
        extreme_type = "peak"
    elif len(valley_distances) > 0:
        extreme_idx = valleys[np.argmax(valley_distances)]
        extreme_type = "valley"
    else:
        extreme_idx = None
        extreme_type = None

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(timestamps, joint5_data, 'b-', label='Joint 5')
    
    # Plot all peaks and valleys
    plt.plot(timestamps[peaks], joint5_data[peaks], 'r*', label='Peaks')
    plt.plot(timestamps[valleys], joint5_data[valleys], 'g*', label='Valleys')
    
    # Highlight the most extreme outlier
    if extreme_idx is not None:
        plt.plot(timestamps[extreme_idx], joint5_data[extreme_idx], 'k*', 
                markersize=15, label=f'Most extreme {extreme_type}')
        
    plt.xlabel('Time (seconds)')
    plt.ylabel('Joint 5 Position (radians)')
    plt.title('Joint 5 Position over Time with Peak Detection')
    plt.grid(True)
    plt.legend()
    plt.savefig(f"{folder_name}/joint5_position_over_time.png")
    return timestamps[extreme_idx] if extreme_idx is not None else None

def run_v4l2_ctl(device, control, value):
    """Helper function to run v4l2-ctl commands."""
    command = ['v4l2-ctl', '-d', device, '-c', f'{control}={value}']
    print(f"Running command: {' '.join(command)}")
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        print(f"Successfully set {control}={value}")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error setting {control}={value}: {e}")
        print(f"Stderr: {e.stderr}")
    except FileNotFoundError:
        print("Error: v4l2-ctl command not found. Is it installed and in PATH?")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def recording_thread_func(zed, runtime_parameters):
    """
    This function runs in a separate thread and handles the ZED camera frame grabbing.
    """
    global frames_recorded, exit_signal_received
    print("Recording thread started.")
    # Reset frame count for this session if needed, though it's global now
    # frames_recorded = 0

    while not exit_signal_received:
        # Grab an image - recording happens automatically by the SDK
        grab_status = zed.grab(runtime_parameters)
        if grab_status == sl.ERROR_CODE.SUCCESS:
            frames_recorded += 1
            # Optional: Print progress infrequently
            # if frames_recorded % 100 == 0:
            #     print(f"Frames recorded: {frames_recorded}", end='\r')
        elif grab_status == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
            print("Recording thread: End of stream reached.")
            exit_signal_received = True # Signal main thread too
            break
        else:
            print(f"Recording thread: Error during grab: {grab_status}")
            exit_signal_received = True # Signal main thread too
            break
        # Minimal sleep to potentially yield CPU, grab() likely blocks anyway
        time.sleep(0.001)

    print("Recording thread finished.")

def main(
    collect_360: bool,
    collect_wrist: bool,
    collect_zed: bool,
    wrist_cam_manual_exposure: bool = False, # Default to auto exposure (Aperture Priority)
    wrist_cam_exposure_time_absolute: int = 156, # Default exposure time if manual
    wrist_cam_auto_white_balance: bool = True, # Default to auto white balance
):
    """
    Main data collection script.

    Args:
        collect_360: Whether to collect data from the 360 camera.
        collect_wrist: Whether to collect data from the wrist camera.
        collect_zed: Whether to collect data from the ZED camera.
        manual_exposure: Set exposure manually. Defaults to Aperture Priority Mode.
        exposure_time_absolute: Absolute exposure time (used only if manual_exposure is True). Range typically 3-2047.
        auto_white_balance: Enable automatic white balance. Defaults to True.
    """
    global exit_signal_received, frames_recorded,record_thread
    viser_server = viser.ViserServer()
    # Don't initialize camera here yet
    # if collect_wrist:
    #     from eye.camera import OCVCamera
    #     wrist_cam = OCVCamera(1280, 720, 30)
    #     # Don't open camera here yet
    #     # wrist_cam.open_camera()
    # Configuration
    robot_ip = "172.22.22.2"
    poll_rate = 200  # Hz
    poll_interval = 1.0 / poll_rate

    # Initialize data storage
    joint_data = []
    timestamps = []
    recording_demo = False
    demo_joint_data = []
    demo_gripper_data = []
    demo_timestamps = []


    # Initialize interfaces
    r_inter = rtde_receive.RTDEReceiveInterface(robot_ip)
    gripper = RobotiqGripper()
    gripper.connect(hostname=robot_ip, port=63352)
    print("Connected to robot and gripper")
    demo_folder = "./data/demos"
    folder_name = viser_server.gui.add_text("Folder Name", initial_value=time.strftime("%Y%m%d_%H%M%S"))
    num_demos_viser_scene = viser_server.gui.add_text("Number of Demos", initial_value="0",disabled=True)
    peak_time = None
    global_folder_name = None
    svo_output_filename = None
    # Camera/ffmpeg related variables
    wrist_cam = None
    ffmpeg_process = None # Handle for the ffmpeg subprocess


    # Initialize the insta360 camera object here
    camera = Camera()
    
    # Discover and open the camera
    if camera.discover_and_open():
        print("Camera opened successfully")
        
        # Get the serial number of the camera
        serial_number = camera.get_serial_number()
        print(f"Camera serial number: {serial_number}")
    camera.stop_recording()

    if collect_zed:
            # --- Camera Setup ---
        print("Initializing ZED camera...")
        zed = sl.Camera()
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.HD1080
        init_params.camera_fps = 30
        # init_params.set_from_serial_number(12345)

        err = zed.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            print(f"Failed to open ZED camera: {err}")
            zed.close()
            sys.exit(1)
        print("ZED camera opened successfully.")

        

    start_system = viser_server.gui.add_button("Start")
    stop_calibration = viser_server.gui.add_button("Stop Calibration", visible=False)

    
    run_calib = True


    @start_system.on_click
    def start_system_callback(_):
        # Make variables from outer scope available
        nonlocal global_folder_name, wrist_cam, ffmpeg_process, run_calib,svo_output_filename
        global record_thread

        folder_name.disabled = True
        global_folder_name = f"{demo_folder}/{folder_name.value}"
        os.makedirs(global_folder_name, exist_ok=True)
        start_system.visible = False
        stop_calibration.visible = True

        # Initialize OCVCamera primarily to get settings
        # We won't use its read_frame in a loop for saving anymore
        if collect_wrist:
            # --- Check Camera Availability ---
            if not os.path.exists(WRIST_CAM_DEVICE):
                print(f"Error: Wrist camera device {WRIST_CAM_DEVICE} not found.")
                viser_server.gui.add_markdown(f"**ERROR:** Wrist camera device `{WRIST_CAM_DEVICE}` not found.")
                # Optional: Stop the script or raise an error
                return # Stop this callback

            print(f"Checking access to wrist camera: {WRIST_CAM_DEVICE}")
            temp_cap = cv2.VideoCapture(WRIST_CAM_DEVICE)
            if not temp_cap.isOpened():
                print(f"Error: Could not open wrist camera {WRIST_CAM_DEVICE}. It might be busy.")
                print(f"Suggestion: Try running 'sudo fuser -k {WRIST_CAM_DEVICE}' in your terminal.")
                viser_server.gui.add_markdown(f"**ERROR:** Could not open wrist camera `{WRIST_CAM_DEVICE}`. It might be busy. Suggestion: `sudo fuser -k {WRIST_CAM_DEVICE}`")
                temp_cap.release()
                return # Stop this callback
            else:
                # Attempt to read a frame to be more sure
                ret, _ = temp_cap.read()
                temp_cap.release() # <<< IMPORTANT: Release the device immediately
                if not ret:
                     print(f"Error: Could not read frame from wrist camera {WRIST_CAM_DEVICE}. It might be busy or malfunctioning.")
                     print(f"Suggestion: Try running 'sudo fuser -k {WRIST_CAM_DEVICE}'.")
                     viser_server.gui.add_markdown(f"**ERROR:** Could not read frame from `{WRIST_CAM_DEVICE}`. Suggestion: `sudo fuser -k {WRIST_CAM_DEVICE}`")
                     return # Stop this callback
                print(f"Wrist camera {WRIST_CAM_DEVICE} accessed successfully.")
            # --- End Camera Availability Check ---

            try:
                from eye.camera import OCVCamera # Import here or ensure it's globally available
                cam_width, cam_height, cam_fps = 1280, 720, 30

                # --- Set Camera Controls using v4l2-ctl BEFORE starting ffmpeg ---
                print("Setting camera controls via v4l2-ctl...")
                # Set focus: Turn off continuous auto focus, set absolute focus to 0 (infinity)
                # Note: Confirm 'focus_automatic_continuous' and 'focus_absolute' are the correct controls
                # and '0' is the correct value for infinity for YOUR camera using `v4l2-ctl -d /dev/videoX --list-ctrls`
                run_v4l2_ctl(WRIST_CAM_DEVICE, 'focus_automatic_continuous', 0)
                run_v4l2_ctl(WRIST_CAM_DEVICE, 'focus_absolute', 0)

                # Set White Balance
                run_v4l2_ctl(WRIST_CAM_DEVICE, 'white_balance_automatic', 1 if wrist_cam_auto_white_balance else 0)

                # Set Exposure
                if wrist_cam_manual_exposure:
                    # Set to Manual Exposure Mode
                    run_v4l2_ctl(WRIST_CAM_DEVICE, 'auto_exposure', 0) # 0 = Manual Mode
                    # Set Absolute Exposure Time
                    run_v4l2_ctl(WRIST_CAM_DEVICE, 'exposure_time_absolute', wrist_cam_exposure_time_absolute)
                else:
                    # Set to Aperture Priority Mode (common auto mode)
                    run_v4l2_ctl(WRIST_CAM_DEVICE, 'auto_exposure', 3) # 3 = Aperture Priority Mode
                print("Finished setting camera controls.")
                # --- End Camera Control Setting ---


                # --- Start ffmpeg recording ---
                # Use .mp4 container for H.265
                video_path = f"{global_folder_name}/wrist_video.mp4"
                print(f"Attempting to start ffmpeg recording (H.265 encoding) for device {WRIST_CAM_DEVICE} to {video_path}")

                ffmpeg_command = [
                    'ffmpeg',
                    '-f', 'v4l2',             # Input format
                    '-input_format', 'mjpeg', # Specify input pixel format
                    '-video_size', f'{cam_width}x{cam_height}', # Request specific size
                    '-framerate', str(cam_fps),   # Request specific framerate
                    '-i', WRIST_CAM_DEVICE,   # Input device

                    # --- Encoding Options ---
                    '-c:v', 'hevc_nvenc',     # Use the NVIDIA HEVC encoder
                    '-pix_fmt', 'yuv420p', # Force common pixel format for compatibility

                    '-r', '30',               # Set output frame rate to exactly 30 fps
                    '-y',                     # Overwrite output file without asking
                    video_path                # Output file path
                ]

                print(f"Running command: {' '.join(ffmpeg_command)}")
                # Start ffmpeg as a subprocess
                # Use preexec_fn=os.setsid to create a new process group,
                # allowing us to send signal to the entire group if needed.
                ffmpeg_process = subprocess.Popen(ffmpeg_command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, preexec_fn=os.setsid)
                print(f"ffmpeg process started with PID: {ffmpeg_process.pid}")
                # You might want to add a small delay and check ffmpeg_process.poll()
                # to see if it exited immediately (indicating an error)
                time.sleep(1) # Give ffmpeg a moment to start
                if ffmpeg_process.poll() is not None:
                     print("Error: ffmpeg process exited unexpectedly. Check stderr:")
                     stderr_output = ffmpeg_process.stderr.read()
                     print(stderr_output)
                     # Handle error - maybe stop calibration?
                     # For now, just print
                     ffmpeg_process = None # Reset process handle
                     # Optional: Clean up OCVCamera if it was opened
                     if wrist_cam is not None: wrist_cam.release_camera()
                else:
                     print("ffmpeg seems to be running.")
                     # Optionally, you could start a thread to monitor ffmpeg's stderr for errors during recording


            except Exception as e:
                print(f"Error initializing OCVCamera or starting ffmpeg: {e}")
                if wrist_cam is not None: wrist_cam.release_camera()
                ffmpeg_process = None # Ensure process handle is None


        # Start 360 camera recording
        if collect_360:
             if camera.start_recording(resolution=VideoResolution.RES_2880_2880P30, bitrate = 20 * 1024 * 1024):
                 print("360 Recording started successfully")
             else:
                 print("Failed to start 360 recording.")

        if collect_zed:
            # --- SVO Recording Setup ---
            svo_output_filename = f"{global_folder_name}/output.svo"
            print(f"Setting up SVO recording to '{svo_output_filename}'...")
            recording_param = sl.RecordingParameters(
                svo_output_filename,
                sl.SVO_COMPRESSION_MODE.H264
            )
            err = zed.enable_recording(recording_param)
            if err != sl.ERROR_CODE.SUCCESS:
                print(f"Failed to enable SVO recording: {err}")
                zed.close()
                sys.exit(1)
            print("SVO recording enabled.")

            # --- Runtime Parameters ---
            runtime_parameters = sl.RuntimeParameters()

            # --- Create and Start Recording Thread ---
            record_thread = threading.Thread(
                target=recording_thread_func, # Function to run in thread
                args=(zed, runtime_parameters) # Arguments for the function
            )
            record_thread.start()
            print("Starting ZED recording thread...")
        # Calibration loop
        while run_calib:
            start_time = time.perf_counter()

            # Get robot joint positions
            robot_joints = r_inter.getActualQ()
            joint_data.append(robot_joints)
            timestamps.append(time.perf_counter())

            # Sleep to maintain the desired polling rate
            elapsed = time.perf_counter() - start_time
            sleep_time = max(0, poll_interval - elapsed)
            time.sleep(sleep_time)

        # Calibration finished
        nonlocal peak_time
        peak_time = find_main_peak(timestamps, joint_data, global_folder_name)
        if peak_time is not None:
            print(f"Peak time: {peak_time}")
            # save the peak time to a file
            with open(f"{global_folder_name}/peak_time.txt", "w") as f:
                f.write(f"{peak_time}")
        else:
             print("Could not determine peak time.")

    def start_demo_callback(_):
        nonlocal recording_demo, demo_joint_data, demo_timestamps, demo_gripper_data, num_demos_viser_scene
        demo_joint_data = []
        demo_timestamps = []
        demo_gripper_data = []
        recording_demo = True
        num_demos_viser_scene.value = str(int(num_demos_viser_scene.value) + 1)

    lock = threading.Lock()
    def stop_demo_callback(_):
        nonlocal recording_demo, num_demos_viser_scene
        id = num_demos_viser_scene.value
        # dump joint data into a file
        # save as an hdf5 file with a unique timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        os.makedirs(f"{global_folder_name}/sequences", exist_ok=True)
        with lock:
            with h5py.File(f"{global_folder_name}/sequences/{timestamp}_joint_data_{id}.h5", "w") as f:
                f.create_dataset("joint_data", data=demo_joint_data)
                f.create_dataset("timestamps", data=demo_timestamps)
                f.create_dataset("gripper_data", data=demo_gripper_data)
        recording_demo = False
        

    def on_press(key):
        nonlocal recording_demo
        try:
            if key.char == 'b':  # Footpedal press
                if not recording_demo:
                    start_demo_callback(None)
                else:
                    stop_demo_callback(None)
        except AttributeError:
            pass

    def on_release(key):
        pass

    # Initialize keyboard listener
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    @stop_calibration.on_click
    def stop_calibration_callback(_):
        nonlocal run_calib
        run_calib = False
        stop_calibration.visible = False
        stop_calibration.disabled = True
        listener.start()


    try:
        while True:
            # Main loop now mostly handles demo recording logic and background updates
            if recording_demo:
                with lock:
                    demo_timestamps.append(time.perf_counter())
                    robot_joints = r_inter.getActualQ()
                    gripper_data = gripper.get_current_position()
                    demo_joint_data.append(robot_joints)
                    demo_gripper_data.append(gripper_data)
                viser_server.scene.set_background_image(np.array([0,255,0])[None,None,:])
            else:
                viser_server.scene.set_background_image(np.array([255,0,0])[None,None,:])
            # Add a small sleep to prevent this loop from consuming 100% CPU
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("\nKeyboard interrupt detected. Stopping...")
        exit_signal_received = True
        if collect_zed:
            record_thread.join()
            # --- Cleanup (happens after thread has finished) ---
            # No need for try/finally here as join() ensures thread completion
            # unless an unhandled exception occurs *before* join(), which is unlikely here.
            # The finally block in the original code was more about the while loop.
            if zed.is_opened():
                print("\nDisabling SVO recording...")
                zed.disable_recording()
                print("Closing ZED camera...")
                zed.close()
                print(f"Recording stopped. SVO file saved as '{svo_output_filename}'.")
                print(f"Total frames recorded: {frames_recorded}") # Access global count
            else:
                # Should only happen if camera failed to open
                print("Camera was not open or already closed before cleanup.")
        

        # --- Stop ffmpeg Process First ---
        if ffmpeg_process and ffmpeg_process.poll() is None: # Check if it's running
            print("Stopping ffmpeg process gracefully...")
            try:
                # Send SIGINT (Ctrl+C) to the process group to allow ffmpeg to finalize the file
                os.killpg(os.getpgid(ffmpeg_process.pid), signal.SIGINT)
                # Wait for ffmpeg to terminate
                ffmpeg_process.wait(timeout=5.0) # Add a timeout
                print("ffmpeg process stopped.")
            except ProcessLookupError:
                 print("ffmpeg process already terminated.")
            except subprocess.TimeoutExpired:
                print("ffmpeg did not terminate gracefully after 5s, killing...")
                # Force kill if it doesn't respond
                os.killpg(os.getpgid(ffmpeg_process.pid), signal.SIGKILL)
                ffmpeg_process.wait() # Wait after killing
                print("ffmpeg process killed.")
            except Exception as e:
                print(f"Error stopping ffmpeg: {e}")
            finally:
                # Ensure stdout/stderr are closed if needed, though Popen usually handles this
                 if ffmpeg_process.stdout: ffmpeg_process.stdout.close()
                 if ffmpeg_process.stderr: ffmpeg_process.stderr.close()
                 if ffmpeg_process.stdin: ffmpeg_process.stdin.close()
        elif ffmpeg_process:
             print("ffmpeg process was not running or already stopped.")


        # Release OCVCamera if it was initialized
        if wrist_cam is not None:
            print("Releasing OCVCamera...")
            wrist_cam.release_camera()
            print("OCVCamera released.")


        # Stop 360 camera recording
        if collect_360:
            print("Stopping 360 camera recording...")
            filenames = camera.stop_recording()
            #save the filenames to a file
            if global_folder_name: # Ensure folder name exists
                with open(f"{global_folder_name}/insta360_filenames.txt", "w") as f:
                    f.write(f"{filenames}")
                print(f"360 camera filenames saved to {global_folder_name}/insta360_filenames.txt")
            else:
                print("Could not save 360 filenames, folder name not set.")
            camera.close()
            print("360 Camera stopped and closed successfully.")

        # Stop keyboard listener
        listener.stop()
        print("Keyboard listener stopped.")

        print("Cleanup complete. Exiting.")

if __name__ == "__main__":
    tyro.cli(main)