import sys
import pyzed.sl as sl
import numpy as np
import cv2
from pathlib import Path
import argparse
import os
import ffmpeg

def progress_bar(percent_done, bar_length=50):
    #Display a progress bar
    done_length = int(bar_length * percent_done / 100)
    bar = '=' * done_length + '-' * (bar_length - done_length)
    sys.stdout.write('[%s] %i%s\r' % (bar, percent_done, '%'))
    sys.stdout.flush()


def main():
    # Get input parameters
    svo_input_path = opt.input_svo_file
    output_dir = opt.output_path_dir
    mp4_output_path = opt.output_mp4_file
    output_as_video = True    
    if opt.mode != 0:
        output_as_video = False

    if not output_as_video and not os.path.isdir(output_dir):
        sys.stdout.write("Input directory doesn't exist. Check permissions or create it.\n",
                         output_dir, "\n")
        exit()

    # Specify SVO path parameter
    init_params = sl.InitParameters()
    init_params.set_from_svo_file(svo_input_path)
    init_params.svo_real_time_mode = False  # Don't convert in realtime
    init_params.coordinate_units = sl.UNIT.MILLIMETER  # Use milliliter units (for depth measurements)

    # Create ZED objects
    zed = sl.Camera()

    # Open the SVO file specified as a parameter
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        sys.stdout.write(repr(err))
        zed.close()
        exit()
    
    # Get image size
    image_size = zed.get_camera_information().camera_configuration.resolution
    width = image_size.width
    height = image_size.height
    target_fps = max(zed.get_camera_information().camera_configuration.fps, 25) # Use the same FPS for VideoWriter and time calculation
    target_frame_duration_us = 1_000_000 / target_fps
    
    # Prepare image container
    left_image = sl.Mat()

    # Prepare ffmpeg process if outputting video
    ffmpeg_process = None
    if output_as_video:
        try:
            # Configure ffmpeg input stream (reading raw RGB frames from stdin)
            input_stream = ffmpeg.input(
                'pipe:',
                format='rawvideo',
                pix_fmt='rgb24', # Input format from OpenCV
                s=f'{width}x{height}',
                r=target_fps
            )
            # Configure ffmpeg output stream
            # Use h264_nvenc for NVIDIA GPU acceleration.
            # Alternatives: h264_vaapi (Intel/AMD Linux), h264_videotoolbox (macOS), libx264 (CPU)
            # You might need to install ffmpeg with specific compilation flags (e.g., --enable-nvenc)
            output_stream = ffmpeg.output(
                input_stream,
                mp4_output_path,
                vcodec='h264_nvenc', # Try changing if NVENC is not available
                pix_fmt='yuv420p',  # Common pixel format for H.264 MP4
                r=target_fps
            ).overwrite_output()
            # Start the ffmpeg process
            ffmpeg_process = ffmpeg.run_async(output_stream, pipe_stdin=True)
            sys.stdout.write(f"Started ffmpeg process for {mp4_output_path} using h264_nvenc.\n")

        except ffmpeg.Error as e:
            sys.stdout.write(f"ffmpeg error: {e.stderr.decode()}\n")
            sys.stdout.write("Failed to start ffmpeg. Check if ffmpeg is installed, in PATH, "
                             "and compiled with the necessary codecs (e.g., h264_nvenc).\n")
            zed.close()
            exit()

    rt_param = sl.RuntimeParameters()

    # Start SVO conversion to MP4/SEQUENCE
    sys.stdout.write("Converting SVO... Use Ctrl-C to interrupt conversion.\n")

    nb_frames = zed.get_svo_number_of_frames()
    
    # Variables for timestamp-based frame padding
    first_timestamp_us = None
    previous_frame_rgb = None
    output_frame_count = 0

    while True:
        try:
            err = zed.grab(rt_param)
            if err == sl.ERROR_CODE.SUCCESS:
                svo_position = zed.get_svo_position()
                # Retrieve left image only
                zed.retrieve_image(left_image, sl.VIEW.LEFT)
                current_timestamp_us = left_image.timestamp.get_microseconds()

                if output_as_video and ffmpeg_process:
                    # Get the left image data and convert from RGBA to RGB
                    left_data = left_image.get_data()
                    current_frame_rgb = cv2.cvtColor(left_data, cv2.COLOR_BGRA2RGB)

                    if first_timestamp_us is None:
                        # First frame initialization
                        first_timestamp_us = current_timestamp_us
                        # Write frame bytes to ffmpeg's stdin
                        ffmpeg_process.stdin.write(current_frame_rgb.tobytes())
                        output_frame_count = 1
                        previous_frame_rgb = current_frame_rgb
                    else:
                        # Calculate expected frames vs actual frames based on time elapsed
                        elapsed_us = current_timestamp_us - first_timestamp_us
                        # +1 because frame count is 1-based relative to the first frame
                        expected_total_frames = round(elapsed_us / target_frame_duration_us) + 1 
                        
                        # Calculate padding needed. Should be at least 0.
                        num_padding_frames = max(0, expected_total_frames - output_frame_count - 1) 

                        if num_padding_frames > 0 and previous_frame_rgb is not None:
                            # Write the previous frame to pad the time gap
                            for _ in range(num_padding_frames):
                                print("Padding frame") # Optional: Keep for debugging
                                ffmpeg_process.stdin.write(previous_frame_rgb.tobytes())
                            output_frame_count += num_padding_frames
                        
                        # Write the current frame
                        ffmpeg_process.stdin.write(current_frame_rgb.tobytes())
                        output_frame_count += 1
                        previous_frame_rgb = current_frame_rgb # Store for potential future padding
                elif not output_as_video:
                    # Generate file name for left image
                    filename = output_dir +"/"+ ("left%s.png" % str(svo_position).zfill(6))
                    # Save left image
                    cv2.imwrite(str(filename), left_image.get_data())

                # Display progress
                progress_bar((svo_position + 1) / nb_frames * 100, 30)
            elif err == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
                progress_bar(100 , 30)
                sys.stdout.write("\nSVO end has been reached. Exiting now.\n")
                break
        except KeyboardInterrupt:
             sys.stdout.write("\nConversion interrupted by user (Ctrl-C).\n")
             break
        except Exception as e:
             sys.stdout.write(f"\nAn error occurred during processing: {e}\n")
             break # Or handle more gracefully

    if output_as_video and ffmpeg_process:
        # Close the ffmpeg stdin pipe and wait for the process to finish
        try:
            ffmpeg_process.stdin.close()
            ffmpeg_process.wait()
            sys.stdout.write("\nffmpeg process finished.\n")
        except Exception as e:
            sys.stdout.write(f"\nError closing ffmpeg process: {e}\n")

    zed.close()
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--mode', type = int, required=True, help= " Mode 0 is to export LEFT view as MP4. \n Mode 1 is to export LEFT view as image sequence.")
    parser.add_argument('--input_svo_file', type=str, required=True, help='Path to the .svo file')
    parser.add_argument('--output_mp4_file', type=str, help='Path to the output .mp4 file, if mode is 0', default = '') 
    parser.add_argument('--output_path_dir', type = str, help = 'Path to a directory, where .png will be written, if mode is 1', default = '')
    opt = parser.parse_args()
    if opt.mode > 1 or opt.mode < 0 :
        print("Mode should be 0 or 1. \n Mode 0 is to export LEFT view as MP4. \n Mode 1 is to export LEFT view as image sequence.")
        exit()
    if not opt.input_svo_file.endswith(".svo") and not opt.input_svo_file.endswith(".svo2"): 
        print("--input_svo_file parameter should be a .svo file but is not : ",opt.input_svo_file,"Exit program.")
        exit()
    if not os.path.isfile(opt.input_svo_file):
        print("--input_svo_file parameter should be an existing file but is not : ",opt.input_svo_file,"Exit program.")
        exit()
    if opt.mode == 0 and len(opt.output_mp4_file)==0:
        print("In mode 0, output_mp4_file parameter needs to be specified.")
        exit()
    if opt.mode == 0 and not opt.output_mp4_file.endswith(".mp4"):
        print("--output_mp4_file parameter should be a .mp4 file but is not : ",opt.output_mp4_file,"Exit program.")
        exit()
    if opt.mode == 1 and len(opt.output_path_dir)==0:
        print("In mode 1, output_path_dir parameter needs to be specified.")
        exit()
    if opt.mode == 1 and not os.path.isdir(opt.output_path_dir):
        print("--output_path_dir parameter should be an existing folder but is not : ",opt.output_path_dir,"Exit program.")
        exit()
    main() 
