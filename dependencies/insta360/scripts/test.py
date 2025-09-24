from insta360 import Camera, VideoResolution
import time
import os

def main():
    # Create a camera instance
    camera = Camera()
    
    # Discover and open the camera
    if camera.discover_and_open():
        print("Camera opened successfully")
        
        # Get the serial number of the camera
        serial_number = camera.get_serial_number()
        print(f"Camera serial number: {serial_number}")
        # Start recording with default resolution
        if camera.start_recording(resolution=VideoResolution.RES_2880_2880P30, bitrate=20 * 1024 * 1024):
            print("Recording started successfully")
            for i in range(100):
                print(i)
                time.sleep(.01)
            
            # Wait for 2 seconds
            print("Recording for 2 seconds...")
            time.sleep(2)

            
            # Stop recording and get the file URLs
            file_urls = camera.stop_recording()
            print(file_urls)
        else:
            print("Failed to start recording")
        
        # Close the camera connection
        # camera.close()
        print("Camera connection closed")
    else:
        print("Failed to open camera")

if __name__ == "__main__":
    main()

