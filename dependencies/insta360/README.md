# Insta360 Python

Python bindings for the Insta360 Camera SDK.

## Installation

```bash
pip install insta360
```

## Usage

```python
from insta360 import Camera, VideoResolution

# Create a camera instance
camera = Camera()

# Open the first available camera
if camera.discover_and_open():
    print("Camera opened successfully")
    
    # Start recording
    if camera.start_recording(resolution=VideoResolution.RES_2880_2880P30):
        print("Recording started")
        
        # Wait for some time
        import time
        time.sleep(5)
        
        # Stop recording
        file_urls = camera.stop_recording()
        print(f"Recording stopped, files: {file_urls}")
        
        # Download the recorded file
        if file_urls:
            camera.download_file(file_urls[0], "recording.mp4", 
                                lambda current, total: print(f"Download progress: {current}/{total}"))
            print("File downloaded")
    
    # Close the camera connection
    camera.close()
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## USB Permission Setup for Insta360 Cameras

When working with Insta360 cameras on Linux systems, you might encounter permission issues that prevent accessing the camera without sudo privileges. Below are instructions to set up proper USB device permissions using udev rules.

### Setting up udev rules

1. **Identify your camera's USB vendor and product IDs**

   Connect your Insta360 camera and run:
   ```bash
   lsusb | grep -i insta360
   ```
   
   You should see output similar to:
   ```
   Bus 001 Device 007: ID 2e1a:0001 Insta360 ONE R
   ```
   
   The format is `ID [vendor_id]:[product_id]` (in this example, vendor_id is `2e1a`)

2. **Create a udev rules file**

   Create a new file in the `/etc/udev/rules.d/` directory:
   ```bash
   sudo nano /etc/udev/rules.d/99-insta360.rules
   ```

3. **Add the following rule to the file**

   ```
   # Insta360 camera udev rules
   SUBSYSTEM=="usb", ATTRS{idVendor}=="2e1a", ATTRS{idProduct}=="*", MODE="0666", GROUP="plugdev"
   ```
   
   Make sure to replace `2e1a` with your camera's vendor ID if it's different.

4. **Save the file and exit the editor**

   In nano, press `Ctrl+O` to save and `Ctrl+X` to exit.

5. **Reload the udev rules**

   ```bash
   sudo udevadm control --reload-rules
   sudo udevadm trigger
   ```

6. **Add your user to the plugdev group**

   ```bash
   sudo usermod -a -G plugdev $USER
   ```

7. **Log out and log back in**

   This is necessary for the group changes to take effect.

8. **Test the camera access**

   Disconnect and reconnect your camera, then try running your application without sudo.

### Troubleshooting

If you're still experiencing issues after setting up the udev rules:

1. Verify the rules are loaded:
   ```bash
   udevadm test $(udevadm info -q path -n /dev/bus/usb/XXX/YYY)
   ```
   (Replace XXX/YYY with the bus and device numbers from the lsusb output)

2. Check if your user is in the plugdev group:
   ```bash
   groups $USER
   ```

3. Make sure you've logged out and logged back in after adding your user to the plugdev group.

4. Some systems may use a different group like "video" instead of "plugdev". If issues persist, try:
   ```
   SUBSYSTEM=="usb", ATTRS{idVendor}=="2e1a", ATTRS{idProduct}=="*", MODE="0666", GROUP="video"
   ```
   And add your user to the video group:
   ```bash
   sudo usermod -a -G video $USER
   ```