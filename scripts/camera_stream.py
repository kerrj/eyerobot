import time
import signal
import sys
from eye.zmq_tools import Publisher
from eye.camera import OCVCamera
import cv2

class FramePublisher:
    def __init__(self):
        self.count = 0
        self.frame_publisher = Publisher("ipc:///tmp/eye_frame")
        self.is_running = True
        self.camera = OCVCamera(width=1920, height=1200, fps=100, exposure=None)#300 for night
        self.camera.open_camera()

    def run(self):
        self.start_time = time.time()
        while self.is_running:
            try:
                frame = self.camera.read_frame(blocking=True)
                #rotate BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.frame_publisher.send_bytes(frame)
                self.count += 1
            except Exception as e:
                print(f"Error in frame_loop: {e}")
                self.stop()

    def stop(self):
        self.is_running = False
        elapsed_time = time.time() - self.start_time
        print(f"\nProcessed {self.count} frames in {elapsed_time:.2f} seconds")
        print(f"Average FPS: {self.count / elapsed_time:.2f}")


def signal_handler(signum, frame):
    print("\nCtrl+C received. Stopping the publisher...")
    publisher.stop()
    sys.exit(0)


if __name__ == "__main__":
    publisher = FramePublisher()

    signal.signal(signal.SIGINT, signal_handler)

    print("Starting FramePublisher. Press Ctrl+C to stop.")
    publisher.run()
