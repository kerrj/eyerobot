import threading
from eye.eyeball import Eyeball
from eye.zmq_tools import Subscriber
import numpy as np
import time
import torch


class EyeController:
    def __init__(self):
        self.eye = Eyeball()
        self.frame_subscriber = Subscriber("ipc:///tmp/eye_frame")
        self.click_subscriber = Subscriber("ipc:///tmp/eye_click")
        self.last_available_frame = None
        self.frame_lock = threading.Lock()
        self.is_running = True
        self.click = None
        self.count = 0
        self.frame_subscriber.on_recv_bytes(self.on_frame)
        self.click_subscriber.on_recv_bytes(self.on_click)

    def on_frame(self, frame):
        buf = np.frombuffer(frame, dtype=np.uint8).reshape(1200, 1920, 3)
        with self.frame_lock:
            self.last_available_frame = buf

    def on_click(self, click):
        self.click = np.frombuffer(click, dtype=np.int32)

    def loop_forever(self):
        self.start_time = time.time()
        self.ts = []
        time.sleep(0.1)
        while self.is_running:
            with self.frame_lock:
                current_frame = self.last_available_frame
                self.last_available_frame = None

            if self.click is not None and current_frame is not None:
                x, y = self.click
                print("Click at", x, y)
                self.eye.saccade(current_frame, x, y)
                self.click = None
            self.eye.analytic_control_step(current_frame)
            self.count += 1

    def stop(self):
        elapsed_time = time.time() - self.start_time
        self.is_running = False
        self.eye.teardown()
        print(f"Average FPS: {self.count / elapsed_time:.2f}")


if __name__ == "__main__":
    import signal
    import sys

    def signal_handler(signum, frame):
        print("\nCtrl+C received. Stopping the controller...")
        controller.stop()
        sys.exit(0)

    controller = EyeController()

    signal.signal(signal.SIGINT, signal_handler)

    print("Starting EyeController. Press Ctrl+C to stop.")
    try:
        controller.loop_forever()
    except Exception as e:
        print(f"Error in main loop: {e}")
    finally:
        controller.stop()
