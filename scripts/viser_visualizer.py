import viser
import numpy as np
from eye.zmq_tools import Subscriber, Publisher
import time


class EyeFrameVisualizer:
    def __init__(self):
        self.server = viser.ViserServer()
        self.subscriber = Subscriber("ipc:///tmp/eye_frame")
        self.click_publisher = Publisher("ipc:///tmp/eye_click")

        # Set up the frame callback
        self.subscriber.on_recv_bytes(self.update_frame)

        # Set up click handling for each client
        @self.server.on_client_connect
        def _(client: viser.ClientHandle) -> None:
            @client.scene.on_pointer_event(event_type="click")
            def handle_click(event: viser.ScenePointerEvent) -> None:
                # Get screen position from the click event
                screen_pos = event.screen_pos[0]  # Get first position tuple
                print(f"Click at {screen_pos}")
                # Convert normalized coordinates (0-1) to pixel coordinates
                x = int(screen_pos[0] * 1920)
                y = int(screen_pos[1] * 1200)

                # Send click coordinates through ZMQ
                self.click_publisher.send_bytes(
                    np.array([x, y], dtype=np.int32).tobytes()
                )

    def update_frame(self, frame_bytes):
        # Convert bytes to numpy array
        frame = np.frombuffer(frame_bytes, dtype=np.uint8).reshape(1200, 1920, 3)

        # Update the background image
        self.server.scene.set_background_image(frame, format="jpeg", jpeg_quality=60)

    def run(self):
        # Keep the server running
        try:
            while True:
                time.sleep(0.01)  # Small sleep to prevent CPU overload
        except KeyboardInterrupt:
            print("Shutting down...")


def main():
    visualizer = EyeFrameVisualizer()
    visualizer.run()


if __name__ == "__main__":
    main()
