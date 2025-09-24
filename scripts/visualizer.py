import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel
from PyQt5.QtGui import QImage, QPixmap, QKeyEvent
from PyQt5.QtCore import Qt, QTimer
from eye.zmq_tools import Subscriber, Publisher
import cv2
import numpy as np


class EyeFrameVisualizer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.subscriber = Subscriber("ipc:///tmp/eye_frame")
        self.subscriber.on_recv_bytes(self.update_frame)
        self.click_publisher = Publisher("ipc:///tmp/eye_click")

    def initUI(self):
        self.setWindowTitle("Eye Frame Visualizer")
        self.image_label = QLabel(self)
        self.setCentralWidget(self.image_label)

    def update_frame(self, frame_bytes):
        frame = np.frombuffer(frame_bytes, dtype=np.uint8).reshape(1200, 1920, 3)
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_image = QImage(
            frame.data, width, height, bytes_per_line, QImage.Format_RGB888
        ).rgbSwapped()

        # Draw green dot in the middle
        center_x, center_y = 1920 // 2, 1200 // 2
        for i in range(-2, 2):
            for j in range(-2, 2):
                q_image.setPixelColor(center_x + i, center_y + j, Qt.green)

        pixmap = QPixmap.fromImage(q_image)
        self.image_label.setPixmap(pixmap)
        self.resize(pixmap.width(), pixmap.height())

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key_Q or event.key() == Qt.Key_Escape:
            self.close()

    def mousePressEvent(self, event):
        x = event.x()
        y = event.y()
        self.click_publisher.send_bytes(np.array([x, y], dtype=np.int32).tobytes())


def main():
    app = QApplication(sys.argv)
    visualizer = EyeFrameVisualizer()
    visualizer.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
