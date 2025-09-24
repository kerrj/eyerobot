import cv2
import numpy as np
from eye.lkm import LKMotor
import viser.transforms as vtf
import time
import serial
from typing import Tuple, Optional
import threading
import torch


def _compute_residual_and_jacobian(
    x: torch.Tensor,
    y: torch.Tensor,
    xd: torch.Tensor,
    yd: torch.Tensor,
    distortion_params: torch.Tensor,
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Auxiliary function of radial_and_tangential_undistort() that computes residuals and jacobians.
    Adapted from MultiNeRF:
    https://github.com/google-research/multinerf/blob/b02228160d3179300c7d499dca28cb9ca3677f32/internal/camera_utils.py#L427-L474

    Args:
        x: The updated x coordinates.
        y: The updated y coordinates.
        xd: The distorted x coordinates.
        yd: The distorted y coordinates.
        distortion_params: The distortion parameters [k1, k2, p1, p2, k3, k4]. same as opencv

    Returns:
        The residuals (fx, fy) and jacobians (fx_x, fx_y, fy_x, fy_y).
    """

    k1 = distortion_params[..., 0]
    k2 = distortion_params[..., 1]
    k3 = distortion_params[..., 4]
    k4 = distortion_params[..., 5]
    p1 = distortion_params[..., 2]
    p2 = distortion_params[..., 3]

    # let r(x, y) = x^2 + y^2;
    #     d(x, y) = 1 + k1 * r(x, y) + k2 * r(x, y) ^2 + k3 * r(x, y)^3 +
    #                   k4 * r(x, y)^4;
    r = x * x + y * y
    d = 1.0 + r * (k1 + r * (k2 + r * (k3 + r * k4)))

    # The perfect projection is:
    # xd = x * d(x, y) + 2 * p1 * x * y + p2 * (r(x, y) + 2 * x^2);
    # yd = y * d(x, y) + 2 * p2 * x * y + p1 * (r(x, y) + 2 * y^2);
    #
    # Let's define
    #
    # fx(x, y) = x * d(x, y) + 2 * p1 * x * y + p2 * (r(x, y) + 2 * x^2) - xd;
    # fy(x, y) = y * d(x, y) + 2 * p2 * x * y + p1 * (r(x, y) + 2 * y^2) - yd;
    #
    # We are looking for a solution that satisfies
    # fx(x, y) = fy(x, y) = 0;
    fx = d * x + 2 * p1 * x * y + p2 * (r + 2 * x * x) - xd
    fy = d * y + 2 * p2 * x * y + p1 * (r + 2 * y * y) - yd

    # Compute derivative of d over [x, y]
    d_r = k1 + r * (2.0 * k2 + r * (3.0 * k3 + r * 4.0 * k4))
    d_x = 2.0 * x * d_r
    d_y = 2.0 * y * d_r

    # Compute derivative of fx over x and y.
    fx_x = d + d_x * x + 2.0 * p1 * y + 6.0 * p2 * x
    fx_y = d_y * x + 2.0 * p1 * x + 2.0 * p2 * y

    # Compute derivative of fy over x and y.
    fy_x = d_x * y + 2.0 * p2 * y + 2.0 * p1 * x
    fy_y = d + d_y * y + 2.0 * p2 * x + 6.0 * p1 * y

    return fx, fy, fx_x, fx_y, fy_x, fy_y


def radial_and_tangential_undistort(
    coords: torch.Tensor,
    distortion_params: torch.Tensor,
    eps: float = 1e-4,
    max_iterations: int = 20,
) -> torch.Tensor:
    """Computes undistorted coords given opencv distortion parameters.
    Adapted from MultiNeRF
    https://github.com/google-research/multinerf/blob/b02228160d3179300c7d499dca28cb9ca3677f32/internal/camera_utils.py#L477-L509

    Args:
        coords: The distorted coordinates.
        distortion_params: The distortion parameters [k1, k2, p1, p2, k3]. same as opencv
        eps: The epsilon for the convergence.
        max_iterations: The maximum number of iterations to perform.

    Returns:
        The undistorted coordinates.
    """

    # Initialize from the distorted point.
    x = coords[..., 0]
    y = coords[..., 1]

    for _ in range(max_iterations):
        fx, fy, fx_x, fx_y, fy_x, fy_y = _compute_residual_and_jacobian(
            x=x,
            y=y,
            xd=coords[..., 0],
            yd=coords[..., 1],
            distortion_params=distortion_params,
        )
        denominator = fy_x * fx_y - fx_x * fy_y
        x_numerator = fx * fy_y - fy * fx_y
        y_numerator = fy * fx_x - fx * fy_x
        step_x = torch.where(
            torch.abs(denominator) > eps,
            x_numerator / denominator,
            torch.zeros_like(denominator),
        )
        step_y = torch.where(
            torch.abs(denominator) > eps,
            y_numerator / denominator,
            torch.zeros_like(denominator),
        )

        x = x + step_x
        y = y + step_y

    return torch.stack([x, y], dim=-1)


class OCVCamera:
    def __init__(self, width=1280, height=800, fps=90, exposure: Optional[int] = None, video_dev:int = -1):
        self.width = width
        self.height = height
        self.fps = fps
        self.exposure = exposure
        self.cap = None
        self.video_dev = video_dev
        # Below is for ELP 120deg 2.1mm lens
        self.K = np.array(
            [
                [1.06128777e03, 0.00000000e00, width / 2],
                [0.00000000e00, 1.06214523e03, height / 2],
                [0.00000000e00, 0.00000000e00, 1.00000000e00],
            ]
        )

        self.dist_coeffs = np.array(
            [
                [
                    -3.96121855e-01,
                    1.75764171e-01,
                    -4.96420444e-04,
                    -2.67503763e-04,
                    -3.93952186e-02,
                ]
            ]
        )

        self.frame = None
        self.running = False
        self.thread = None
        self.lock = threading.Lock()

    def get_centerpoint(self):
        return self.K[0, 2], self.K[1, 2]

    def open_camera(self):
        self.cap = cv2.VideoCapture(self.video_dev)

        if not self.cap.isOpened():
            raise Exception("Could not open camera")

        # Enable NVIDIA hardware decoding
        self.cap.set(cv2.CAP_PROP_CODEC_PIXEL_FORMAT, cv2.CAP_OPENCV_MJPEG)
        self.cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)

        # Set the FOURCC to MJPEG
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        self.cap.set(cv2.CAP_PROP_FOURCC, fourcc)

        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        # Set exposure mode based on the provided value
        if self.exposure is None:
            # Enable auto exposure
            # Note: Value 1 = auto, 3 = manual on some systems. Try 1 first.
            print("Setting auto exposure.")
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3) # Or 3 depending on the camera/driver
        else:
            # Disable auto exposure and set manual exposure
            # Note: Value 1 = auto, 3 = manual. 0 is often used to disable auto. Test what works.
            # The exposure value range can vary (-13 to 0 is common for UVC cameras).
            print(f"Setting manual exposure to {self.exposure}.")
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1) 
            set_ok = self.cap.set(cv2.CAP_PROP_EXPOSURE, self.exposure)
            if not set_ok:
                print(f"Warning: Failed to set exposure property to {self.exposure}.")

        # Verify hardware acceleration is enabled
        hw_accel = self.cap.get(cv2.CAP_PROP_HW_ACCELERATION)
        print(f"Hardware acceleration status: {hw_accel}")

        actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        actual_exposure_mode = self.cap.get(cv2.CAP_PROP_AUTO_EXPOSURE) # Check if auto exposure is on
        actual_exposure_val = self.cap.get(cv2.CAP_PROP_EXPOSURE) # Read back the exposure value

        print(
            f"Requested resolution: {self.width}x{self.height}, actual resolution: {int(actual_width)}x{int(actual_height)}"
        )
        print(f"Requested FPS: {self.fps}, Actual FPS: {actual_fps}")
        print(f"Actual Auto Exposure Mode: {actual_exposure_mode}")
        print(f"Actual Exposure Value: {actual_exposure_val}") # Print actual exposure value

        # Start the frame grabbing thread
        self.running = True
        self.thread = threading.Thread(target=self._frame_grabber, daemon=True)
        self.thread.start()

    def _frame_grabber(self):
        while self.running:
            if self.cap.isOpened():
                ret = self.cap.grab()
                if ret:
                    _, frame = self.cap.retrieve()
                    with self.lock:
                        self.frame = frame
            time.sleep(0.00001)

    def read_frame(self, blocking=True):
        if self.cap is None or not self.cap.isOpened():
            raise Exception("Camera is not open")

        if blocking:
            while self.frame is None:
                time.sleep(0.00001)  # Wait for the first frame to be grabbed

        with self.lock:
            frame_copy = None if self.frame is None else self.frame
            self.frame = None
        return frame_copy

    def get_angles(self, pixels):
        """
        Calculates the elevation and azimuth angles for each pixel location, accounting for the intrinsic camera matrix and distortion coefficients.

        Parameters:
        - K: numpy array of shape (3, 3), the intrinsic camera matrix.
        - distortions: numpy array of shape (1, 5) or (5,), the distortion coefficients.
        - pixels: numpy array of shape (N, 2), the pixel locations.

        Returns:
        - angles: numpy array of shape (N, 2), where each row contains [elevation, azimuth] in radians.
        """
        # Convert pixel coordinates to float32 for cv2 functions
        pixels = pixels.astype(np.float32)

        # Undistort the pixel points
        undistorted_points = cv2.undistortPoints(pixels, self.K, self.dist_coeffs)

        # Convert undistorted normalized points (Nx1x2) to Nx2 format
        undistorted_points = undistorted_points.reshape(-1, 2)

        # Normalized camera coordinates
        x = undistorted_points[:, 0]
        y = undistorted_points[:, 1]

        # Calculate elevation and azimuth
        elevation = np.arctan2(y, np.sqrt(x**2 + 1))  # Elevation angle
        azimuth = np.arctan2(x, 1)  # Azimuthal angle

        # Combine elevation and azimuth into an Nx2 array
        angles = np.stack((azimuth, elevation), axis=-1)

        return angles

    def release_camera(self):
        self.running = False
        # Wait for the frame grabbing thread to finish
        if self.thread is not None:
            self.thread.join()
        if self.cap is not None:
            print("Releasing camera...")
            self.cap.release()
            self.cap = None
            print("Camera released.")

    def __del__(self):
        self.release_camera()


def get_default_video_config(w=1920, h=1200):
    """
    Get default video configuration parameters used across the codebase.

    Args:
        w: Width of the video frame
        h: Height of the video frame

    Returns:
        Tuple of (camera_matrix, distortion_coefficients, is_fisheye)
    """
    K = np.array(
        [
            [1059.15420448, 0.00000000e00, w / 2],
            [0.00000000e00, 1059.15420448, h / 2],
            [0.00000000e00, 0.00000000e00, 1.00000000e00],
        ]
    )
    is_fisheye = True
    dist_coeffs = torch.tensor(
        [-0.08717226, -0.01354166, 0, 0, -0.00625567, 0.00485816]
    ).float()
    return K, dist_coeffs, is_fisheye
