import cv2
import numpy as np
from eye.camera import OCVCamera

"""
run from Jan 20, 300 images pretty good coverage
=== Standard Calibration Parameters ===

Camera Matrix (copy-paste format):
np.array([
    [1404.01165059, 0.00000000, 945.97802326],
    [0.00000000, 1415.30228207, 595.22133356],
    [0.00000000, 0.00000000, 1.00000000],
])

Distortion Coefficients (copy-paste format):
np.array([
    [-0.50523430, 0.23055764, 0.00158941, 0.00055170, -0.04957008]
])

=== Fisheye Calibration Parameters ===

Camera Matrix (copy-paste format):
np.array([
    [1059.15420448, 0.00000000, 924.49949221],
    [0.00000000, 1058.16700723, 609.46814478],
    [0.00000000, 0.00000000, 1.00000000],
])

Distortion Coefficients (copy-paste format):
np.array([
    [-0.08717226, -0.01354166, -0.00625567, 0.00485816]
])

"""


def calibrate_camera_both(camera, board_width, board_height, num_images=200):
    # Termination criteria for corner sub-pixel accuracy
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)

    # Prepare object points based on the chessboard dimensions
    objp = np.zeros((board_width * board_height, 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_width, 0:board_height].T.reshape(-1, 2)

    # Arrays to store object points and image points from all images
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane
    objpoints_fish = []  # 3d points for fisheye calibration
    imgpoints_fish = []  # 2d points for fisheye calibration

    found_images = 0
    print("Starting camera calibration... Press 'q' to quit early.")
    from viser import ViserServer

    ser = ViserServer()
    stop = ser.gui.add_button("Stop Calibration")
    running = True

    @stop.on_click
    def stop_calibration(_):
        nonlocal running
        running = False

    while found_images < num_images:
        frame = camera.read_frame()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(
            gray, (board_width, board_height), None
        )

        if ret:
            # Standard calibration points
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Fisheye calibration points
            objpoints_fish.append(objp.reshape(-1, 1, 3))
            imgpoints_fish.append(corners2.reshape(-1, 1, 2))

            # Draw and display the corners
            frame = cv2.drawChessboardCorners(
                frame, (board_width, board_height), corners2, ret
            )
            found_images += 1
            print(f"Found {found_images} chessboard images")

        # Display the frame with detected corners
        ser.scene.set_background_image(frame, jpeg_quality=50)
        if not running:
            break

    if len(objpoints) == 0 or len(imgpoints) == 0:
        print("No corners found in images. Calibration failed.")
        return None, None, None, None

    # Fisheye calibration
    print("\nPerforming fisheye calibration...")
    calibration_flags = (
        cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
        + cv2.fisheye.CALIB_CHECK_COND
        + cv2.fisheye.CALIB_FIX_SKEW
    )

    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    ret_fish, mtx_fish, dist_fish, _, _ = cv2.fisheye.calibrate(
        objpoints_fish,
        imgpoints_fish,
        gray.shape[::-1],
        K,
        D,
        None,
        None,
        calibration_flags,
        criteria,
    )

    # Standard calibration
    print("\nPerforming standard calibration...")
    ret_std, mtx_std, dist_std, rvecs_std, tvecs_std = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )

    if ret_std and ret_fish:
        print("\n=== Standard Calibration Parameters ===")
        print("\nCamera Matrix (copy-paste format):")
        print("np.array([")
        for row in mtx_std:
            print(f"    [{', '.join(f'{x:.8f}' for x in row)}],")
        print("])")

        print("\nDistortion Coefficients (copy-paste format):")
        print("np.array([")
        print(f"    [{', '.join(f'{x:.8f}' for x in dist_std.ravel())}]")
        print("])")

        print("\n=== Fisheye Calibration Parameters ===")
        print("\nCamera Matrix (copy-paste format):")
        print("np.array([")
        for row in mtx_fish:
            print(f"    [{', '.join(f'{x:.8f}' for x in row)}],")
        print("])")

        print("\nDistortion Coefficients (copy-paste format):")
        print("np.array([")
        print(f"    [{', '.join(f'{x:.8f}' for x in dist_fish.ravel())}]")
        print("])")
    else:
        print("Calibration failed.")
        return None, None, None, None

    return mtx_std, dist_std, mtx_fish, dist_fish


if __name__ == "__main__":
    # Set the desired resolution for calibration
    camera = OCVCamera(width=1920, height=1200, fps=5)
    camera.open_camera()

    try:
        # Perform both calibrations
        mtx_std, dist_std, mtx_fish, dist_fish = calibrate_camera_both(
            camera, board_width=8, board_height=6, num_images=300
        )
    finally:
        camera.release_camera()
