import numpy as np
import cv2
import serial
import time
from typing import Tuple, Optional
import viser.transforms as vtf
from eye.lkm import LKMotor
from eye.camera import OCVCamera
from eye.tracking import FocalTracker
from eye.pid import PIDController
import contextlib


def angle_direction(source, target):
    source = np.mod(source, 360)
    target = np.mod(target, 360)
    # Calculate the difference
    difference = np.mod((target - source), 360)

    # Determine direction
    if difference == 0:
        return 0  # Angles are the same
    elif difference > 0 and difference < 180:
        return 0  # Counterclockwise
    else:
        return 1  # Clockwise


class Eyeball:
    zero_elev: float = 110.22  # degrees
    pursuit_speed = 300  # dps
    max_saccade_speed = 1000  # dps

    def __init__(
        self,
        P_pos_1: float = 70,
        I_pos_1: float = 3,
        P_speed_1: float = 2000,
        I_speed_1: float = 30,
        P_pos_2: float = 70,
        I_pos_2: float = 3,
        P_speed_2: float = 2000,
        I_speed_2: float = 30,
    ):
        self.P_pos_1 = P_pos_1
        self.I_pos_1 = I_pos_1
        self.P_speed_1 = P_speed_1
        self.I_speed_1 = I_speed_1
        self.P_pos_2 = P_pos_2
        self.I_pos_2 = I_pos_2
        self.P_speed_2 = P_speed_2
        self.I_speed_2 = I_speed_2

        self.con = serial.Serial("/dev/ttyUSB0", 2_000_000)
        self.azimuth_motor = LKMotor(self.con, 0x01)
        self.elev_motor = LKMotor(self.con, 0x02)
        self.saccading = False

        

        self.azimuth_motor.turn_on()
        self.azimuth_motor.set_pid_parameters(
            0x96, self.P_pos_1, self.I_pos_1, 0, to_rom=True
        )  # position
        self.azimuth_motor.set_pid_parameters(
            0x97, self.P_speed_1, self.I_speed_1, 0, to_rom=True
        )  # speed
        self.elev_motor.turn_on()
        self.elev_motor.set_pid_parameters(
            0x96, self.P_pos_2, self.I_pos_2, 0, to_rom=True
        )  # position
        self.elev_motor.set_pid_parameters(
            0x96, self.P_speed_2, self.I_speed_2, 0, to_rom=True
        )  # position
        self.cam = OCVCamera(width=1920, height=1200, fps=90)
        self.tracker = None
        self.azimuth_pid = PIDController(
            1.8,
            0.3,
            0.001,
            max_output=Eyeball.pursuit_speed,
            integral_bound=Eyeball.pursuit_speed,
            leakage_factor=0.99,
        )
        self.elev_pid = PIDController(
            1.8,
            0.3,
            0.001,
            max_output=Eyeball.pursuit_speed,
            integral_bound=Eyeball.pursuit_speed,
            leakage_factor=0.99,
        )

    def camera_angle_to_eye(
        self, cam_azim, cam_elev, pixel_azim, pixel_elev
    ) -> Tuple[float, float]:
        """
        cam azim, elev is the current *physical* position of the camera (motors)
        pixel azim, elev is the camera-relative pixel elevation, azimuth

        this function converts this information to a single global eye azimuth, elevation
        """

        cam_so3 = vtf.SO3.from_z_radians(cam_azim) @ vtf.SO3.from_y_radians(-cam_elev)
        # this transform has the x axis pointing along the camera's line of sight
        # now we want to get the ray so3 which is computed by rotating the camera so3
        ray_so3 = (
            cam_so3
            @ vtf.SO3.from_z_radians(-pixel_azim)
            @ vtf.SO3.from_y_radians(pixel_elev)
        )
        # now convert this so3 into a vector and convert back to azimuth, elevation
        ray_vec = ray_so3.as_matrix()[:, 0]  # use x axis for line of sight
        ray_azim = np.arctan2(ray_vec[1], ray_vec[0])
        ray_elev = np.arctan2(ray_vec[2], np.sqrt(ray_vec[0] ** 2 + ray_vec[1] ** 2))
        return ray_azim, ray_elev

    def azimuth_elev(self):
        return np.deg2rad(np.mod(self.azimuth_motor.get_angle(), 360)), np.deg2rad(np.mod(self.elev_motor.get_angle(), 360) - Eyeball.zero_elev)

    def saccade(self, frame, pix_x, pix_y):
        pix_azim_elev = self.cam.get_angles(np.array([[pix_x, pix_y]]))
        cam_elev = np.mod(self.elev_motor.get_angle(), 360) - Eyeball.zero_elev
        cam_azim = np.mod(self.azimuth_motor.get_angle(), 360)
        eye_azim, eye_elev = self.camera_angle_to_eye(
            np.deg2rad(cam_azim),
            np.deg2rad(cam_elev),
            pix_azim_elev[0, 0],
            pix_azim_elev[0, 1],
        )
        # convert back to degrees
        eye_azim = np.rad2deg(eye_azim)
        eye_elev = np.rad2deg(eye_elev) + Eyeball.zero_elev
        eye_azim = np.mod(eye_azim, 360)
        azim_direction = angle_direction(cam_azim, eye_azim)
        elev_direction = angle_direction(cam_elev + Eyeball.zero_elev, eye_elev)
        self.saccading = True
        self.sacc_start = time.time()
        self.azimuth_motor.set_single_loop_angle(
            eye_azim, azim_direction, speed=self.max_saccade_speed
        )
        self.elev_motor.set_single_loop_angle(
            eye_elev, elev_direction, speed=self.max_saccade_speed
        )
        if self.tracker is not None:
            self.tracker.reset_tracking(frame, pix_x, pix_y)
            # however we need to manually seed the tracker with the center pix as the start otherwise when saccading ends itll be totally off
            self.tracker.seed_points = self.tracker._get_seed(*self.cam.get_centerpoint())
            # also reset the integral controllers
            self.azimuth_pid.reset_setpoint()
            self.elev_pid.reset_setpoint()

    def analytic_control_step(
        self, frame: Optional[np.ndarray] = None, motor_lock=contextlib.nullcontext()
    ):
        # ignore frames unless we are initialized
        if self.tracker is None and frame is None:
            return
        elif self.tracker is None and frame is not None:
            self.tracker = FocalTracker(frame)
        now = time.time()
        if self.saccading:
            tol = 0.05  # dps
            # patiently wait until the motors are done moving
            with motor_lock:
                if (
                    abs(self.azimuth_motor.get_speed()) < tol
                    and abs(self.elev_motor.get_speed()) < tol
                ):
                    self.saccading = False
                    print("Saccade took", now - self.sacc_start)
        else:
            # smooth pursuit
            if frame is not None:
                self.tracked_point, _, success = self.tracker.track(frame, now)
                self.old_track = self.tracked_point
            else:
                self.tracked_point = self.tracker.extrapolate(now)
            azimuth_speed = self.azimuth_pid.update(
                self.cam.get_centerpoint()[0], self.tracked_point[0], now
            )
            elev_speed = self.elev_pid.update(
                self.cam.get_centerpoint()[1], self.tracked_point[1], now
            )
            with motor_lock:
                self.azimuth_motor.set_speed(azimuth_speed)
                self.elev_motor.set_speed(elev_speed)
            if frame is not None and not success:
                self.tracker.reset_tracking(
                    frame, frame.shape[1] // 2, frame.shape[0] // 2
                )

    def teardown(self):
        self.azimuth_motor.turn_off()
        self.elev_motor.turn_off()
        self.con.close()
