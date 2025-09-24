from eye.zmq_tools import Subscriber
import time
import numpy as np
from gello.robots.ur import URRobot
import math

class TrajectoryInterpolator:
    def __init__(self):
        self.subscriber = Subscriber("ipc:///tmp/eye_action_chunk")
        self.action_chunks = []
        self.initial_timestamps = []
        self.fps = 30
        self.timeout = 0.5
        self.ur5 = URRobot(robot_ip="172.22.22.2", no_gripper=False)
        self.subscriber.on_recv_action_chunk(self.on_recv_action_chunk)

    def on_recv_action_chunk(self, action_chunk, timestamp):
        self.initial_timestamps.append(timestamp)
        self.action_chunks.append(action_chunk)

trajectory_interpolator = TrajectoryInterpolator()
while True:
    current_time = time.perf_counter()
    if(len(trajectory_interpolator.action_chunks) == 0 or current_time - trajectory_interpolator.initial_timestamps[-1] > trajectory_interpolator.timeout):
        if not trajectory_interpolator.ur5.freedrive_enabled():
            trajectory_interpolator.ur5.robot.servoStop(a=.5)
            print("Servo stopping")
            time.sleep(0.2)
            trajectory_interpolator.ur5.set_freedrive_mode(True)
    else:
        avg_action = 0
        count = 0
        time_per_step = 1.0 / trajectory_interpolator.fps # adjust this value based on your actual time step
        while int((current_time - trajectory_interpolator.initial_timestamps[0]) * trajectory_interpolator.fps) >= trajectory_interpolator.action_chunks[0].shape[2] - 1:
            # also remove all old action chunks that are no longer needed
            trajectory_interpolator.action_chunks.pop(0)
            trajectory_interpolator.initial_timestamps.pop(0)
            if len(trajectory_interpolator.action_chunks) == 0:
                break
        if len(trajectory_interpolator.action_chunks) == 0:
                continue
        num_chunks = len(trajectory_interpolator.action_chunks)
        total_w = 0
        for i in range(num_chunks):
            time_delta = current_time - trajectory_interpolator.initial_timestamps[i]
            index = int(time_delta * trajectory_interpolator.fps)
            if trajectory_interpolator.ur5.freedrive_enabled():
                trajectory_interpolator.ur5.set_freedrive_mode(False)
            joint_state_1 = trajectory_interpolator.action_chunks[i][0, 0, index, :]
            joint_state_2 = trajectory_interpolator.action_chunks[i][0, 0, index + 1, :]
            
            # Calculate interpolation factor (0 to 1)
            # Assuming each index represents a fixed time step (e.g., 0.1 seconds)
            alpha = time_delta / time_per_step
            alpha = max(0.0, min(1.0, alpha))  # Clamp between 0 and 1
            
            # Linear interpolation
            interpolated_state = joint_state_1 * (1 - alpha) + joint_state_2 * alpha
            w = math.exp(-(num_chunks-i)*.01)#higher weights earlier chunks more
            total_w += w
            avg_action += interpolated_state * w
        avg_action /= total_w
        trajectory_interpolator.ur5.command_joint_state(avg_action,lookahead_time=0.2, gain=100, acceleration=0.5, velocity = 0.1)
