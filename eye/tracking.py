import cv2
import numpy as np
import matplotlib.pyplot as plt


class OpticalFlowCalculator:
    @staticmethod
    def calculate_flow(prev_gray, frame, points, seed_points=None):
        prev_gray = cv2.cvtColor(prev_gray, cv2.COLOR_BGR2GRAY)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Convert points to a NumPy array of type float32, if not already in that format
        points = np.array(points, dtype=np.float32).reshape(-1, 2)

        # Calculate optical flow using the Lucas-Kanade method
        # mostly using defaults, except more iterations, larger winsize
        nextPts, status, _ = cv2.calcOpticalFlowPyrLK(
            prevImg=prev_gray,
            nextImg=gray,
            prevPts=points,
            nextPts=seed_points,
            winSize=(25, 25),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 0.01),
            flags=cv2.OPTFLOW_USE_INITIAL_FLOW if seed_points is not None else 0,
            minEigThreshold=1e-4,
        )

        # Filter out points where status is 0 (tracking failed)
        valid_points = nextPts[(status == 1).squeeze()]
        flow_vectors = valid_points - points[(status == 1).squeeze()]

        return valid_points.reshape(-1, 2), flow_vectors.reshape(-1, 2), status


def find_main_soft_mode(data):
    """
    Find the strongest soft mode using numpy's histogram.

    Parameters:
    -----------
    data : array-like
        Input array of numerical values
    bin_width : float
        Width of each bin (controls smoothness)
    vis : bool, optional
        If True, displays a visualization of the histograms and mode

    Returns:
    --------
    mode_value : float
        The position of the strongest mode
    mode_height : float
        The density at the mode
    """
    # Compute histogram
    hist, bin_edges = np.histogram(data, bins="auto", density=True)
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2

    # Find the highest peak
    peak_idx = np.argmax(hist)
    mode_value = bin_centers[peak_idx]
    mode_height = hist[peak_idx]
    return mode_value, mode_height, (bin_edges[1] - bin_edges[0])


class FocalTracker:
    downsize_fac: int = 1

    def __init__(self, init_frame):
        self.init_frame = self._downsize_frame(init_frame)
        # Used for extrapolating the next point
        self.past_ts = []
        self.past_points = []
        # memoize the points
        R = 4  # radius of the biggest circle
        N_circs = 3  # number of concentric circles
        N_points = 30  # number of points per circle
        pts = [(0.0, 0.0)]
        for i in range(N_circs):
            r = R // N_circs * (i + 1)
            for j in range(N_points):
                theta = 2 * np.pi / N_points * j
                x = r * np.cos(theta) + 0
                y = r * np.sin(theta) + 0
                pts.append([x, y])
        self.zero_centered_seeds = np.array(pts, dtype=np.float32)
        middle_x = (init_frame.shape[1] / self.downsize_fac) // 2
        middle_y = (init_frame.shape[0] / self.downsize_fac) // 2
        pts = self._get_seed(middle_x, middle_y)
        self.start_points = pts
        self.seed_points = pts

    def _downsize_frame(self, frame):
        if self.downsize_fac == 1:
            return frame
        return cv2.resize(
            frame,
            (frame.shape[1] // self.downsize_fac, frame.shape[0] // self.downsize_fac),
            interpolation=cv2.INTER_LINEAR,
        )

    def _get_seed(self, middle_x, middle_y):
        result = np.empty_like(self.zero_centered_seeds)
        result[...] = self.zero_centered_seeds + np.array([middle_x, middle_y])
        return result

    def track(self, frame, now):
        frame = self._downsize_frame(frame)
        valid_pts, flow_vectors, _ = OpticalFlowCalculator.calculate_flow(
            self.init_frame, frame, self.start_points, self.seed_points
        )
        if flow_vectors.shape[0] == 0:
            return np.zeros(2), np.zeros(2), False
        flow_mags = np.linalg.norm(flow_vectors, axis=1)
        mode_flow, _, bin_size = find_main_soft_mode(flow_mags)
        s = bin_size * 3
        # filter out points that are too far from the mean
        check = (flow_mags < mode_flow + s) & (flow_mags > mode_flow - s)
        valid_pts = valid_pts[check]
        new_point = np.median(valid_pts, axis=0)
        self.seed_points = self._get_seed(new_point[0], new_point[1])
        self.init_frame = frame
        self.start_points = self.seed_points
        # update the past points
        self.past_ts.append(now)
        self.past_points.append(new_point)
        # only keep the last two points
        if len(self.past_ts) > 2:
            self.past_ts.pop(0)
            self.past_points.pop(0)
        return new_point * self.downsize_fac, flow_vectors * self.downsize_fac, True

    def extrapolate(self, now):
        # linearly interpolates the last two points to get the next point based on time
        if len(self.past_ts) == 0 or len(self.past_ts) == 1:
            return np.array(
                [self.init_frame.shape[1] // 2, self.init_frame.shape[0] // 2]
            )
        t1, t2 = self.past_ts
        p1, p2 = self.past_points
        # linear interpolation
        return (p1 + (p2 - p1) * (now - t1) / (t2 - t1)) * self.downsize_fac

    def reset_tracking(self, frame, x, y):
        self.init_frame = self._downsize_frame(frame)
        self.start_points = self._get_seed(x / self.downsize_fac, y / self.downsize_fac)
        self.seed_points = self.start_points
        self.past_ts = []
        self.past_points = []
