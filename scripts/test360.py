from viser import ViserServer
import torch
import numpy as np
import viser.transforms as vtf
import time
import cv2
from typing import Tuple
from eye.camera import radial_and_tangential_undistort, get_default_video_config
import math
from eye.sim.spherical_video import SphericalVideo
from eye.sim.rewards import ClipReward
import argparse
from collections import deque

from eye.image_utils import pad_to_aspect_ratio


if __name__ == "__main__":
    from pathlib import Path
    from eye.transforms import SO3
    
    
    parser = argparse.ArgumentParser(description='View 360° videos with foveated rendering')
    parser.add_argument('--video', type=Path, help='Path to local video file or YouTube video ID/URL', default=Path("downloads/boris_tracker.mp4"))
    args = parser.parse_args()

    # check if it's a youtube video if it's just an id...
    # this means it's just a youtube video id and no extention
    if args.video.suffix == "":
        print(f"Downloading video from YouTube: {args.video}")
    else:
        print(f"Using local video: {args.video}")

    server = ViserServer()
    w = 1920
    h = 1200
    K, dist_coeffs, is_fisheye = get_default_video_config(w, h)
    from eye.foveal_encoders import crop_sizes_from_levels
    crop_sizes = crop_sizes_from_levels(4, 224, 1792)
    sim = SphericalVideo(args.video, K, dist_coeffs, h, w, is_fisheye, "cuda", crop_sizes, decoder_device="cpu")
    sim.reset(True)
    playing = server.gui.add_text("Video", args.video.name)
    playing = server.gui.add_button("Update", False)
    playing = server.gui.add_checkbox("Playing", False)

    # Benchmarking setup
    render_times = deque(maxlen=100)  # Keep last 100 measurements
    frame_count = 0

    while True:
        clients = server.get_clients()
        if len(clients) == 0:
            time.sleep(0.01)
            continue
        client = clients[max(clients.keys())]
        R = SO3(torch.from_numpy(client.camera.wxyz).float().cuda())
        
        # Benchmark render_multicrop
        start_time = time.perf_counter()
        multicrop = sim.render_multicrop(R)  # Now shape is (1, n_levels, 3, h, w)
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        render_time = end_time - start_time
        render_times.append(render_time)
        frame_count += 1
        
        # Print benchmarking stats every 30 frames
        if frame_count % 30 == 0:
            avg_time = sum(render_times) / len(render_times)
            min_time = min(render_times)
            max_time = max(render_times)
            fps = 1.0 / avg_time if avg_time > 0 else 0
            print(f"render_multicrop - Current: {render_time*1000:.2f}ms, "
                  f"Avg: {avg_time*1000:.2f}ms, Min: {min_time*1000:.2f}ms, "
                  f"Max: {max_time*1000:.2f}ms, FPS: {fps:.1f}")
        
        # Remove batch dimension and rearrange for 2x2 grid
        multicrop = multicrop.squeeze(0)  # Now shape is (n_levels, 3, h, w)
        n_crops = multicrop.shape[0]
        assert n_crops == 4, "This visualization assumes exactly 4 crop levels"

        # First concatenate pairs horizontally, then stack vertically
        # Reversed order: wider views (3,2) on top, zoomed views (1,0) on bottom
        top_row = torch.concatenate([multicrop[3], multicrop[2]], axis=2)
        bottom_row = torch.concatenate([multicrop[1], multicrop[0]], axis=2)
        tiled_multicrop = torch.concatenate([top_row, bottom_row], axis=1)

        view = (255 * tiled_multicrop).byte().cpu().numpy().transpose(1, 2, 0)
        aspect = client.camera.aspect
        view = pad_to_aspect_ratio(view, aspect)
        server.scene.set_background_image(view, jpeg_quality=99)
        if playing.value:
            if not sim.advance():
                print("reset, seek to 0")
                sim.reset()
