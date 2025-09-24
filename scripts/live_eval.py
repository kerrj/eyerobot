from viser import ViserServer
import av
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
from eye.sim.gyms import EyeGym
from eye.transforms import SO3
from eye.agents import PatchTokAgent
from eye.foveal_encoders import crop_sizes_from_levels
from pathlib import Path
from eye.foveal_encoders import create_foveated_batch
import time
from transformers import SiglipModel, SiglipProcessor
import open_clip
import torchvision

def pad_to_aspect_ratio(img, target_aspect, pad_value=0):
    """
    Pad an image to match a target aspect ratio (width/height)

    Args:
        img: numpy array of shape (H, W, C)
        target_aspect: float, desired width/height ratio
        pad_value: int/float, value to use for padding

    Returns:
        padded_img: numpy array padded to match target aspect ratio
    """
    h, w = img.shape[:2]
    current_aspect = w / h

    if current_aspect < target_aspect:
        # Need to add width padding
        new_w = int(h * target_aspect)
        pad_w = new_w - w
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        padded_img = np.pad(
            img,
            ((0, 0), (pad_left, pad_right), (0, 0)),
            mode="constant",
            constant_values=pad_value,
        )
    else:
        # Need to add height padding
        new_h = int(w / target_aspect)
        pad_h = new_h - h
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        padded_img = np.pad(
            img,
            ((pad_top, pad_bottom), (0, 0), (0, 0)),
            mode="constant",
            constant_values=pad_value,
        )

    return padded_img

def draw_action_arrows(view, probs, action_dict, scale=200, base_thickness=2):
    """
    Draw arrows representing the action distribution on the image.
    
    Args:
        view: numpy array of shape (H, W, C)
        probs: torch tensor of action probabilities
        action_dict: dictionary mapping action indices to (azimuth, elevation) pairs
        scale: scaling factor for arrow length
        base_thickness: base thickness of arrows
    """    
    h, w = view.shape[:2]
    center = (w // 2, h // 2)
    
    # Convert probs to numpy
    probs = probs.squeeze().cpu().numpy()
    
    for action_idx, prob in enumerate(probs):
        if prob < 0.01:  # Skip very low probability actions
            continue
            
        azimuth, elevation = action_dict[action_idx]
        # Convert to pixel coordinates (azimuth is x, elevation is y)
        dx = int(-azimuth * scale)
        dy = int(-elevation * scale)  # Negative because y increases downward in image coordinates
        
        # Calculate arrow endpoint
        end_point = (center[0] + dx, center[1] + dy)
        
        # Scale thickness by probability
        thickness = int(base_thickness + prob * 3)
        color = (0, 55 + int(200 * prob), 0)  # Brighter green for higher probabilities
        
        cv2.arrowedLine(view, center, end_point, color, thickness, tipLength=0.2)

if __name__ == "__main__":
    # Initialize viser server and GUI controls
    server = ViserServer()
    w = 1920
    h = 1200
    K, dist_coeffs, is_fisheye = get_default_video_config(w, h)
    
    # Setup video
    vid_path = Path("downloads/The VR Contemporary Art Gallery in Opole. 8K 360_360_2160p.mp4")
    # vid_path = Path("data/x4_test.mp4")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_scales = 4
    crop_sizes = crop_sizes_from_levels(n_scales, 224, w)
    
    # Create video reader
    sim = SphericalVideo(vid_path, K, dist_coeffs, h, w, is_fisheye, device, crop_sizes=crop_sizes)
    
    # Initialize agent
    agent = PatchTokAgent(
        transformer_type="decoder",
        device=device,
        vit_type="dino",
        crop_sizes=crop_sizes,
        action_type="discrete"
    ).to(device)
    
    # Load pretrained weights
    # ckpt_path = "runs/ppo_360__1__1738297494/checkpoint_1000.pt" #AdamW and elevation only
    # ckpt_path = "runs/ppo_360__1__1738310014/checkpoint_1000.pt" # Adam model this one has the nicest peripheral behavior
    # ckpt_path = "runs/ppo_360__1__1738312752/checkpoint_1000.pt" # Adam model with big batch size
    # ckpt_path = "runs/ppo_360__1__1738447374/checkpoint_1000.pt"# first one with random z aug

    # ckpt_path = "runs/ppo_360__2__1738537693/checkpoint_1000.pt"# z aug but doesnt do well??
    # 4 seeds below
    # ckpt_path = "runs/ppo_360__4__1738372328/checkpoint_1400.pt" # good periphery
    # ckpt_path = "runs/ppo_360__2__1738372289/checkpoint_1400.pt" # good periphery
    # ckpt_path = "runs/ppo_360__3__1738372307/checkpoint_1400.pt" # bad left periphery
    # ckpt_path = "runs/ppo_360__1__1738372246/checkpoint_1400.pt" # bad left periphery, but better than prev

    # ckpt_path = "runs/ppo_360__1__1738633519/checkpoint_1000.pt" # dino w/ adamw+warmup, siglip reward (make sure to remove target proj)
    # ckpt_path = "runs/ppo_360__1__1738632990/checkpoint_1000.pt" # dino w/ adamw+warmup, openclip reward
    # ckpt_path = "runs/ppo_360__1__1738633044/checkpoint_1000.pt" # dino w/ adamw+warmup, 4 registers
    
    # ckpt_path = "runs/ppo_360__1__1738878801/checkpoint_500.pt"
    # ckpt_path = "runs/ppo_360__1__1738916016/checkpoint_600.pt" #feet, faces, surgical mask, ceiling, foor
    # ckpt_path = "runs/ppo_360__1__1739222781/checkpoint_final.pt" # feet,faces,surgical mask,ceiling,floor for 10M steps
    # ckpt_path = "runs/ppo_360__1__1738958568/checkpoint_600.pt" #all the prompts
    # ckpt_path = "runs/ppo_360__1__1739420263/checkpoint_600.pt" # KL target run
    ckpt_path = "runs/ppo_360__1__1739909618/checkpoint_100.pt"
    # state_dict = torch.load(ckpt_path, map_location=device)
    # agent.load_state_dict(state_dict["model_state_dict"])
    agent.eval()

    # Add GUI controls
    playing = server.gui.add_checkbox("Play Video", False)
    manual_control = server.gui.add_checkbox("Manual Control", True)
    
    # Initialize reward function
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-16", pretrained="laion2b_s34b_b88k", device=device, precision='fp16'
    )
    model.eval()
    preprocess = torchvision.transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711],
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-16")
    prompt_box = server.gui.add_text("Prompt",'a face')
    change_prompt = server.gui.add_button("Update Prompt")
    def get_prompt_embedding(prompt):
        with torch.no_grad():
            tokens = tokenizer([prompt]).to(device)
            embedding = model.encode_text(tokens)
            embedding /= embedding.norm(dim=-1, keepdim=True)
            return embedding.float()
    
    target_embedding = get_prompt_embedding(prompt_box.value)
    @change_prompt.on_click
    def _(_):
        # Calculate the embedding vector for the positive prompts
        global target_embedding
        target_embedding = get_prompt_embedding(prompt_box.value)

    while True:
        clients = server.get_clients()
        if len(clients) == 0:
            time.sleep(0.01)
            continue
            
        client = clients[max(clients.keys())]
        
        # Get current camera orientation from viser
        R = SO3(torch.from_numpy(client.camera.wxyz).float().cuda())
        
        multicrop = sim.render_multicrop(R).unsqueeze(0)
        if not manual_control.value:
            # Create observation from current frame and camera direction
            with torch.no_grad():
                obs = {
                    "multicrop": multicrop,
                    "eye_direction": R.as_matrix()[:, 2].unsqueeze(0),
                    "target_clip_vec": target_embedding
                }
                
                # Get action from agent
                action, _, _, _, probs = agent.get_action_and_value(obs, deterministic=True, return_distribution=True)
                #probs is a Categorical distribution over the actions.
                # each action ID can be mapped to azimuth,elevation via agent.action_dict[action_id]
                
                # Update camera orientation based on agent's action
                delta_azimuth, delta_elevation = action.squeeze()
                
                # Reconstruct rotation from clipped angles
                new_R = SO3.from_z_radians(delta_azimuth) @ R @ SO3.from_x_radians(delta_elevation)
                
                # Update client camera
                client.camera.wxyz = new_R.wxyz.cpu().numpy()
        
        # Display frame in grid layout that adapts to number of crops
        multicrop = multicrop.squeeze(0)  # Remove batch dimension (1, n_levels, 3, h, w) -> (n_levels, 3, h, w)
        n_crops = multicrop.shape[0]
        
        # Handle different numbers of crops
        if n_crops == 1:
            tiled_multicrop = multicrop[0]
        elif n_crops == 2:
            tiled_multicrop = torch.concatenate([multicrop[1], multicrop[0]], axis=2)
        elif n_crops == 3:
            # Create a 2x2 grid with empty top-left quadrant
            top_row = torch.concatenate([torch.zeros_like(multicrop[2]), multicrop[2]], axis=2)
            bottom_row = torch.concatenate([multicrop[1], multicrop[0]], axis=2)
            tiled_multicrop = torch.concatenate([top_row, bottom_row], axis=1)
        elif n_crops == 4:
            # 2x2 grid layout
            top_row = torch.concatenate([multicrop[3], multicrop[2]], axis=2)
            bottom_row = torch.concatenate([multicrop[1], multicrop[0]], axis=2)
            tiled_multicrop = torch.concatenate([top_row, bottom_row], axis=1)
        else:
            raise ValueError(f"Unsupported number of crops: {n_crops}. Must be between 1 and 4.")

        view = (255 * tiled_multicrop).byte().cpu().numpy().transpose(1, 2, 0)
        view = np.ascontiguousarray(view)
        if agent.action_type == "discrete" and not manual_control.value:
            draw_action_arrows(view, probs.probs, agent.action_dict)
        aspect = client.camera.aspect
        view = pad_to_aspect_ratio(view, aspect)
        server.scene.set_background_image(view, jpeg_quality=90)
        time.sleep(0.001)
        
        # Advance video if playing
        if playing.value:
            if not sim.advance():
                print("reset, seek to 0")
                sim.reset()
