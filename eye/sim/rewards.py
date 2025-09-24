import torch
import torch.nn.functional as F
import torchvision
from typing import List
import open_clip


class ClipReward:
    """
    Reward function that rewards agent based on what's in the fovea compared to CLIP vector
    """

    def __init__(
        self, 
        positive_prompts: List[str], 
        negative_prompts: List[str], 
        model: torch.nn.Module,
        preprocess,
        tokenizer,
        device="cuda"
    ):
        self.model = model
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.positive_prompts = positive_prompts
        # Load positive images from file paths
        positive_imgs = []
        for img_path in positive_prompts:
            # Load image and convert to float tensor in [0,1] range
            img = torchvision.io.read_image(img_path).float().div(255.0)[:3].half()
            # Resize to 224x224 if not already
            if img.shape[1:] != (224, 224):
                img = F.interpolate(img.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False)[0]
            # Add batch dimension and preprocess
            img = self.preprocess(img.unsqueeze(0))
            positive_imgs.append(img)
        positive_imgs = torch.cat(positive_imgs, dim=0).cuda()
        # Load negative images
        negative_imgs = []
        if negative_prompts:
            for img_path in negative_prompts:
                # Load image and convert to float tensor in [0,1] range
                img = torchvision.io.read_image(img_path).float().div(255.0)[:3].half()
                # Resize to 224x224 if not already
                if img.shape[1:] != (224, 224):
                    img = F.interpolate(img.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False)[0]
                # Add batch dimension and preprocess
                img = self.preprocess(img.unsqueeze(0))
                negative_imgs.append(img)
            # Stack all negative images
            if negative_imgs:
                negative_imgs = torch.cat(negative_imgs, dim=0).cuda()

        # Pre-compute text features for positive and negative prompts
        with torch.no_grad():
            # Tokenize and encode positive prompts
            # pos_tokens = self.tokenizer(positive_prompts).to(device)
            # self.positive_features = self.model.encode_text(pos_tokens)
            self.positive_features = self.model.encode_image(positive_imgs)
            self.positive_features /= self.positive_features.norm(dim=-1, keepdim=True)

            # Tokenize and encode negative prompts
            if negative_prompts:
                # neg_tokens = self.tokenizer(negative_prompts).to(device)
                # self.negative_features = self.model.encode_text(neg_tokens)
                self.negative_features = self.model.encode_image(negative_imgs)
                self.negative_features /= self.negative_features.norm(
                    dim=-1, keepdim=True
                )
            else:
                self.negative_features = None
        self._target_clip_vec = self.positive_features.mean(dim=0, keepdim=True)
        self._target_clip_vec /= self._target_clip_vec.norm(dim=-1, keepdim=True)

    @property
    def target_clip_vec(self):
        return self._target_clip_vec

    def get_relevancy(self, imgs: torch.Tensor, clipped=True):
        with torch.no_grad():
            # Preprocess and encode image
            imgs = self.preprocess(imgs.half())

            # Encode image
            image_features = self.model.encode_image(imgs)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            # Compute all similarities
            pos_similarities = (
                image_features @ self.positive_features.T
            )  # Shape: (B, num_positive)

            neg_similarities = (
                image_features @ self.negative_features.T
            )  # Shape: (B, num_negative)
            # For each positive prompt, compare against all negative prompts
            all_relevancies = []
            for pos_idx in range(pos_similarities.shape[-1]):
                pos_vals = pos_similarities[..., pos_idx : pos_idx + 1]  # (B, 1)
                repeated_pos = pos_vals.repeat(
                    1, neg_similarities.shape[-1]
                )  # (B, num_negative)

                # Stack positive and negative similarities
                sims = torch.stack(
                    (repeated_pos, neg_similarities), dim=-1
                )  # (B, num_negative, 2)

                # Apply softmax with temperature of 10
                softmax = torch.softmax(10 * sims, dim=-1)  # (B, num_negative, 2)

                # Find worst case comparison
                best_idx = softmax[..., 0].argmin(dim=1)  # (B,)
                relevancy = torch.gather(
                    softmax,
                    1,
                    best_idx[..., None, None].expand(
                        -1, neg_similarities.shape[-1], 2
                    ),
                )[:, 0, :]  # (B, 2)
                all_relevancies.append(relevancy)

            # Take max relevancy across all positive prompts
            relevancy = torch.stack(all_relevancies, dim=1)  # (B, num_positive, 2)
            relevancy = relevancy[..., 0].max(dim=-1)[0]  # (B,)
            if clipped:
                relevancy = torch.clip(relevancy - 0.5, 0.0, 1.0)
            return relevancy

    def __call__(self, fovea: torch.Tensor) -> torch.Tensor:
        """
        Returns scalar reward based on what's in the fovea compared to CLIP vector

        Args:
            fovea: Tensor of shape (B, C, H, W) with values in [0, 1]
                  This should be the highest resolution crop from the multicrop output

        Returns:
            reward: Scalar tensor representing the softmaxed CLIP similarity score
        """
        with torch.no_grad():
            # Preprocess and encode image
            # Resize to 224x224 if not already that size
            if fovea.shape[-2:] != (224, 224):
                fovea = F.interpolate(
                    fovea, size=(224, 224), mode="bilinear", align_corners=False
                )

            # Normalize using CLIP's mean and std
            fovea = self.preprocess(fovea.half())

            # Encode image
            image_features = self.model.encode_image(fovea)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            # Compute all similarities
            pos_similarities = (
                image_features @ self.positive_features.T
            )  # Shape: (B, num_positive)

            if self.negative_features is not None:
                neg_similarities = (
                    image_features @ self.negative_features.T
                )  # Shape: (B, num_negative)
                # For each positive prompt, compare against all negative prompts
                all_relevancies = []
                for pos_idx in range(pos_similarities.shape[-1]):
                    pos_vals = pos_similarities[..., pos_idx : pos_idx + 1]  # (B, 1)
                    repeated_pos = pos_vals.repeat(
                        1, neg_similarities.shape[-1]
                    )  # (B, num_negative)

                    # Stack positive and negative similarities
                    sims = torch.stack(
                        (repeated_pos, neg_similarities), dim=-1
                    )  # (B, num_negative, 2)

                    # Apply softmax with temperature of 10
                    softmax = torch.softmax(10 * sims, dim=-1)  # (B, num_negative, 2)

                    # Find worst case comparison
                    best_idx = softmax[..., 0].argmin(dim=1)  # (B,)
                    relevancy = torch.gather(
                        softmax,
                        1,
                        best_idx[..., None, None].expand(
                            -1, neg_similarities.shape[-1], 2
                        ),
                    )[:, 0, :]  # (B, 2)
                    all_relevancies.append(relevancy)

                # Take max relevancy across all positive prompts
                relevancy = torch.stack(all_relevancies, dim=1)  # (B, num_positive, 2)
                reward = relevancy[..., 0].max(dim=-1)[0]  # (B,)
            else:
                # If no negative prompts, just take max similarity to positive prompts
                reward = pos_similarities.max(dim=-1)[0]  # (B,)
            relevancy = torch.clip(reward - 0.5, 0.0, 1.0)
            weighting = torch.arange(relevancy.shape[0]-1, -1, -1, device=relevancy.device).float().softmax(dim=0)
            return (relevancy * weighting).sum()


class ConstancyReward:
    """
    Reward function that rewards agent based on how consistent the fovea is
    between consecutive frames. Penalizes rapid changes in the fovea content.
    """
    def __init__(self, device: torch.device):
        """
        Initialize the constancy reward function.
        
        Args:
            device: The device to store tensors on
            alpha: Weight factor for the reward calculation (higher = stronger penalty)
        """
        self.device = device
        self.prev_fovea = None
        
    def __call__(self, fovea: torch.Tensor) -> torch.Tensor:
        """
        Returns scalar reward based on how consistent the fovea is compared to previous frame.
        Lower values (more negative) indicate more change between frames.
        
        Args:
            fovea: Tensor of shape (B, C, H, W) with values in [0, 1]
                  This should be the highest resolution crop from the multicrop output
                  
        Returns:
            reward: Scalar tensor representing the negative change between frames
                   (0 = no change, negative values = more change)
        """
        if self.prev_fovea is None:
            # If no previous frame, reward is 0
            reward = torch.tensor(0.0, device=self.device)
        else:
            # Ensure shapes match
            assert self.prev_fovea.shape == fovea.shape
            
            # Calculate L2 distance between current and previous fovea
            # Lower is better (less change)
            diff = ((fovea - self.prev_fovea) ** 2).mean()
            
            # Convert to reward (negative, since we want to penalize change)
            reward = -diff
            
        # Store current fovea for next comparison
        self.prev_fovea = fovea.detach().clone()
        
        return reward
    
    def reset(self):
        """
        Reset the stored previous frame.
        Should be called when starting a new episode or when there's a discontinuity.
        """
        self.prev_fovea = None