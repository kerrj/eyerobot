from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Tuple, Union, Optional
import json
import torch
from pathlib import Path
from eye.foveal_encoders import crop_sizes_from_levels


@dataclass
class VisualSearchAgentConfig:
    window_size: int = 224
    device: str = "cuda"
    freeze_encoder: bool = True
    learnable_registers: int = 0
    n_levels: int = 4
    fovea_size: int = 224
    window_length: int = 3
    sphere_size: int = 1920
    action_type: str = "discrete"
    magnitudes: Tuple[float, ...] = field(default_factory=lambda: (10.0, 3.0, 1.0))
    mask_past_img_attn: bool = True
    n_blocks: int = 3
    pool_size: int = 4
    decoder_only: bool = False
    
    @property
    def crop_sizes(self) -> List[int]:
        """Compute crop sizes from n_levels, fovea_size, and sphere_size."""
        return crop_sizes_from_levels(self.n_levels, self.fovea_size, self.sphere_size)

    @staticmethod
    def from_dict(config_dict: Dict[str, Any]) -> 'VisualSearchAgentConfig':
        """Create config from dictionary with backwards compatibility."""
        # Handle backwards compatibility for crop_sizes -> n_levels, fovea_size, sphere_size
        config_dict = config_dict.copy()
        if 'crop_sizes' in config_dict and 'n_levels' not in config_dict:
            crop_sizes = config_dict.pop('crop_sizes')
            if isinstance(crop_sizes, list) and len(crop_sizes) > 0:
                config_dict['n_levels'] = len(crop_sizes)
                config_dict['fovea_size'] = crop_sizes[0]
                # Infer sphere_size from largest crop if reasonable
                if len(crop_sizes) > 1 and crop_sizes[-1] <= 2048:
                    config_dict['sphere_size'] = crop_sizes[-1] * 2
        
        # Filter out keys that don't exist in the dataclass
        valid_keys = {f.name for f in VisualSearchAgentConfig.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_keys}
        return VisualSearchAgentConfig(**filtered_dict)

    @staticmethod
    def from_json(json_path: Union[str, Path]) -> 'VisualSearchAgentConfig':
        """Load config from JSON file."""
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        return VisualSearchAgentConfig.from_dict(config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    def save(self, json_path: Union[str, Path]) -> None:
        """Save config to JSON file."""
        with open(json_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


@dataclass
class RobotAgentConfig:
    action_chunk_size: int
    proprio_dropout: float
    crop_size: int
    num_blocks: int = 3
    device: str = "cuda"
    relative_act: bool = False

    @staticmethod
    def from_dict(config_dict: Dict[str, Any]) -> 'RobotAgentConfig':
        """Create config from dictionary with backwards compatibility."""
        valid_keys = {f.name for f in RobotAgentConfig.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_keys}
        return RobotAgentConfig(**filtered_dict)

    @staticmethod
    def from_json(json_path: Union[str, Path]) -> 'RobotAgentConfig':
        """Load config from JSON file."""
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        return RobotAgentConfig.from_dict(config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    def save(self, json_path: Union[str, Path]) -> None:
        """Save config to JSON file."""
        with open(json_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


@dataclass
class EyeRobotAgentConfig:
    window_size: int = 224
    device: str = "cuda"
    freeze_encoder: bool = True
    n_levels: int = 4
    n_hand_levels: int = 2
    fovea_size: int = 224
    sphere_size: int = 1920
    action_chunk_size: int = 30
    window_length: int = 3
    proprio_dropout: float = 0.0
    proprio_hidden_dropout: float = 0.0
    num_blocks: int = 3
    use_learnable_joint_token: bool = True
    relative_act: bool = False
    eye_action_type: str = "continuous"
    eye_magnitudes: Tuple[float, ...] = field(default_factory=lambda: (10.0, 3.0, 1.0))
    pool_size: int = 2
    use_se3_actions: bool = False
    decoder_only: bool = False

    @property
    def crop_sizes(self) -> List[int]:
        """Compute crop sizes from n_levels, fovea_size, and sphere_size."""
        return crop_sizes_from_levels(self.n_levels, self.fovea_size, self.sphere_size)

    @staticmethod
    def from_dict(config_dict: Dict[str, Any]) -> 'EyeRobotAgentConfig':
        """Create config from dictionary with backwards compatibility."""
        # Handle backwards compatibility for crop_sizes -> n_levels, fovea_size, sphere_size
        config_dict = config_dict.copy()
        if 'crop_sizes' in config_dict and 'n_levels' not in config_dict:
            crop_sizes = config_dict.pop('crop_sizes')
            if isinstance(crop_sizes, list) and len(crop_sizes) > 0:
                config_dict['n_levels'] = len(crop_sizes)
                config_dict['fovea_size'] = crop_sizes[0]
                # Infer sphere_size from largest crop if reasonable
                if len(crop_sizes) > 1 and crop_sizes[-1] <= 2048:
                    config_dict['sphere_size'] = crop_sizes[-1] * 2
        
        valid_keys = {f.name for f in EyeRobotAgentConfig.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_keys}
        return EyeRobotAgentConfig(**filtered_dict)

    @staticmethod
    def from_json(json_path: Union[str, Path]) -> 'EyeRobotAgentConfig':
        """Load config from JSON file."""
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        return EyeRobotAgentConfig.from_dict(config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    def save(self, json_path: Union[str, Path]) -> None:
        """Save config to JSON file."""
        with open(json_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


class AgentConfigManager:
    """Utility class for managing agent configs with checkpoints."""
    
    @staticmethod
    def save_config_with_checkpoint(config: Union[VisualSearchAgentConfig, RobotAgentConfig, EyeRobotAgentConfig], 
                                   checkpoint_path: Union[str, Path]) -> None:
        """Save agent config along with model checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        
        # Determine config file name based on checkpoint name
        config_path = checkpoint_path.with_suffix('.json')
        config.save(config_path)
        
        # Also save config inside the checkpoint if it's a .pt file
        if checkpoint_path.suffix == '.pt':
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            checkpoint['config'] = config.to_dict()
            checkpoint['config_type'] = type(config).__name__
            torch.save(checkpoint, checkpoint_path)

    @staticmethod
    def load_config_from_checkpoint(checkpoint_path: Union[str, Path]) -> Optional[Union[VisualSearchAgentConfig, RobotAgentConfig, EyeRobotAgentConfig]]:
        """Load agent config from checkpoint file."""
        checkpoint_path = Path(checkpoint_path)
        
        # Try to load from checkpoint first
        if checkpoint_path.suffix == '.pt':
            try:
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                if 'config' in checkpoint and 'config_type' in checkpoint:
                    config_dict = checkpoint['config']
                    config_type = checkpoint['config_type']
                    
                    if config_type == 'VisualSearchAgentConfig':
                        return VisualSearchAgentConfig.from_dict(config_dict)
                    elif config_type == 'RobotAgentConfig':
                        return RobotAgentConfig.from_dict(config_dict)
                    elif config_type == 'EyeRobotAgentConfig':
                        return EyeRobotAgentConfig.from_dict(config_dict)
            except Exception:
                pass
        
        # Try to load from separate JSON file
        config_path = checkpoint_path.with_suffix('.json')
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config_dict = json.load(f)
                
                # Try to infer config type from dictionary keys
                if 'magnitudes' in config_dict and 'action_type' in config_dict:
                    return VisualSearchAgentConfig.from_dict(config_dict)
                elif 'eye_magnitudes' in config_dict and 'eye_action_type' in config_dict:
                    return EyeRobotAgentConfig.from_dict(config_dict)
                elif 'action_chunk_size' in config_dict and 'crop_size' in config_dict:
                    return RobotAgentConfig.from_dict(config_dict)
                    
            except Exception:
                pass
                
        return None