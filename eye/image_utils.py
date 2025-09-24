import numpy as np


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