import torch
from torch import Tensor
from typing import Tuple

def cxcywh_to_corners(targets: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Extracts the 4 corners of bounding boxes from YOLO format targets.
    Expects targets in shape [B, N, C] where the first 4 columns are [cx, cy, w, h].
    
    Returns:
        x_corners: Tensor of shape [B, N, 4] containing X coordinates of corners.
        y_corners: Tensor of shape [B, N, 4] containing Y coordinates of corners.
        valid_mask: Boolean Tensor of shape [B, N] masking out empty padding boxes.
    """
    cx = targets[..., 0]
    cy = targets[..., 1]
    w  = targets[..., 2]
    h  = targets[..., 3]

    # Mask out empty padding boxes (w > 0 and h > 0)
    valid_mask = (w > 0) & (h > 0)

    dx = w / 2.0
    dy = h / 2.0

    # Create corners: Top-Left, Top-Right, Bottom-Right, Bottom-Left
    x_corners = torch.stack([cx - dx, cx + dx, cx + dx, cx - dx], dim=-1)
    y_corners = torch.stack([cy - dy, cy - dy, cy + dy, cy + dy], dim=-1)

    return x_corners, y_corners, valid_mask


def corners_to_cxcywh(targets: Tensor, x_corners: Tensor, y_corners: Tensor, valid_mask: Tensor) -> Tensor:
    """
    Converts modified corners back into YOLO [cx, cy, w, h] format, 
    clamping them to image boundaries [0, 1].
    
    Args:
        targets: Original targets tensor to clone.
        x_corners: Modified X coordinates of shape [B, N, 4].
        y_corners: Modified Y coordinates of shape [B, N, 4].
        valid_mask: Boolean Tensor of shape [B, N] to only update valid boxes.
        
    Returns:
        new_targets: Updated targets tensor.
    """
    new_targets = targets.clone()

    # Find the new axis-aligned bounds by getting min/max of the 4 modified corners
    min_x, _ = torch.min(x_corners, dim=-1)
    max_x, _ = torch.max(x_corners, dim=-1)
    min_y, _ = torch.min(y_corners, dim=-1)
    max_y, _ = torch.max(y_corners, dim=-1)

    # Clamp coordinates to ensure boxes don't exceed image boundaries
    min_x = torch.clamp(min_x, 0.0, 1.0)
    max_x = torch.clamp(max_x, 0.0, 1.0)
    min_y = torch.clamp(min_y, 0.0, 1.0)
    max_y = torch.clamp(max_y, 0.0, 1.0)

    # Convert back to cx, cy, w, h
    new_cx = (min_x + max_x) / 2.0
    new_cy = (min_y + max_y) / 2.0
    new_w  = max_x - min_x
    new_h  = max_y - min_y

    # Update the cloned targets tensor ONLY for valid boxes
    new_targets[..., 0] = torch.where(valid_mask, new_cx, new_targets[..., 0])
    new_targets[..., 1] = torch.where(valid_mask, new_cy, new_targets[..., 1])
    new_targets[..., 2] = torch.where(valid_mask, new_w,  new_targets[..., 2])
    new_targets[..., 3] = torch.where(valid_mask, new_h,  new_targets[..., 3])

    return new_targets

def filter_dead_boxes(targets: Tensor, valid_mask: Tensor, min_size: float = 0.001) -> Tensor:
    """
    Ensures any box that collapsed to zero/negative dimensions or fell out of 
    bounds during augmentation is safely invalidated (-1).
    """
    new_targets = targets.clone()
    new_w = new_targets[..., 2]
    new_h = new_targets[..., 3]

    undead_boxes_mask = (new_w > min_size) & (new_h > min_size) & valid_mask
    dead_mask = ~undead_boxes_mask

    # Zero out coordinates/confidence and set class index to -1
    new_targets[dead_mask] = 0.0
    if new_targets.shape[-1] >= 6:
        new_targets[..., 5][dead_mask] = -1.0  # Assuming class is at index 5
        
    return new_targets