from typing import List, Optional, Tuple
import torch
from torch import Tensor
from typing_extensions import Dict

from transformation.abstract_transformation import AbstractTransformation
from utils.bbox_utils import cxcywh_to_corners


class CutoutTransform(AbstractTransformation):
    def __init__(self, start_with_identity=True):
        super().__init__()
        self._cx = "cx"
        self._cy = "cy"
        self._size = "size"
        self._start_with_identity = start_with_identity

    def get_required_params_amount(self) -> int:
        return 3

    def configure_transformation(self, params: Tensor) -> Dict[str, Tensor]:
        return {
            self._cx: params[:, 0],
            self._cy: params[:, 1],
            self._size: params[:, 2]
        }

    def transform2domain(self, params: Dict[str, Tensor]) -> Dict[str, Tensor]:
        if not self._start_with_identity:
            raise NotImplementedError("Cutout with non-identity is not implemented yet.")

        return {
            # Map exactly to [-0.8, 0.8]
            self._cx: (params[self._cx] - 0.5) * 1.6,
            self._cy: (params[self._cy] - 0.5) * 1.6,

            # --- THE PERFECT 0.5 MAPPING ---
            # Raw 0.5 -> 0.0 (Invisible Cutout)
            # Raw 1.0 -> +0.4 (40% Cutout Box)
            # Raw 0.0 -> -0.4 (Also a 40% Cutout Box correctly inverted)
            self._size: torch.abs((params[self._size] - 0.5) * 0.8)
        }

    def apply_transform(self, img: Tensor, params: Dict[str, Tensor], targets: Tensor) -> Tuple[Tensor, Tensor]:
        domain_params = self.transform2domain(params)
        B, C, H, W = img.shape

        cx = domain_params[self._cx].view(-1, 1, 1, 1)
        cy = domain_params[self._cy].view(-1, 1, 1, 1)
        size = domain_params[self._size].view(-1, 1, 1, 1)

        y_grid, x_grid = torch.meshgrid(
            torch.linspace(-1, 1, H, device=img.device, dtype=img.dtype),
            torch.linspace(-1, 1, W, device=img.device, dtype=img.dtype),
            indexing='ij'
        )
        x_grid = x_grid.view(1, 1, H, W).expand(B, 1, H, W)
        y_grid = y_grid.view(1, 1, H, W).expand(B, 1, H, W)

        cut_xmin = cx - size
        cut_xmax = cx + size
        cut_ymin = cy - size
        cut_ymax = cy + size

        box_mask = (x_grid >= cut_xmin) & (x_grid <= cut_xmax) & \
                   (y_grid >= cut_ymin) & (y_grid <= cut_ymax)
        
        cropped_img = img * (~box_mask).float()

        # Handle edge case where object of interest might be 100% cropped
        targets = targets.clone()
        x_corners, y_corners, valid_mask = cxcywh_to_corners(targets)
        
        # Get min/max of the bounding box
        obj_xmin, _ = torch.min(x_corners, dim=-1)
        obj_xmax, _ = torch.max(x_corners, dim=-1)
        obj_ymin, _ = torch.min(y_corners, dim=-1)
        obj_ymax, _ = torch.max(y_corners, dim=-1)

        # Convert Cutout [-1, 1] bounds to [0, 1] bounds to match YOLO targets
        # .view(B, 1) allows it to broadcast against the N targets
        cut_xmin = ((cx - size + 1.0) / 2.0).view(B, 1)
        cut_xmax = ((cx + size + 1.0) / 2.0).view(B, 1)
        cut_ymin = ((cy - size + 1.0) / 2.0).view(B, 1)
        cut_ymax = ((cy + size + 1.0) / 2.0).view(B, 1)

        # Check if the object is 100% inside the cutout square
        is_swallowed = (obj_xmin >= cut_xmin) & (obj_xmax <= cut_xmax) & \
                        (obj_ymin >= cut_ymin) & (obj_ymax <= cut_ymax)
        
        # Kill the box only if it was valid AND it got swallowed
        dead_mask = valid_mask & is_swallowed
        
        # Set everything, including confidence, to 0 and set class to invalid object
        targets[dead_mask] = 0.0
        targets[..., 5][dead_mask] = -1.0

        return cropped_img, targets

    def get_identity_params(self) -> List[float]:
        # Everything starts exactly at the center of the Sigmoid
        return [0.5, 0.5, 0.5]