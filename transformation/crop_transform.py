from typing import List, Optional, Tuple
import torch
from torch import Tensor
from typing_extensions import Dict
from torch.nn import functional as F

from transformation.abstract_transformation import AbstractTransformation
from utils.bbox_utils import cxcywh_to_corners, corners_to_cxcywh, filter_dead_boxes

class CropTransform(AbstractTransformation):
    def __init__(self, start_with_identity=True):
        super().__init__()
        self._start_with_identity = start_with_identity

    def get_required_params_amount(self) -> int:
        return 4

    def configure_transformation(self, params: Tensor) -> Dict[str, Tensor]:
        return {
            'cx': params[:, 0],
            'cy': params[:, 1],
            'scale': params[:, 2],        # Replacing independent sx with Area Scale
            'aspect_ratio': params[:, 3]  # Replacing independent sy with Aspect Ratio
        }

    def transform2domain(self, params: Dict[str, Tensor]) -> Dict[str, Tensor]:
        if not self._start_with_identity:
            raise NotImplementedError("Crop transform with non-identity is not implemented yet.")

        # --- 1. THE SCALE (ZOOM) FIX ---
        # Raw 0.5 -> 1.0 (Identity).
        # Raw 0.0 or 1.0 -> 0.3 (Max Zoom In).
        # Using abs() guarantees healthy gradients and prevents zooming OUT (>1.0)
        zoom_intensity = torch.abs((params['scale'] - 0.5) * 2.0)
        scale = 1.0 - (zoom_intensity * 0.7)  # Maps perfectly to [0.3, 1.0]

        # --- 2. THE ASPECT RATIO FIX ---
        # Raw 0.5 -> 1.0 (Identity).
        # Bounds exactly to standard SimCLR [0.75, 1.33]
        # log(4/3) is approx 0.28768
        log_ar = (params['aspect_ratio'] - 0.5) * 0.57536
        ar = torch.exp(log_ar)

        # Compute affine sx and sy from Scale and Aspect Ratio
        sx = scale * torch.sqrt(ar)
        sy = scale / torch.sqrt(ar)

        # Hard clamp to 1.0 to mathematically guarantee PyTorch never hits reflection padding
        sx = torch.clamp(sx, max=1.0)
        sy = torch.clamp(sy, max=1.0)

        # --- 3. THE PAN FIX ---
        cx = (params['cx'] - 0.5) * 2.0 * torch.abs(1.0 - sx)
        cy = (params['cy'] - 0.5) * 2.0 * torch.abs(1.0 - sy)

        return {'cx': cx, 'cy': cy, 'sx': sx, 'sy': sy}

    def apply_transform(self, img: Tensor, params: Dict[str, Tensor], targets: Tensor) -> Tuple[Tensor, Tensor]:
        domain_params = self.transform2domain(params)
        B = img.shape[0]

        affine_matrices = torch.zeros(B, 2, 3, device=img.device, dtype=img.dtype)
        affine_matrices[:, 0, 0] = domain_params['sx']
        affine_matrices[:, 1, 1] = domain_params['sy']
        affine_matrices[:, 0, 2] = domain_params['cx']
        affine_matrices[:, 1, 2] = domain_params['cy']

        dimensions = [dim for dim in img.shape]
        grid = F.affine_grid(affine_matrices, dimensions, align_corners=False)
        cropped_img = F.grid_sample(img, grid, align_corners=False, padding_mode=self.padding_mode)

        cropped_targets = None
        # 1. Unpack the original corners
        x_corners, y_corners, valid_mask = cxcywh_to_corners(targets)

        # Reshape parameters for broadcasting: [B] -> [B, 1, 1]
        sx = domain_params['sx'].view(B, 1, 1)
        sy = domain_params['sy'].view(B, 1, 1)
        cx = domain_params['cx'].view(B, 1, 1)
        cy = domain_params['cy'].view(B, 1, 1)

        # 2. Calculate the top-left offset in normalized [0, 1] coordinates
        x_offset = 0.5 * (1.0 - sx + cx)
        y_offset = 0.5 * (1.0 - sy + cy)

        # Shift the corners by the offset, then scale them up by the zoom factor
        x_corners = (x_corners - x_offset) / sx
        y_corners = (y_corners - y_offset) / sy

        # 3. Clamp any boxes that go out of bounds
        cropped_targets = corners_to_cxcywh(targets, x_corners, y_corners, valid_mask)

        # Filter dead boxes that are not inside image bounds anymore
        targets = filter_dead_boxes(cropped_targets, valid_mask)

        return cropped_img, targets

    def get_identity_params(self) -> List[float]:
        # Perfectly centered logit biases for all 4 parameters
        return [0.5, 0.5, 0.5, 0.5]