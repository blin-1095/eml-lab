from typing import List
import torch
from torch import Tensor
from typing_extensions import Dict

from transformation.abstract_transformation import AbstractTransformation


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
            # Raw 0.0 -> -0.4 (Also a 40% Cutout Box due to sign math)
            self._size: (params[self._size] - 0.5) * 0.8
        }

    def apply_transform(self, img: Tensor, params: Dict[str, Tensor]) -> Tensor:
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

        steepness = 5.0
        mask_x = torch.sigmoid(steepness * (x_grid - (cx - size))) - torch.sigmoid(steepness * (x_grid - (cx + size)))
        mask_y = torch.sigmoid(steepness * (y_grid - (cy - size))) - torch.sigmoid(steepness * (y_grid - (cy + size)))

        box_mask = mask_x * mask_y

        return img * (1.0 - box_mask)

    def get_identity_params(self) -> List[float]:
        # Everything starts exactly at the center of the Sigmoid
        return [0.5, 0.5, 0.5]