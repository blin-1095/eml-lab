import math
from typing import List

#from statsmodels.tsa.arima import params
from typing_extensions import Dict

import torch
import torchvision.transforms.functional as TF

from torch import Tensor
from torch.nn import functional as F

from transformation.abstract_transformation import AbstractTransformation


class RotationTransform(AbstractTransformation):

    def __init__(self, min_rotation: float = 0.0):
        """
        :param min_rotation: restricts the rotation from [min_rotation, 360 - min_rotation]. This is used to keep the
            model from learning the identity. I.e. force the model to rotate the image by at least min_rotation degrees.
        """
        super().__init__()
        self._angle = 'angle'
        self._min_rotation = min_rotation
        self.max_angle = 180

    def transform2domain(self, params: Dict[str, Tensor]) -> Dict[str, Tensor]:
        # Raw 0.5 -> 0.0 degrees (Identity)
        # Raw 1.0 -> +max_angle
        # Raw 0.0 -> -max_angle
        angle = (params['angle'] - 0.5) * 2.0 * self.max_angle
        return {'angle': angle}

    def get_required_params_amount(self) -> int:
        return 1

    def configure_transformation(self, params: Tensor) -> Dict[str, Tensor]:
        """
        Configures the rotation transformation with the given parameters.
        :param rotation: The angle of rotation as value between [0,1]. 0 corresponds to 0 degrees and 1 corresponds to 360
            degrees.
        :type rotation: Tensor of shape [B, 1].
        """
        rotation = params
        return {self._angle: rotation}

    def apply_transform(self, img: Tensor, params: Dict) -> Tensor:
        """
        Applies the rotation transformation to the batched images via grid sampling.
        :param img: 4D Tensor of shape [B, C, H, W]
        :param params: Dictionary containing batched parameter Tensors
        """
        batch_size = img.shape[0]

        # Convert [0, 1] range to radians [0, 2π]
        angles_deg = self.transform2domain(params)[self._angle]
        angles_rad = angles_deg / 180.0 * math.pi

        # For counter-clockwise rotation, the sampling grid requires the inverse rotation matrix
        cos_a = torch.cos(-angles_rad)
        sin_a = torch.sin(-angles_rad)

        # Construct the batch of affine matrices: shape [B, 2, 3]
        affine_matrices = torch.zeros(batch_size, 2, 3, device=img.device, dtype=img.dtype)
        affine_matrices[:, 0, 0] = cos_a
        affine_matrices[:, 0, 1] = -sin_a
        affine_matrices[:, 1, 0] = sin_a
        affine_matrices[:, 1, 1] = cos_a

        # Generate the grid and sample the whole batch at once
        dimensions = [dim for dim in img.shape]
        grid = F.affine_grid(affine_matrices, dimensions, align_corners=False)
        rotated_img = F.grid_sample(img, grid, align_corners=False, padding_mode=self.padding_mode)

        return rotated_img

    def get_identity_params(self) -> List[float]:
        return [0.5]
