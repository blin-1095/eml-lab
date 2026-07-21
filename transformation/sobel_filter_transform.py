from typing import List

import torch
from torch import Tensor
from typing_extensions import Dict
from torch.nn import functional as F

from transformation.abstract_transformation import AbstractTransformation


class SobelFilterTransform(AbstractTransformation):
    """
    Applies Sobel Filtering to detect edges and blends it with the original image.
    Requires 1 parameter: intensity (blend amount).
    """

    def __init__(self):
        super().__init__()
        self._intensity = "intensity"

    def get_required_params_amount(self) -> int:
        return 1

    def configure_transformation(self, params: Tensor) -> Dict[str, Tensor]:
        """
        :param params: Tensor of shape [B,]
        """
        return {self._intensity: params}

    def apply_transform(self, img: Tensor, params: Dict[str, Tensor]) -> Tensor:
        intensity = params[self._intensity].view(-1, 1, 1, 1)
        B, C, H, W = img.shape

        # Convert to grayscale first for Sobel application
        if C == 3:
            gray = (img[:, 0:1] * 0.2989 + img[:, 1:2] * 0.5870 + img[:, 2:3] * 0.1140)
        else:
            gray = img

        # Define 3x3 Sobel kernels
        sobel_x = torch.tensor([[-1., 0., 1.],
                                [-2., 0., 2.],
                                [-1., 0., 1.]], device=img.device, dtype=img.dtype).view(1, 1, 3, 3)

        sobel_y = torch.tensor([[-1., -2., -1.],
                                [0., 0., 0.],
                                [1., 2., 1.]], device=img.device, dtype=img.dtype).view(1, 1, 3, 3)

        # Apply grouped convolution across batch
        gx = F.conv2d(gray, sobel_x, padding=1)
        gy = F.conv2d(gray, sobel_y, padding=1)

        # Calculate gradient magnitude
        magnitude = torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)

        # Approximate normalization to [0, 1] range based on theoretical max
        magnitude = magnitude / 4.0
        magnitude = torch.clamp(magnitude, 0.0, 1.0)

        if C == 3:
            magnitude = magnitude.expand(-1, 3, -1, -1)

        return (1.0 - intensity) * img + intensity * magnitude

    def get_identity_params(self) -> List[float]:
        return [0.0]