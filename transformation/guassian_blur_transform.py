from typing import List

import torch
from torch import Tensor
from typing_extensions import Dict
from torch.nn import functional as F

from transformation.abstract_transformation import AbstractTransformation


class GaussianBlurTransform(AbstractTransformation):
    """
    Applies Gaussian Blur to the image.
    It requires 1 parameter: sigma (blur intensity).
    Vectorized using grouped 2D convolutions to allow each image in the batch to have its own sigma.
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self._sigma = 'sigma'
        self.kernel_size = kernel_size

    def get_required_params_amount(self) -> int:
        return 1

    def configure_transformation(self, params: Tensor) -> Dict[str, Tensor]:
        """
        :param sigma: Tensor of shape [B]
        """
        sigma = params
        return {self._sigma: sigma}

    def transform2domain(self, params: Dict[str, Tensor]) -> Dict[str, Tensor]:
        # Maps [0, 1] to a valid sigma range, e.g., [0.1, 2.0]
        sigma = params[self._sigma] * 1.9 + 0.1
        return {self._sigma: sigma}

    def apply_transform(self, img: Tensor, params: Dict[str, Tensor]) -> Tensor:
        domain_params = self.transform2domain(params)
        sigma = domain_params[self._sigma]

        B, C, H, W = img.shape
        K = self.kernel_size
        half_k = K // 2

        # Create a 1D grid for the kernel
        x = torch.arange(-half_k, half_k + 1, device=img.device, dtype=img.dtype)  # [K]
        x = x.view(1, K).expand(B, K)  # [B, K]

        # Calculate 1D Gaussian kernel
        sigma = sigma.view(B, 1)
        kernel_1d = torch.exp(-0.5 * (x / sigma) ** 2)
        kernel_1d = kernel_1d / kernel_1d.sum(dim=1, keepdim=True)  # [B, K]

        # Outer product to get 2D Gaussian kernel
        kernel_2d = torch.bmm(kernel_1d.unsqueeze(2), kernel_1d.unsqueeze(1))  # [B, K, K]

        # Reshape kernel for grouped convolution: [B*C, 1, K, K]
        kernel_2d = kernel_2d.unsqueeze(1).expand(B, C, K, K).reshape(B * C, 1, K, K)

        # Reshape image and apply reflection padding
        img_reshaped = img.reshape(1, B * C, H, W)
        img_padded = F.pad(img_reshaped, (half_k, half_k, half_k, half_k), mode='reflect')

        # Apply convolution where groups=B*C allows different kernels per image
        blurred = F.conv2d(img_padded, kernel_2d, groups=B * C)

        return blurred.reshape(B, C, H, W)

    def get_identity_params(self) -> List[float]:
        return [0.9 / 1.9]