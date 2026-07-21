import math
from typing import List
import torch
from torch import Tensor
from typing_extensions import Dict

from transformation.abstract_transformation import AbstractTransformation

class ColorDistortionTransform(AbstractTransformation):
    """
    Applies Brightness, Contrast, Saturation, and Hue adjustments.
    Fully vectorized over the batch dimension. Hue uses a differentiable YIQ color space approximation.
    """

    def __init__(self):
        super().__init__()

    def get_required_params_amount(self) -> int:
        return 4

    def configure_transformation(self, params: Tensor) -> Dict[str, Tensor]:
        return {
            'brightness': params[:, 0],
            'contrast': params[:, 1],
            'saturation': params[:, 2],
            'hue': params[:, 3]
        }

    def transform2domain(self, params: Dict[str, Tensor]) -> Dict[str, Tensor]:
        # Maps generator (0, 1) parameters EXACTLY into realistic jitter bounds.
        # No clamping means gradients never die!
        return {
            'brightness': params['brightness'] * 1.6 + 0.2,  # Maps exactly to [0.2, 1.8]
            'contrast': params['contrast'] * 1.6 + 0.2,      # Maps exactly to [0.2, 1.8]
            'saturation': params['saturation'] * 1.6 + 0.2,  # Maps exactly to [0.2, 1.8]
            'hue': params['hue'] - 0.5                       # Maps exactly to [-0.5, 0.5]
        }

    def apply_transform(self, img: Tensor, params: Dict[str, Tensor]) -> Tensor:
        domain_params = self.transform2domain(params)
        B, C, H, W = img.shape

        b_factor = domain_params['brightness'].view(B, 1, 1, 1)
        c_factor = domain_params['contrast'].view(B, 1, 1, 1)

        out = img.clone()

        # 1. Brightness
        out = out * b_factor

        # 2. Contrast
        gray_mean = out.mean(dim=[-3, -2, -1], keepdim=True)
        out = (out - gray_mean) * c_factor + gray_mean

        if C == 3:
            s_factor = domain_params['saturation'].view(B, 1, 1, 1)

            # 3. Saturation (blend with grayscale)
            gray = (out[:, 0:1] * 0.2989 + out[:, 1:2] * 0.5870 + out[:, 2:3] * 0.1140)
            out = (out - gray) * s_factor + gray

            # 4. Hue (Vectorized YIQ approximation)
            rgb2yiq = torch.tensor([[0.299, 0.587, 0.114],
                                    [0.596, -0.274, -0.321],
                                    [0.211, -0.523, 0.311]], device=img.device, dtype=img.dtype)
            yiq2rgb = torch.tensor([[1.0, 0.956, 0.621],
                                    [1.0, -0.272, -0.647],
                                    [1.0, -1.107, 1.705]], device=img.device, dtype=img.dtype)

            out_flat = out.permute(0, 2, 3, 1).reshape(-1, 3)
            yiq = torch.matmul(out_flat, rgb2yiq.T)

            cos_h = torch.cos(domain_params['hue']).view(B, 1, 1, 1).expand(B, 1, H, W).reshape(-1, 1)
            sin_h = torch.sin(domain_params['hue']).view(B, 1, 1, 1).expand(B, 1, H, W).reshape(-1, 1)

            y, i, q = yiq[:, 0:1], yiq[:, 1:2], yiq[:, 2:3]

            i_new = i * cos_h - q * sin_h
            q_new = i * sin_h + q * cos_h

            yiq_new = torch.cat([y, i_new, q_new], dim=1)
            out_rgb = torch.matmul(yiq_new, yiq2rgb.T)
            out = out_rgb.view(B, H, W, 3).permute(0, 3, 1, 2)

        return out

    def get_identity_params(self) -> List[float]:
        # Raw 0.5 maps perfectly to physical 1.0 (or 0.0 for hue)
        return [0.5, 0.5, 0.5, 0.5]