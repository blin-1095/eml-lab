from typing import List, Optional, Tuple

import torch
from torch import Tensor
from typing_extensions import Dict

from transformation.abstract_transformation import AbstractTransformation


class GaussianNoiseTransform(AbstractTransformation):
    """
    Adds Gaussian Noise to the image.
    Requires 1 parameter: noise_std (standard deviation).
    """

    def __init__(self):
        super().__init__()
        self._noise_std = "noise_std"

    def get_required_params_amount(self) -> int:
        return 1

    def configure_transformation(self, params: Tensor) -> Dict[str, Tensor]:
        """
        :param params: Tensor of shape [B, ]
        """
        return {self._noise_std: params}

    def transform2domain(self, params: Dict[str, Tensor]) -> Dict[str, Tensor]:
        # Limit max std deviation to a reasonable value for normalized image [0, 1]
        return {self._noise_std: params[self._noise_std] * 0.1}

    def apply_transform(self, img: Tensor, params: Dict[str, Tensor], targets: Tensor) -> Tuple[Tensor, Tensor]:
        std = self.transform2domain(params)[self._noise_std].view(-1, 1, 1, 1)
        noise = torch.randn_like(img) * std

        # Add noise and clamp back to standard image boundaries
        out = img + noise
        return torch.clamp(out, 0.0, 1.0), targets

    def get_identity_params(self) -> List[float]:
        return [0.0]