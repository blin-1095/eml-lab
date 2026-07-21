from typing import List

import torch
from torch import Tensor
from typing_extensions import Dict

from transformation.abstract_transformation import AbstractTransformation


class GrayscaleTransform(AbstractTransformation):
    """
    Converts the image to grayscale (Color Drop in SimCLR).
    Requires 1 parameter: drop_probability.
    Vectorized and uses soft blending.
    """

    def __init__(self):
        super().__init__()
        self._drop_prob = "drop_prob"

    def get_required_params_amount(self) -> int:
        return 1

    def configure_transformation(self, params: Tensor) -> Dict[str, Tensor]:
        """
        :param params: Tensor of shape [B,]
        """
        return {self._drop_prob: params}

    def transform2domain(self, params: Dict) -> Dict:
        prob = torch.abs((params[self._drop_prob] - 0.5) * 2.0)
        return {self._drop_prob: prob}

    def apply_transform(self, img: Tensor, params: Dict[str, Tensor]) -> Tensor:
        B, C, H, W = img.shape
        transformed = self.transform2domain(params)
        p = transformed[self._drop_prob].view(B, 1, 1, 1)

        if C == 3:
            # Standard luminance preserving weights
            gray = (img[:, 0:1] * 0.2989 + img[:, 1:2] * 0.5870 + img[:, 2:3] * 0.1140)
            gray = gray.expand(-1, 3, -1, -1)
            return (1.0 - p) * img + p * gray

        return img

    def get_identity_params(self) -> List[float]:
        return [0.5]