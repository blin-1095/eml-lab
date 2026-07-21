from typing import List

import torch
from torch import Tensor
from typing_extensions import Dict

from transformation.abstract_transformation import AbstractTransformation


class HorizontalFlipTransform(AbstractTransformation):
    """
    Applies a horizontal flip to the image.
    Requires 1 parameter: flip_probability.
    Uses a soft blend to maintain differentiability for the generator.
    """

    def __init__(self):
        super().__init__()
        self._flip_prob = "flip_prob"

    def get_required_params_amount(self) -> int:
        return 1

    def configure_transformation(self, params: Tensor) -> Dict[str, Tensor]:
        """
        :param params: Tensor of shape [B,]
        """
        return {self._flip_prob: params}

    def transform2domain(self, params: Dict) -> Dict:
        prob = torch.abs((params[self._flip_prob] - 0.5) * 2.0)
        return {self._flip_prob: prob}

    def apply_transform(self, img: Tensor, params: Dict[str, Tensor]) -> Tensor:
        transformed = self.transform2domain(params)
        p = transformed[self._flip_prob].view(-1, 1, 1, 1)
        flipped = torch.flip(img, dims=[3])

        # Soft blend: if p is 1, it's fully flipped. If p is 0, it's original.
        return (1.0 - p) * img + p * flipped

    def get_identity_params(self) -> List[float]:
        return [0.5]