from typing import List, Optional, Tuple

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

    def get_required_params_amount(self) -> int:
        return 0

    def configure_transformation(self, params: Tensor) -> Dict[str, Tensor]:
        return {}

    def transform2domain(self, params: Dict) -> Dict:
        return {}

    def apply_transform(self, img: Tensor, params: Dict[str, Tensor], targets: Tensor) -> Tuple[Tensor, Tensor]:
        transformed = self.transform2domain(params)

        flipped_images = torch.flip(img, dims=[3])

        # Flip the targets
        flipped_targets=None
        flipped_targets = targets.clone()
        flipped_targets[..., 0] = 1.0 - flipped_targets[..., 0]

        return flipped_images, flipped_targets

    def get_identity_params(self) -> List[float]:
        return [0.5]