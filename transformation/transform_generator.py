from typing import Dict, Optional, List, Union, cast

import torch
from torch import Tensor, nn

from transformation.abstract_transformation import AbstractTransformation


class TransformGenerator(nn.Module):
    """
    Represents a deterministic Generator capable of transforming images.
    Learns to apply transformations by directly mapping noise to parameters.
    """
    def __init__(self, transformations: List[AbstractTransformation], noise_dim: int = 8) -> None:
        super().__init__()
        self._transformations = transformations
        self.__noise_dim = 8
        self._noise_dim = torch.tensor([self.__noise_dim])
        self._output_dim = sum([t.get_required_params_amount() for t in transformations])

    def get_transformations(self) -> List[AbstractTransformation]:
        return self._transformations

    def transform(self, noise: Tensor, images: Optional[Tensor] = None) -> Union[Tensor, List[Dict[str, Tensor]]]:
        """
        Apply transformation to images with given noise.

        Input: Noise tensor of shape (batch_size, noise_dim)
               and optional img tensor of shape (batch_size, C, H, W).
        Output: Transformed image tensor of shape (batch_size, C, H, W) OR list of parameter dicts.
        """

        return_params = images is None

        # Predict batched parameters. Output shape: [B, output_dim]
        raw_params = self._network(noise)

        current_images = images
        param_idx = 0
        img_params: List[Dict[str, Tensor]] = []

        for transform in self._transformations:
            num_params = transform.get_required_params_amount()

            # Slice the parameters for the current batch: shape [B, num_params]
            transform_params = raw_params[:, param_idx: param_idx + num_params]
            param_idx += num_params

            # If the transform requires exactly 1 parameter, squeeze it to shape [B]
            if num_params == 1:
                transform_params = transform_params.squeeze(-1)

            config_dict = transform.configure_transformation(transform_params)
            img_params.append(config_dict)

            if not return_params:
                # Apply transformation to the entire batch instantly
                current_img = transform.apply_transform(cast(Tensor, current_images), config_dict)

        if return_params:
            return img_params

        assert current_images is not None
        return current_images