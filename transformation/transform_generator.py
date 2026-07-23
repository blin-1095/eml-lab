import torch
from torch import Tensor
from typing import Dict, Optional, List, Tuple, cast
from transformation.abstract_transformation import AbstractTransformation


class TransformGenerator():
    """
    Represents a deterministic Generator capable of transforming images.
    :param transformations: List of the transformations to be applied
    :param probability: Probability of each of the possible transformations to occur
    """
    def __init__(self, transformations: List[AbstractTransformation], probability: float = 0.0) -> None:
        self._transformations = transformations
        self._probability = probability

    def get_transformations(self) -> List[AbstractTransformation]:
        return self._transformations

    def transform(self, params: Tensor, images: Tensor, targets: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Apply transformation to images with given params.

        Input: Parameter tensor of shape (batch_size, parameters)
               Optional img tensor of shape (batch_size, C, H, W)
        Output: Transformed image tensor of shape (batch_size, C, H, W) OR list of parameter dicts.
        """

        current_images = images.clone()
        if targets is not None:
            targets = targets.clone()

        B = current_images.shape[0]
        raw_params = params
        param_idx = 0
        img_params: List[Dict[str, Tensor]] = []

        for transform in self._transformations:
            num_params = transform.get_required_params_amount()
            
            # Slice the parameters for the current batch: shape [B, num_params]
            # Check required since e.g. horizontal flip does not need any parameters
            if num_params > 0:
                transform_params = raw_params[:, param_idx: param_idx + num_params]
                param_idx += num_params
            else:
                # If a transform (like flip) requires 0 params, create an empty tensor
                transform_params = torch.empty((B, 0), device=current_images.device)

            # Select random images which get the current transformation
            apply_mask = torch.rand(B, device=current_images.device) < self._probability

            # Check if any images have been selected to have the mask applied
            if not apply_mask.any():
                img_params.append({})
                continue

            selected_params = transform_params[apply_mask]
            selected_images = current_images[apply_mask]
            selected_targets = targets[apply_mask] if targets is not None else None

            # configure and apply to selected sub-batch
            # If the transform requires exactly 1 parameter, squeeze it to shape [B]
            if num_params == 1:
                selected_params = selected_params.squeeze(-1)

            config_dict = transform.configure_transformation(selected_params)
            img_params.append(config_dict)

            transformed_images, transformed_targets = transform.apply_transform(
                cast(Tensor, selected_images), config_dict, selected_targets
            ) 

            # inject transformed images and targets back into main batch
            current_images[apply_mask] = transformed_images
            if targets is not None and transformed_targets is not None:
                targets[apply_mask] = transformed_targets


        assert current_images is not None
        return current_images, targets