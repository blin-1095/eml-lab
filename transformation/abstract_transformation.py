from abc import ABC, abstractmethod

from torch import Tensor
from typing_extensions import Tuple, Dict, List, Optional


class AbstractTransformation(ABC):
    """
    Represents an abstract transformation that can be applied to images. This is an interface for transformations that
    can be applied to images, such as rotations, flips, color adjustments, etc.
    Given the parameters needed for the tranformation (such as degree for rotation), a transformation is created
    that when multiplied to the image yields the transformed image.
    """

    def __init__(self) -> None:
        self.padding_mode = 'border'
        super().__init__()

    @abstractmethod
    def get_required_params_amount(self) -> int:
        """
        Returns the amount of parameters required for the transformation. For example, a rotation transformation might
        require 1 parameter (the degree of rotation), while a color adjustment transformation might require 3
        parameters (the amount of adjustment for each color channel).
        """
        pass

    def transform2domain(self, params: Dict) -> Dict:
        """
        Transforms the given parameters from the [0, 1] range to the domain of the transformation. For example, for a
        rotation transformation, it might transform a parameter in the [0, 1] range to an angle in the [0, 360] degree
        range.
        :param params: The parameters in the [0, 1] range. The keys of the dict are the same as the keys returned by
            configure_transformation().
        :return: The transformed parameters in the domain of the transformation. The keys of the dict are the same as the
        keys returned by configure_transformation().
        """
        return params


    @abstractmethod
    def configure_transformation(self, params: Tensor) -> Dict[str, Tensor]:
        """
        Configures the transformation with the given parameters.
        :param args: The arguments required for the transformation. Could for example be degree for rotation.
        """
        pass

    @abstractmethod
    def apply_transform(self, img: Tensor, params: Dict, targets: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Applies the transformation to the given image.
        :param img: The images to be transformed
        :param params: The parameters required for the transformation, such as degree for rotation.
            The params dict is the output of self.configure_transformation()
        :param targets: The bounding boxes of the images to be transformed
        """
        pass

    @abstractmethod
    def get_identity_params(self) -> List[float]:
        """
        Returns a list of [0, 1] values that correspond to the identity mapping
        for this specific transformation.
        """
        pass
