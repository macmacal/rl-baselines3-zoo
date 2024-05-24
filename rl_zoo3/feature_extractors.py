from typing import Type
from collections import OrderedDict
from abc import ABC, abstractmethod
import numpy as np

import gymnasium as gym
import torch as th
from gymnasium import spaces
from torch import nn


from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.preprocessing import is_image_space


class AbstractVisionExtractor(BaseFeaturesExtractor, ABC):
    """
    An abstract class for preparing pre-defined vision feature extractors,
    like CNN, VAE, ViT, M. Hopfield etc.

    It adds an extra nn.Linear layer with the selected activation function for
    scaling the output number of features.

    If the observation consists of stacked frames, these frames will be split
    and given to the feature extractor separately.

    :param observation_space:
    :param normalized_image: Whether to assume that the image is already normalized
        or not (this disables dtype and bounds checks): when True, it only checks that
        the space is a Box and has 3 dimensions.
        Otherwise, it checks that it has expected dtype (uint8)
        and bounds (values in [0, 255]).
    :param out_features_per_frame: Number of extracted features from a single image.
        This corresponds to the number of unit in the last layer.
        If the frames are stacked, the output dimension will be equal to
        out_features_per_frame * stacking_frames.
    :param linear_activation_fn: The activation function to use after the last layer.
    :param stacking_frames: The number of the stacked frames.
        Must be the same as the hyperparamter `frame_stack`.
    :param frame_channels: The number of color channels in the input images.
        Typically (1) for monochromatic, (3) for RGB and (4) for RGBD.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        out_features_per_frame: int,
        linear_activation_fn: Type[nn.Module],
        stacking_frames: int | None,
        frame_channels: int,
        normalized_image: bool,
    ):
        self._validate_params(observation_space, normalized_image, stacking_frames, frame_channels)
        if self.is_single_frame:
            stacking_frames = 1
        output_dim = out_features_per_frame * stacking_frames
        super().__init__(observation_space, output_dim)

        feature_extractor = self._prepare_feature_extractor()
        flatten = nn.Flatten()

        # Compute shape by doing one forward pass
        with th.no_grad():
            obs = th.as_tensor(observation_space.sample()[None]).float()
            if not self.is_single_frame:
                obs = th.split(obs, frame_channels, dim=1)[0]
            n_flatten = flatten(feature_extractor(obs)).squeeze().shape[0]

        # The output dimension of the linear layer is just for one frame.
        # The result will match the output_dim in the forward() method.
        linear = nn.Sequential(nn.Linear(n_flatten, out_features_per_frame), linear_activation_fn())
        self.fe_model = nn.Sequential(feature_extractor, flatten, linear)

    def _validate_params(
        self, observation_space: gym.Space, normalized_image: bool, stacking_frames: int | None, frame_channels: int
    ):
        assert isinstance(observation_space, spaces.Box), (
            "PreTrainedVisionExtractor must be used with a gym.spaces.Box ",
            f"observation space, not {observation_space}",
        )
        self.is_single_frame = stacking_frames is None or stacking_frames < 2
        if self.is_single_frame:
            assert is_image_space(observation_space, check_channels=True, normalized_image=normalized_image), (
                f"You should use a VisionExtractor only with images not with {observation_space}.\n"
                "If the `stackig_frames` is greater than 1, please set the Extractor params `stackig_frames`and `frame_channels` accordingly."
            )
        else:
            self.frame_channels = frame_channels
            n_channels = observation_space.shape[0] 
            divisions, modulo = np.divmod(n_channels, frame_channels)
            division = np.ceil(divisions)
            assert modulo == 0 and division == stacking_frames, (
                f"The number of passed channels ({n_channels}) is not matching stacked frames ({stacking_frames}) and "
                f"defined frame channels ({frame_channels}). Check `stacking_frames` and `frame_channels` configuration for given observation."
            )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        if self.is_single_frame:
            return self.fe_model(observations)

        # Split stacked frames and extract features from them
        frames = th.split(observations, self.frame_channels, dim=1)
        results = [None] * len(frames)
        for cnt, frame in enumerate(frames):
            results[cnt] = self.fe_model(frame)

        return th.cat(results, dim=1)

    @abstractmethod
    def _prepare_feature_extractor(self):
        raise NotImplementedError()


class CustomVisionExtractor(AbstractVisionExtractor):
    """
    Wraps user defined model for usage as visual feature extractor.

    Check AbstractVisionExtractor for more information.

    :param observation_space:
    :param custom_feature_extractor: the nn.Module defined by the user.
    :param out_features_per_frame: Number of extracted features from a single image.
        This corresponds to the number of unit in the last layer.
        If the frames are stacked, the output dimension will be equal to
        out_features_per_frame * stacking_frames.
    :param linear_activation_fn: The activation function to use after the last layer.
    :param stacking_frames: The number of the stacked frames.
        Must be the same as the hyperparamter `frame_stack`.
    :param frame_channels: The number of color channels in the input images.
        Typically (1) for monochromatic, (3) for RGB and (4) for RGBD.
    :param normalized_image: Whether to assume that the image is already normalized
        or not (this disables dtype and bounds checks): when True, it only checks that
        the space is a Box and has 3 dimensions.
        Otherwise, it checks that it has expected dtype (uint8)
        and bounds (values in [0, 255]).
    """

    def __init__(
        self,
        observation_space: gym.Space,
        custom_feature_extractor: Type[nn.Module],
        out_features_per_frame: int = 512,
        linear_activation_fn: Type[nn.Module] = nn.ReLU,
        stacking_frames: int = None,
        frame_channels: int = 3,
        normalized_image: bool = False,
    ):
        self._custom_model = custom_feature_extractor
        super().__init__(
            observation_space=observation_space,
            out_features_per_frame=out_features_per_frame,
            linear_activation_fn=linear_activation_fn,
            stacking_frames=stacking_frames,
            frame_channels=frame_channels,
            normalized_image=normalized_image,
        )

    def _prepare_feature_extractor(self):
        return self._custom_model


class PreTrainedVisionExtractor(AbstractVisionExtractor):
    """
    Loads a pre-trained model from the torchvision library as feature extractor.
    List of available models: https://pytorch.org/vision/main/models.html .

    Check AbstractVisionExtractor for more information.

    :param observation_space:
    :param model_name: the name of the model in the PascalCase format.
    :param weights_id: the name of the trained weights (torchvision API).
    :param cut_on_layer: the name of the layer to cut the head from the backbone.
    :param out_features_per_frame: Number of extracted features from a single image.
        This corresponds to the number of unit in the last layer.
        If the frames are stacked, the output dimension will be equal to
        out_features_per_frame * stacking_frames.
    :param linear_activation_fn: The activation function to use after the last layer.
    :param stacking_frames: The number of the stacked frames.
        Must be the same as the hyperparamter `frame_stack`.
    :param frame_channels: The number of color channels in the input images.
        Typically (1) for monochromatic, (3) for RGB and (4) for RGBD.
    :param normalized_image: Whether to assume that the image is already normalized
        or not (this disables dtype and bounds checks): when True, it only checks that
        the space is a Box and has 3 dimensions.
        Otherwise, it checks that it has expected dtype (uint8)
        and bounds (values in [0, 255]).
    """

    def __init__(
        self,
        observation_space: gym.Space,
        model_name: str = None,
        weights_id: str | None = None,
        cut_on_layer: str = None,
        out_features_per_frame: int = 512,
        linear_activation_fn: Type[nn.Module] = nn.ReLU,
        stacking_frames: int = None,
        frame_channels: int = 3,
        normalized_image: bool = False,
    ):
        self._import_torchvision()
        self._model_name = model_name
        self._weights_id = weights_id
        self._cut_on_layer = cut_on_layer
        super().__init__(
            observation_space=observation_space,
            out_features_per_frame=out_features_per_frame,
            linear_activation_fn=linear_activation_fn,
            stacking_frames=stacking_frames,
            frame_channels=frame_channels,
            normalized_image=normalized_image,
        )

    def _import_torchvision(self):
        try:
            self._thvision = __import__("torchvision")
        except ImportError:
            raise ImportError(
                "Can't use PreTrainedVisionExtractor without torchvision. Please install it (`pip install torchvision`)."
            )

    def _prepare_feature_extractor(self):
        pretrained_model = self._load_vision_model(self._model_name, self._weights_id)
        return self._cut_head_layers(pretrained_model, self._cut_on_layer)

    def _load_vision_model(self, model_name: str, weights_id: str | None = None) -> nn.Module:
        try:
            weights = weights_id if weights_id is None else self._thvision.models.get_weight(weights_id)
            model = self._thvision.models.get_model(model_name, weights=weights)

            # TODO add feature to unfreeze speecific layers
            for param in model.parameters():
                param.requires_grad = False

            return model
        except ValueError as e:
            raise ValueError(
                f"{e}.\nFailed to load the '{model_name}' model with '{weights_id}' weights. Ensure that the name is in "
                f"the PascalCase format and it is listed in https://pytorch.org/vision/main/models.html."
            )

    def _cut_head_layers(self, model: nn.Module, cut_layer: str) -> nn.Module:
        layers = OrderedDict()

        for layer_name, layer in model.named_children():
            if layer_name == cut_layer:
                break
            layers[layer_name] = layer

        return nn.Sequential(layers)
