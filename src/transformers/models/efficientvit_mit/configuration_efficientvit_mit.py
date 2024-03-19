# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" EfficientVitMit Transformer model configuration"""

from typing import List

from ...configuration_utils import PretrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)

EFFICIENTVIT_MIT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "mit-han-lab/efficientvit": "https://huggingface.co/mit-han-lab/efficientvit/resolve/main/config.json",
}


class EfficientVitMitConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`EfficientVitMitModel`]. It is used to instantiate a Swin
    Transformer v2 model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the Swin Transformer v2
    [mit-han-lab/efficientvit](https://huggingface.co/mit-han-lab/efficientvit)
    architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        width_list (:obj:`List[int]`, `optional`, defaults to [24, 48, 96, 192, 384]):
            The list of channel width for each stage.
        depth_list (:obj:`List[int]`, `optional`, defaults to [1, 3, 4, 4, 6]):
            The list of number of layers for each stage.
        in_channels (:obj:`int`, `optional`, defaults to 3):
            The number of input channels.
        dim (:obj:`int`, `optional`, defaults to 32):
            The number of output channels of the first stage.
        expand_ratio (:obj:`int`, `optional`, defaults to 4):
            The expansion ratio of the bottleneck layer.
        norm (:obj:`str`, `optional`, defaults to "bn2d"):
            The normalization layer used in the model.
        act_func (:obj:`str`, `optional`, defaults to "hswish"):
            The activation function used in the model.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

    Example:

    ```python
    >>> from transformers import EfficientVitMitConfig, EfficientVitMitModel

    >>> # Initializing a EfficientVitMit mit-han-lab/efficientvit style configuration
    >>> configuration = EfficientVitMitConfig()

    >>> # Initializing a model (with random weights) from the mit-han-lab/efficientvit style configuration
    >>> model = EfficientVitMitModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "efficientvit_mit"

    def __init__(
        self,
        width_list: List[int] = [24, 48, 96, 192, 384],
        depth_list: List[int] = [1, 3, 4, 4, 6],
        in_channels=3,
        dim=32,
        expand_ratio=4,
        norm="bn2d",
        act_func="hswish",
        initializer_range=0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.width_list = width_list
        self.depth_list = depth_list
        self.in_channels = in_channels
        self.dim = dim
        self.expand_ratio = expand_ratio
        self.norm = norm
        self.act_func = act_func
        self.initializer_range = initializer_range
