"""
Feature extractors for the asymmetric SAC policy.

Actor (DepthResNet): depth frames + bbox -> per-frame ResNet-18 -> concat with bbox -> Linear -> 512
Critic (CriticExtractor): receives relative 9D state of the target [pos, vel, acc]
"""

import torch
import torch.nn as nn
import torchvision.models as models
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym


class DepthResNet(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 512):
        super().__init__(observation_space, features_dim)

        image_space = observation_space["image"]
        total_channels = image_space.shape[0]
        self.frame_stack = total_channels // 3

        resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.resnet = nn.Sequential(*list(resnet18.children())[:-1])

        # All layers trainable
        for param in self.resnet.parameters():
            param.requires_grad = True

        self.linear = nn.Sequential(
            nn.Linear(self.frame_stack * 512 + 4, features_dim),
            nn.ReLU(),
        )

    def forward(self, images: torch.Tensor, bbox: torch.Tensor) -> torch.Tensor:
        """
        images: (B, S*3, 224, 224) — S colormapped depth frames (3 RGB channels each)
        bbox:   (B, 4) — relative bounding box [cx, cy, w, h]
        """
        b = images.shape[0]
        s = self.frame_stack

        x = images.reshape(b * s, 3, images.shape[2], images.shape[3])

        x = self.resnet(x)  # (B*S, 512, 1, 1)
        x = x.reshape(b, s, -1)  # (B, S, 512)
        x = torch.flatten(x, start_dim=1)  # (B, S*512)
        x = torch.cat([x, bbox], dim=1)  # (B, S*512 + 4)

        return self.linear(x)  # (B, 512)


class CriticExtractor(BaseFeaturesExtractor):
    """
    Passthrough extractor for the critic — receives raw state tensor.
    Called by the critic as: extract_features(obs["critic_state"], self.features_extractor)
    """

    def __init__(self, observation_space: gym.spaces.Dict):
        state_dim = observation_space["critic_state"].shape[0]
        super().__init__(observation_space, features_dim=state_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return state
