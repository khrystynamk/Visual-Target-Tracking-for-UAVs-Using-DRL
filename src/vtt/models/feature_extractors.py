"""
Feature extractors for the asymmetric SAC policy.

Actor (DepthResNet): (B, S, 1, 224, 224) depth -> per-frame ResNet-18 -> flatten -> Linear -> 512
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
        self.frame_stack = image_space.shape[0]

        # Adapted for depth input
        resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        old_conv1 = resnet18.conv1
        new_conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            new_conv1.weight.copy_(old_conv1.weight.mean(dim=1, keepdim=True))
        resnet18.conv1 = new_conv1

        self.resnet = nn.Sequential(*list(resnet18.children())[:-1])

        # All layers trainable
        for param in self.resnet.parameters():
            param.requires_grad = True

        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.frame_stack * 512, features_dim),
            nn.ReLU(),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        images: (B, S, 224, 224) — stacked depth frames.
        Called by the actor as: extract_features(obs["image"], self.features_extractor)
        """
        b, s, h, w = images.shape

        x = images.reshape(b * s, 1, h, w)  # (B*S, 1, 224, 224)

        x = self.resnet(x)  # (B*S, 512, 1, 1)
        x = x.reshape(b, s, -1)  # (B, S, 512)
        x = torch.flatten(x, start_dim=1)  # (B, S*512)

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
