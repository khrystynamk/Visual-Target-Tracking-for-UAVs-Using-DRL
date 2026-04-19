"""
Module for the feature extractors.

ActorExtractor: depth CNN (ResNet-18) + tracker state -> concatenated features
CriticExtractor: reads tracker and target states
"""

import torch
import torch.nn as nn
import torchvision.models as models
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym


class ActorExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        image_space = observation_space["image"]
        in_channels = image_space.shape[0]
        state_dim = observation_space["actor_state"].shape[0]

        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        old_conv1 = resnet.conv1
        new_conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        with torch.no_grad():
            avg_weight = old_conv1.weight.mean(dim=1, keepdim=True)  # (64, 1, 7, 7)
            new_conv1.weight.copy_(
                avg_weight.repeat(1, in_channels, 1, 1) / in_channels
            )
        resnet.conv1 = new_conv1

        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # (B, 512, 1, 1)

        for param in self.backbone.parameters():
            param.requires_grad = False

        # Unfreeze conv1 (modified) + layer4
        for param in new_conv1.parameters():
            param.requires_grad = True
        for param in resnet.layer4.parameters():
            param.requires_grad = True

        cnn_dim = 512
        self.projection = nn.Sequential(
            nn.Linear(cnn_dim + state_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: dict) -> torch.Tensor:
        images = observations["image"]
        state = observations["actor_state"]

        x = self.backbone(images)  # (B, 512, 1, 1)
        x = x.reshape(x.size(0), -1)  # (B, 512)
        x = torch.cat([x, state], dim=1)  # (B, 518)
        return self.projection(x)  # (B, 256)


class CriticExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 64):
        super().__init__(observation_space, features_dim)
        state_dim = observation_space["critic_state"].shape[0]  # 12
        self.net = nn.Sequential(
            nn.Linear(state_dim, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: dict) -> torch.Tensor:
        return self.net(observations["critic_state"])
