"""
Feature extractors for the asymmetric SAC policy.

Actor (DepthResNet): depth frames -> DeFM ResNet-18 (pretrained on depth) -> concat with bbox -> Linear -> 512
Critic (CriticExtractor): receives relative 9D state of the target [pos, vel, acc]
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym
from defm.utils.utils import preprocess_depth_image


class DepthResNet(BaseFeaturesExtractor):
    """
    DeFM ResNet-18 pretrained on depth images.
    """

    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 512):
        super().__init__(observation_space, features_dim)

        image_space = observation_space["image"]
        self.frame_stack = image_space.shape[0]

        self.defm = torch.hub.load(
            "leggedrobotics/defm:main",
            "defm_resnet18",
            pretrained=True,
            trust_repo=True,
        )

        for param in self.defm.parameters():
            param.requires_grad = True

        self.linear = nn.Sequential(
            nn.Linear(self.frame_stack * 512 + 4, features_dim),
            nn.ReLU(),
        )

    def _preprocess_batch(self, raw_depth_batch: torch.Tensor) -> torch.Tensor:
        """
        raw_depth_batch (B, H, W) — raw depth in meters
        (B, 3, H_out, W_out) — DeFM 3-channel metric format
        """
        preprocessed = []
        for i in range(raw_depth_batch.shape[0]):
            tensor = preprocess_depth_image(
                raw_depth_batch[i].cpu().numpy(), target_size=224
            )
            preprocessed.append(tensor.squeeze(0))
        return torch.stack(preprocessed).to(
            device=raw_depth_batch.device, dtype=torch.float32
        )

    def forward(self, images: torch.Tensor, bbox: torch.Tensor) -> torch.Tensor:
        b = images.shape[0]
        s = self.frame_stack

        flat = images.reshape(b * s, images.shape[2], images.shape[3])
        preprocessed = self._preprocess_batch(flat)  # (B*S, 3, H', W')

        def _defm_forward(x):
            return self.defm(x)["global_backbone"]

        feat = checkpoint(_defm_forward, preprocessed, use_reentrant=False)  # (B*S, 512)

        feat = feat.reshape(b, s * 512)  # (B, S*512)
        feat = torch.cat([feat, bbox], dim=1)  # (B, S*512 + 4)

        return self.linear(feat)  # (B, 512)


class CriticExtractor(BaseFeaturesExtractor):
    """
    Extractor for the critic — receives raw state tensor.
    """

    def __init__(self, observation_space: gym.spaces.Dict):
        state_dim = observation_space["critic_state"].shape[0]
        super().__init__(observation_space, features_dim=state_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return state
