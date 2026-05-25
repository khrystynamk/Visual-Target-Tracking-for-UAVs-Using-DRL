"""
Asymmetric PPO policy.

Actor  (A-DNN): depth image + bbox -> DepthResNet -> action distribution
Critic (C-DNN): privileged state   -> MLP         -> V(s)

The critic uses ``obs["critic_state"]`` directly; depth features never feed the
value head, so value-loss gradients do not flow through the image trunk.
"""

from typing import Tuple

import gymnasium as gym
import torch as th
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from vtt.models.feature_extractors import DepthResNet


class DepthResNetExtractor(BaseFeaturesExtractor):
    """
    Wraps DepthResNet so SB3 can call it as features_extractor(obs_dict).
    """

    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        self.backbone = DepthResNet(observation_space, features_dim=features_dim)

    def forward(self, observations: dict) -> th.Tensor:
        return self.backbone(observations["image"], observations["bbox"])


class AsymmetricPPOPolicy(ActorCriticPolicy):
    """
    Asymmetric actor-critic policy for PPO.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        action_space: gym.spaces.Space,
        lr_schedule,
        critic_hidden_dim: int = 256,
        **kwargs,
    ):
        kwargs.setdefault("features_extractor_class", DepthResNetExtractor)
        kwargs.setdefault("features_extractor_kwargs", {"features_dim": 512})
        kwargs.setdefault("share_features_extractor", True)

        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

        critic_state_dim = observation_space["critic_state"].shape[0]
        self.privileged_value_net = nn.Sequential(
            nn.Linear(critic_state_dim, critic_hidden_dim),
            nn.Tanh(),
            nn.Linear(critic_hidden_dim, critic_hidden_dim),
            nn.Tanh(),
            nn.Linear(critic_hidden_dim, 1),
        )

    def _value_from_obs(self, obs: dict) -> th.Tensor:
        return self.privileged_value_net(obs["critic_state"])

    def forward(
        self, obs: dict, deterministic: bool = False
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        features = self.extract_features(obs)
        latent_pi = self.mlp_extractor.forward_actor(features)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        values = self._value_from_obs(obs)
        actions = actions.reshape((-1, *self.action_space.shape))
        return actions, values, log_prob

    def evaluate_actions(
        self, obs: dict, actions: th.Tensor
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        features = self.extract_features(obs)
        latent_pi = self.mlp_extractor.forward_actor(features)
        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        values = self._value_from_obs(obs)
        return values, log_prob, entropy

    def predict_values(self, obs: dict) -> th.Tensor:
        return self._value_from_obs(obs)
