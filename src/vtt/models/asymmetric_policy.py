"""
Asymmetric SAC policy.

Actor (A-DNN): depth image -> per-frame DepthResNet -> MLP -> action
Critic (C-DNN): privileged state + action -> MLP -> Q-value
"""

from typing import List, Tuple, Type

import gymnasium as gym
import torch as th
import torch.nn.functional as F
from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution
from stable_baselines3.common.policies import BaseModel, BasePolicy
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.sac.policies import SACPolicy
from torch import nn

from vtt.models.feature_extractors import DepthResNet, CriticExtractor

LOG_STD_MAX = 2
LOG_STD_MIN = -20


class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.output_dim = output_dim
        self.layer_1 = nn.Linear(input_dim, hidden_dim)
        self.layer_2 = nn.Linear(hidden_dim, hidden_dim)
        if output_dim > 0:
            self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: th.Tensor) -> th.Tensor:
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        if self.output_dim < 0:
            return x
        return self.output_layer(x)


class Actor(BasePolicy):
    """
    Actor network: depth images -> DepthResNet -> MLP -> action distribution.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        net_arch: List[int],
        features_extractor: nn.Module,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        **kwargs,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )

        self.net_arch = net_arch
        self.features_dim = features_dim
        self.activation_fn = activation_fn

        action_dim = get_action_dim(self.action_space)
        hidden_dim = net_arch[0] if len(net_arch) > 0 else features_dim
        self.latent_pi = MLP(features_dim, -1, hidden_dim)

        # Tanh saturation from the paper
        self.action_dist = SquashedDiagGaussianDistribution(action_dim)
        self.mu = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def get_action_dist_params(self, obs: th.Tensor):
        image_obs = obs["image"]
        features = self.features_extractor(image_obs)
        latent_pi = self.latent_pi(features)
        mean_actions = self.mu(latent_pi)
        log_std = th.clamp(self.log_std(latent_pi), LOG_STD_MIN, LOG_STD_MAX)
        return mean_actions, log_std, {}

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> th.Tensor:
        mean_actions, log_std, kwargs = self.get_action_dist_params(obs)
        return self.action_dist.actions_from_params(
            mean_actions, log_std, deterministic=deterministic, **kwargs
        )

    def action_log_prob(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        mean_actions, log_std, kwargs = self.get_action_dist_params(obs)
        return self.action_dist.log_prob_from_params(mean_actions, log_std, **kwargs)

    def _predict(
        self, observation: th.Tensor, deterministic: bool = False
    ) -> th.Tensor:
        return self.forward(observation, deterministic)


class Critic(BaseModel):
    """
    Critic network: privileged state + action -> Q-value.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        net_arch: List[int],
        features_extractor: nn.Module,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        n_critics: int = 2,
        share_features_extractor: bool = True,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )

        action_dim = get_action_dim(self.action_space)
        self.share_features_extractor = share_features_extractor
        self.n_critics = n_critics
        hidden_dim = net_arch[0] if len(net_arch) > 0 else 256

        self.q_networks = []
        for idx in range(n_critics):
            q_net = MLP(features_dim + action_dim, 1, hidden_dim)
            self.add_module(f"qf{idx}", q_net)
            self.q_networks.append(q_net)

    def forward(self, obs: th.Tensor, actions: th.Tensor):
        critic_state = obs["critic_state"]
        with th.set_grad_enabled(not self.share_features_extractor):
            features = self.features_extractor(critic_state)
        qvalue_input = th.cat([features, actions], dim=1)
        return tuple(q_net(qvalue_input) for q_net in self.q_networks)

    def q1_forward(self, obs: th.Tensor, actions: th.Tensor):
        with th.no_grad():
            features = self.features_extractor(obs["critic_state"])
        return self.q_networks[0](th.cat([features, actions], dim=1))


class AsymmetricSACPolicy(SACPolicy):
    """
    Asymmetric Soft Actor-Critic policy.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def make_actor(self, features_extractor=None) -> Actor:
        actor_kwargs = self.actor_kwargs.copy()

        features_extractor = DepthResNet(self.observation_space, features_dim=512)
        actor_kwargs = self._update_features_extractor(actor_kwargs, features_extractor)
        actor_kwargs["features_dim"] = 512

        return Actor(**actor_kwargs).to(self.device)

    def make_critic(self, features_extractor=None) -> Critic:
        critic_kwargs = self.critic_kwargs.copy()

        critic_extractor = CriticExtractor(self.observation_space)
        critic_kwargs = self._update_features_extractor(critic_kwargs, critic_extractor)
        critic_kwargs["features_dim"] = self.observation_space["critic_state"].shape[0]

        return Critic(**critic_kwargs).to(self.device)

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> th.Tensor:
        return self._predict(obs, deterministic=deterministic)

    def _predict(
        self, observation: th.Tensor, deterministic: bool = False
    ) -> th.Tensor:
        return self.actor(observation, deterministic)
