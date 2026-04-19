from stable_baselines3.sac.policies import SACPolicy

from vtt.models.feature_extractors import ActorExtractor, CriticExtractor


class AsymmetricSACPolicy(SACPolicy):
    """
    SAC policy with asymmetric actor/critic feature extractors.

    - Actor: ActorExtractor (depth CNN + tracker state)
    - Critic: CriticExtractor (privileged relative + tracker state)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            features_extractor_class=ActorExtractor,
            features_extractor_kwargs={"features_dim": 256},
            share_features_extractor=False,
            **kwargs,
        )

    def make_critic(self, features_extractor=None):
        critic_extractor = CriticExtractor(
            self.observation_space,
            features_dim=64,
        )
        critic_extractor = critic_extractor.to(self.device)
        return super().make_critic(features_extractor=critic_extractor)
