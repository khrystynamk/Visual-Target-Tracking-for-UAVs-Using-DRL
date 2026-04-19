"""
Train a SAC agent for visual target tracking.

Usage:
  python scripts/train.py --config configs/drl/sac_depth.yaml
  python scripts/train.py --config configs/drl/sac_rgb.yaml
"""

import argparse
import os

import yaml
import torch
import wandb
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

from vtt.envs.tracking_env import TrackingEnv
from vtt.models.asymmetric_policy import AsymmetricSACPolicy


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def make_env(env_cfg: dict) -> TrackingEnv:
    return TrackingEnv(**env_cfg)


def main():
    parser = argparse.ArgumentParser(description="Train SAC tracking agent")
    parser.add_argument("--config", "-c", required=True, help="Path to YAML config")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    args = parser.parse_args()

    cfg = load_config(args.config)
    env_cfg = cfg["env"]
    sac_cfg = cfg["sac"]
    save_dir = cfg["save_dir"]

    os.makedirs(save_dir, exist_ok=True)

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    if not args.no_wandb:
        wandb.init(
            project=cfg["wandb"]["project"],
            name=cfg["wandb"]["name"],
            config=cfg,
        )

    env = make_env(env_cfg)
    eval_env = make_env(env_cfg)

    model = SAC(
        AsymmetricSACPolicy,
        env,
        learning_rate=sac_cfg["learning_rate"],
        buffer_size=sac_cfg["buffer_size"],
        batch_size=sac_cfg["batch_size"],
        learning_starts=sac_cfg["learning_starts"],
        train_freq=sac_cfg["train_freq"],
        gamma=sac_cfg["gamma"],
        tau=sac_cfg["tau"],
        verbose=1,
        device=device,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(save_dir, "best"),
        log_path=os.path.join(save_dir, "eval_logs"),
        eval_freq=cfg["eval_freq"],
        n_eval_episodes=cfg["eval_episodes"],
        deterministic=True,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=cfg["eval_freq"],
        save_path=os.path.join(save_dir, "checkpoints"),
        name_prefix="sac",
    )

    print(f"Starting training: {sac_cfg['total_timesteps']} timesteps")
    modality = "depth" if env_cfg["use_depth"] else "RGB"
    print(f"Input modality: {modality}")

    model.learn(
        total_timesteps=sac_cfg["total_timesteps"],
        callback=[eval_callback, checkpoint_callback],
        log_interval=10,
    )

    final_path = os.path.join(save_dir, "final_model")
    model.save(final_path)
    print(f"Final model saved to {final_path}")

    if not args.no_wandb:
        wandb.finish()

    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
