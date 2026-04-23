"""
Train a SAC agent for visual target tracking.

Usage:
  python scripts/train.py --config configs/drl/sac_depth.yaml
  python scripts/train.py --config configs/drl/sac_depth.yaml --n-envs 4
"""

import argparse
import os
import signal
import subprocess

import numpy as np
import yaml
import torch
import wandb
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from vtt.envs.tracking_env import TrackingEnv
from vtt.models.asymmetric_policy import AsymmetricSACPolicy

BASE_PORT = 41451
PORT_STEP = 10


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def make_env_fn(env_cfg: dict, api_port: int = 41451):
    """
    Return a callable that creates a Monitor-wrapped TrackingEnv.
    """

    def _init():
        env = TrackingEnv(**env_cfg, api_port=api_port)
        return Monitor(env)

    return _init


def make_single_env(env_cfg: dict, api_port: int = 41451):
    """
    Create a single Monitor-wrapped env.
    """
    env = TrackingEnv(**env_cfg, api_port=api_port)
    return Monitor(env)


def make_train_env(env_cfg: dict, n_envs: int):
    """
    Create training env(s): SubprocVecEnv if n_envs > 1, else DummyVecEnv.
    """
    if n_envs > 1:
        ports = [BASE_PORT + i * PORT_STEP for i in range(n_envs)]
        print(f"Creating {n_envs} parallel training envs on ports: {ports}")
        return SubprocVecEnv([make_env_fn(env_cfg, port) for port in ports])
    else:
        return DummyVecEnv([make_env_fn(env_cfg, BASE_PORT)])


def make_eval_env(env_cfg: dict, n_envs: int):
    """
    Create eval env on a dedicated AirSim port (separate from training).
    """
    eval_port = BASE_PORT + n_envs * PORT_STEP
    print(f"Eval env on port {eval_port}")
    return DummyVecEnv([make_env_fn(env_cfg, eval_port)])


def main():
    parser = argparse.ArgumentParser(description="Train SAC tracking agent")
    parser.add_argument("--config", "-c", required=True, help="Path to YAML config")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show live depth + bbox window during training",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Disable UE viewport rendering (faster)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint .zip, or 'auto' to download latest from R2",
    )
    parser.add_argument(
        "--r2-sync",
        action="store_true",
        help="Enable periodic checkpoint upload to Cloudflare R2",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="R2 run identifier (default: wandb name from config)",
    )
    parser.add_argument(
        "--image-monitor",
        action="store_true",
        help="Log depth image stats + sample frames to disk",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=1,
        help="Number of parallel AirSim environments (default: 1)",
    )
    args = parser.parse_args()

    # Convert SIGTERM to KeyboardInterrupt so the finally block runs
    def _sigterm_handler(_sig, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _sigterm_handler)

    cfg = load_config(args.config)
    env_cfg = cfg["env"]
    sac_cfg = cfg["sac"]
    save_dir = cfg["save_dir"]
    run_id = args.run_id or cfg["wandb"]["name"]

    os.makedirs(save_dir, exist_ok=True)

    # Auto-resume from R2
    if args.resume == "auto":
        resume_dir = os.path.join(save_dir, "_r2_resume")
        os.makedirs(resume_dir, exist_ok=True)
        r2_base = f"s3://vtt-uav-artifacts/runs/{run_id}/latest"
        model_dest = os.path.join(resume_dir, "model.zip")

        result = subprocess.run(
            ["aws", "--profile", "r2", "s3", "cp", f"{r2_base}/model.zip", model_dest],
            capture_output=True,
            timeout=120,
        )
        if result.returncode == 0:
            args.resume = model_dest
            replay_dest = os.path.join(resume_dir, "model_replay_buffer.pkl")
            rb = subprocess.run(
                [
                    "aws",
                    "--profile",
                    "r2",
                    "s3",
                    "cp",
                    f"{r2_base}/replay_buffer.pkl",
                    replay_dest,
                ],
                capture_output=True,
                timeout=300,
            )
            if rb.returncode == 0:
                print("Auto-resume: downloaded model + replay buffer from R2")
            else:
                print("Auto-resume: downloaded model from R2 (no replay buffer found)")
        else:
            print("Auto-resume: no checkpoint found in R2, starting fresh")
            args.resume = None

    if args.no_render:
        print("Note: --no-render requires ViewMode: NoDisplay in settings.json")

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    if not args.no_wandb:
        wandb_kwargs = {
            "project": cfg["wandb"]["project"],
            "name": cfg["wandb"]["name"],
            "config": cfg,
        }
        wandb.init(**wandb_kwargs)

    if args.show:
        env_cfg["show_cv"] = True

    n_envs = args.n_envs
    env = make_train_env(env_cfg, n_envs)
    eval_env = make_eval_env(env_cfg, n_envs)

    print(f"Training with {n_envs} env(s)")

    if args.resume:
        print(f"Resuming from {args.resume}")
        model = SAC.load(
            args.resume,
            env=env,
            device=device,
        )
        replay_path = args.resume.replace(".zip", "_replay_buffer.pkl")
        if os.path.exists(replay_path):
            model.load_replay_buffer(replay_path)
            print(f"Loaded replay buffer from {replay_path}")
    else:
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

    checkpoint_callback = CheckpointCallback(
        save_freq=1000,
        save_path=os.path.join(save_dir, "checkpoints"),
        name_prefix="sac",
    )

    r2_callback = None
    if args.r2_sync:
        from vtt.callbacks.r2_sync import R2SyncCallback

        r2_callback = R2SyncCallback(
            run_id=run_id,
            save_dir=save_dir,
            upload_freq=cfg.get("r2_upload_freq", 5000),
        )

    callbacks = [checkpoint_callback]
    if not args.no_wandb:
        callbacks.append(WandbCallback(verbose=2))
    if r2_callback is not None:
        callbacks.append(r2_callback)

    if args.image_monitor:
        from vtt.callbacks.image_monitor import ImageMonitorCallback

        img_save_dir = os.path.join(save_dir, "image_samples")
        callbacks.append(
            ImageMonitorCallback(
                save_dir=img_save_dir,
                sample_every=10,
                r2_sync_every=50,
                r2_prefix=f"s3://vtt-uav-artifacts/debug/{run_id}/image_samples",
            )
        )

    eval_freq = cfg["eval_freq"]
    eval_episodes = cfg["eval_episodes"]
    total_timesteps = sac_cfg["total_timesteps"]
    best_mean_reward = -np.inf
    best_dir = os.path.join(save_dir, "best")
    os.makedirs(best_dir, exist_ok=True)

    print(f"Starting training: {total_timesteps} timesteps, eval every {eval_freq}")

    try:
        timesteps_done = 0
        while timesteps_done < total_timesteps:
            chunk = min(eval_freq, total_timesteps - timesteps_done)

            model.learn(
                total_timesteps=chunk,
                callback=callbacks,
                log_interval=10,
                reset_num_timesteps=False,
            )
            timesteps_done += chunk

            # Evaluate
            mean_reward, std_reward = evaluate_policy(
                model,
                eval_env,
                n_eval_episodes=eval_episodes,
                deterministic=True,
            )
            print(
                f"Eval at {timesteps_done} steps: "
                f"reward={mean_reward:.2f} +/- {std_reward:.2f}"
            )
            if not args.no_wandb:
                wandb.log(
                    {
                        "eval/mean_reward": mean_reward,
                        "eval/std_reward": std_reward,
                    },
                    step=timesteps_done,
                )

            # Save best model
            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                model.save(os.path.join(best_dir, "best_model"))
                print(f"New best mean reward: {mean_reward:.2f}")

    except KeyboardInterrupt:
        print("\nTraining interrupted! Saving current model...")
    except Exception as e:
        print(f"got unhandled error: {e}")
    finally:
        final_path = os.path.join(save_dir, "final_model")
        model.save(final_path)
        model.save_replay_buffer(
            os.path.join(save_dir, "final_model_replay_buffer.pkl")
        )
        print(f"Model + replay buffer saved to {final_path}")

        if r2_callback is not None:
            print("Uploading final state to R2...")
            r2_callback.upload_final(model)

        if not args.no_wandb:
            wandb.finish()

        env.close()
        eval_env.close()


if __name__ == "__main__":
    main()
