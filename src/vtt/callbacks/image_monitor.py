"""SB3 callback that logs depth image stats and sample frames during training.

Serves as sanity telemetry: confirms AirSim is returning real pixels (not
black/empty frames after a silent render failure). Logs to W&B when available
and always saves periodic sample frames to disk.
"""

from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

MAX_DEPTH_VIS = 20.0  # meters; clip depth images here for visualization


class ImageMonitorCallback(BaseCallback):
    """Logs image-array statistics and saves sample depth frames.

    Args:
        save_dir: Directory for local sample images. Created automatically.
        stats_every: Log scalar stats to W&B every N steps.
        sample_every: Save/log a sample frame every N steps.
        verbose: Verbosity level.
    """

    def __init__(
        self,
        save_dir: str = "experiments/image_samples",
        stats_every: int = 500,
        sample_every: int = 2000,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.save_dir = Path(save_dir)
        self.stats_every = stats_every
        self.sample_every = sample_every
        self._last_stats_step = 0
        self._last_sample_step = 0

    def _init_callback(self) -> None:
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def _get_image(self) -> np.ndarray | None:
        obs = self.locals.get("new_obs")
        if obs is None or "image" not in obs:
            return None
        img = obs["image"]
        if img.ndim == 4:  # vectorized: (N, C, H, W) — take first env
            img = img[0]
        return img  # (frame_stack, H, W), raw depth in meters

    def _log_stats(self, img: np.ndarray) -> None:
        try:
            import wandb

            if wandb.run is None:
                return
            wandb.log(
                {
                    "image/mean": float(img.mean()),
                    "image/std": float(img.std()),
                    "image/min": float(img.min()),
                    "image/max": float(img.max()),
                    "image/zero_frac": float((img == 0).mean()),
                },
                step=self.num_timesteps,
            )
        except ImportError:
            pass

    def _save_sample(self, img: np.ndarray) -> None:
        # Take the latest frame from the stack (last channel = most recent)
        frame = img[-1]  # (H, W), raw depth meters

        # Save raw .npy (lossless, small for single-channel float32)
        npy_path = self.save_dir / f"depth_{self.num_timesteps:08d}.npy"
        np.save(npy_path, frame)

        # Also save a normalized PNG for quick visual inspection
        vis = np.clip(frame, 0, MAX_DEPTH_VIS) / MAX_DEPTH_VIS  # [0, 1]
        vis = (vis * 255).astype(np.uint8)
        try:
            from PIL import Image

            png_path = self.save_dir / f"depth_{self.num_timesteps:08d}.png"
            Image.fromarray(vis).save(png_path)
        except ImportError:
            pass  # PIL not installed — .npy is enough

        # Log to W&B
        try:
            import wandb

            if wandb.run is None:
                return
            wandb.log(
                {
                    "image/sample": wandb.Image(
                        vis,
                        caption=f"depth t={self.num_timesteps}",
                    ),
                },
                step=self.num_timesteps,
            )
        except ImportError:
            pass

    def _on_step(self) -> bool:
        img = self._get_image()
        if img is None:
            return True

        if self.num_timesteps - self._last_stats_step >= self.stats_every:
            self._log_stats(img)
            self._last_stats_step = self.num_timesteps

        if self.num_timesteps - self._last_sample_step >= self.sample_every:
            self._save_sample(img)
            self._last_sample_step = self.num_timesteps

        return True
