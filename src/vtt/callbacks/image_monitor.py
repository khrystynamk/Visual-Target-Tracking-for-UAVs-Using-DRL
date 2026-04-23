"""
SB3 callback that logs depth sample frames during training.

Serves as sanity-check: confirms AirSim is returning real pixels.
"""

import subprocess
from pathlib import Path

import numpy as np
from PIL import Image
from stable_baselines3.common.callbacks import BaseCallback


class ImageMonitorCallback(BaseCallback):
    """
    Logs image-array statistics and saves sample depth frames.

    Args:
        save_dir: Directory for local sample images. Created automatically.
        stats_every: Log scalar stats to W&B every N steps.
        sample_every: Save/log a sample frame every N steps.
        r2_sync_every: Sync save_dir to R2 every N steps (0 = disabled).
        r2_prefix: R2 path prefix for sync.
        verbose: Verbosity level.
    """

    def __init__(
        self,
        save_dir: str = "experiments/image_samples",
        stats_every: int = 500,
        sample_every: int = 2000,
        r2_sync_every: int = 0,
        r2_prefix: str = "s3://vtt-uav-artifacts/debug/image_samples",
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.save_dir = Path(save_dir)
        self.stats_every = stats_every
        self.sample_every = sample_every
        self.r2_sync_every = r2_sync_every
        self.r2_prefix = r2_prefix
        self._last_stats_step = 0
        self._last_sample_step = 0
        self._last_sync_step = 0

    def _init_callback(self) -> None:
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def _get_image(self) -> np.ndarray | None:
        obs = self.locals.get("new_obs")
        if obs is None or "image" not in obs:
            return None
        img = obs["image"]
        if img.ndim == 4:
            img = img[0]
        return img

    def _save_sample(self, img: np.ndarray) -> None:
        frame = img[-1]
        npy_path = self.save_dir / f"depth_{self.num_timesteps:08d}.npy"
        np.save(npy_path, frame)

        d = np.log1p(frame) / np.log1p(100)
        vis = (d * 255).clip(0, 255).astype(np.uint8)
        png_path = self.save_dir / f"depth_{self.num_timesteps:08d}.png"
        Image.fromarray(vis).save(png_path)

    def _r2_sync(self) -> None:
        """
        Batch-sync the local image_samples dir to R2.
        """
        cmd = [
            "aws",
            "--profile",
            "r2",
            "s3",
            "sync",
            str(self.save_dir),
            self.r2_prefix,
            "--exclude",
            "*.npy",
        ]
        print(f"ImageMonitor: syncing PNGs to {self.r2_prefix}")
        subprocess.run(cmd, capture_output=True, timeout=60)

    def _on_step(self) -> bool:
        img = self._get_image()
        if img is None:
            return True

        if self.num_timesteps - self._last_sample_step >= self.sample_every:
            self._save_sample(img)
            self._last_sample_step = self.num_timesteps

        if (
            self.r2_sync_every > 0
            and self.num_timesteps - self._last_sync_step >= self.r2_sync_every
        ):
            self._r2_sync()
            self._last_sync_step = self.num_timesteps

        return True
