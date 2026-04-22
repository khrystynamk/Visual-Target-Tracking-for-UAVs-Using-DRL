"""R2 checkpoint sync callback for stable-baselines3."""

import json
import subprocess
import time
from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback

UPLOAD_TIMEOUT = 300  # seconds; replay buffers can be large


class R2SyncCallback(BaseCallback):
    """Uploads training checkpoints to Cloudflare R2 periodically.

    Designed for vast.ai spot instances: enables pause/resume by keeping
    the latest model + replay buffer in R2.

    Args:
        run_id: Unique run identifier (used as R2 prefix under runs/).
        save_dir: Local directory where checkpoints are saved.
        bucket: R2 bucket name.
        prefix: Top-level R2 prefix.
        upload_freq: Upload every N timesteps (should be a multiple of
            CheckpointCallback.save_freq).
        verbose: Verbosity level.
    """

    def __init__(
        self,
        run_id: str,
        save_dir: str,
        bucket: str = "vtt-uav-artifacts",
        prefix: str = "runs",
        upload_freq: int = 5000,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.run_id = run_id
        self.save_dir = Path(save_dir)
        self.bucket = bucket
        self.r2_base = f"s3://{bucket}/{prefix}/{run_id}"
        self.upload_freq = upload_freq
        self._uploaded: set[str] = set()
        self._last_upload_step = 0

    def _s3_cp(self, local: str, remote: str) -> bool:
        """Upload a file to R2. Returns True on success."""
        cmd = ["aws", "--profile", "r2", "s3", "cp", local, remote]
        if self.verbose:
            print(f"R2SyncCallback: uploading {local} -> {remote}")
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=UPLOAD_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            print(f"R2SyncCallback: upload timed out after {UPLOAD_TIMEOUT}s: {local}")
            return False
        if result.returncode != 0:
            print(f"R2SyncCallback: upload failed: {result.stderr}")
            return False
        return True

    def _upload_meta(self, step: int) -> None:
        """Write and upload meta.json with current training state."""
        meta = {
            "run_id": self.run_id,
            "last_step": step,
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        meta_path = self.save_dir / "meta.json"
        meta_path.write_text(json.dumps(meta, indent=2))
        self._s3_cp(str(meta_path), f"{self.r2_base}/meta.json")

    def _sync_new_checkpoints(self) -> None:
        """Upload any checkpoint files that haven't been uploaded yet."""
        ckpt_dir = self.save_dir / "checkpoints"
        if not ckpt_dir.exists():
            return
        for f in sorted(ckpt_dir.glob("*.zip")):
            if f.name not in self._uploaded:
                if self._s3_cp(str(f), f"{self.r2_base}/checkpoints/{f.name}"):
                    self._uploaded.add(f.name)

        # Also sync best model if it exists
        best = self.save_dir / "best" / "best_model.zip"
        if best.exists():
            mtime = best.stat().st_mtime
            best_key = f"best_model.zip:{mtime}"
            if best_key not in self._uploaded:
                if self._s3_cp(str(best), f"{self.r2_base}/best/best_model.zip"):
                    self._uploaded.add(best_key)

    def _on_step(self) -> bool:
        elapsed = self.num_timesteps - self._last_upload_step
        if elapsed >= self.upload_freq:
            self._sync_new_checkpoints()
            self._upload_meta(self.num_timesteps)
            self._last_upload_step = self.num_timesteps
        return True

    def upload_final(self, model) -> None:
        """Upload final model + replay buffer to runs/{run_id}/latest/.

        Call this from the finally block in train.py after model.save().
        """
        final_zip = self.save_dir / "final_model.zip"
        replay_pkl = self.save_dir / "final_model_replay_buffer.pkl"
        latest = f"{self.r2_base}/latest"

        if final_zip.exists():
            self._s3_cp(str(final_zip), f"{latest}/model.zip")
        if replay_pkl.exists():
            self._s3_cp(str(replay_pkl), f"{latest}/replay_buffer.pkl")

        # Upload any remaining checkpoints
        self._sync_new_checkpoints()

        step = getattr(model, "num_timesteps", None) or self.num_timesteps
        self._upload_meta(step)

        if self.verbose:
            print(f"R2SyncCallback: final state uploaded to {latest}/")
