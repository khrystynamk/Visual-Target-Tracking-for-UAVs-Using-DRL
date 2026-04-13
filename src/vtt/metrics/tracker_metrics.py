import numpy as np


class Metrics:
    """
    Metrics reported
    ----------------
    detect_rate_%         : fraction of frames with a valid detection
    loss_events           : number of times the target disappeared
    centroid_rmse_px      : RMS of (cx, cy) image-centre error (detected frames only)
    mean_dist3d_m         : mean Euclidean tracker-target distance
    dist3d_rmse_m         : RMS of tracker-target distance
    mean_episode_frames   : mean length of continuous detection streaks
    mean_recovery_frames  : mean frames from loss to re-detection
    """

    def __init__(self):
        self.total_frames = 0
        self.detect_frames = 0
        self._cx_sq_sum = 0.0
        self._cy_sq_sum = 0.0
        self._dist3d_sum = 0.0
        self._dist3d_sq_sum = 0.0
        self.loss_events = 0
        # episode = continuous detection streak
        self._episode_lengths = []
        self._cur_episode = 0
        # recovery = frames from loss onset to re-detection
        self._recovery_times = []
        self._in_loss = False
        self._loss_duration = 0

    def update(self, detected: bool, cx_err: float, cy_err: float, dist3d: float):
        self.total_frames += 1
        if detected:
            self.detect_frames += 1
            self._cx_sq_sum += cx_err**2
            self._cy_sq_sum += cy_err**2
            self._dist3d_sum += dist3d
            self._dist3d_sq_sum += dist3d**2
            self._cur_episode += 1
            if self._in_loss:
                self._recovery_times.append(self._loss_duration)
                self._loss_duration = 0
                self._in_loss = False
        else:
            if self._cur_episode > 0:
                self._episode_lengths.append(self._cur_episode)
                self._cur_episode = 0
            if not self._in_loss:
                self.loss_events += 1
                self._in_loss = True
            self._loss_duration += 1

    def summary(self):
        n = self.detect_frames
        total = self.total_frames
        all_episodes = self._episode_lengths + (
            [self._cur_episode] if self._cur_episode > 0 else []
        )
        mean_ep = (
            round(float(np.mean(all_episodes)), 2) if all_episodes else float("nan")
        )
        mean_rec = (
            round(float(np.mean(self._recovery_times)), 2)
            if self._recovery_times
            else float("nan")
        )
        return {
            "total_frames": total,
            "detect_rate_%": round(100.0 * n / total, 2) if total else 0.0,
            "loss_events": self.loss_events,
            "centroid_rmse_px": round(
                np.sqrt((self._cx_sq_sum + self._cy_sq_sum) / n), 3
            )
            if n
            else float("nan"),
            "mean_dist3d_m": round(self._dist3d_sum / n, 3) if n else float("nan"),
            "dist3d_rmse_m": round(np.sqrt(self._dist3d_sq_sum / n), 3)
            if n
            else float("nan"),
            "mean_episode_frames": mean_ep,
            "mean_recovery_frames": mean_rec,
        }
