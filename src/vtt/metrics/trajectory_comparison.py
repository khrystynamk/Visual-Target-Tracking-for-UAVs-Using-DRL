"""
Record tracker trajectory and compare against target ground truth.
"""

from dataclasses import dataclass, field

import numpy as np
import matplotlib.pyplot as plt


@dataclass
class TrajectoryRecord:
    """
    Stores timestamped 3D positions for a vehicle.
    """

    name: str
    times: list[float] = field(default_factory=list)
    positions: list[np.ndarray] = field(default_factory=list)

    def append(self, t: float, position: np.ndarray):
        self.times.append(t)
        self.positions.append(np.asarray(position, dtype=np.float64))

    def as_arrays(self):
        """
        Return (times (N,), positions (N, 3)).
        """
        return np.array(self.times), np.array(self.positions)


def compute_tracking_errors(
    target_times: np.ndarray,
    target_positions: np.ndarray,
    tracker_times: np.ndarray,
    tracker_positions: np.ndarray,
):
    """
    Compute tracking error metrics between target and tracker trajectories.

    Returns a dict with:
        - position_errors: (N, 3) per-axis errors at each tracker timestamp
        - distance_errors: (N,) Euclidean distance at each timestamp
        - rmse: scalar root-mean-square distance error
        - mean_error: scalar mean distance error
        - max_error: scalar maximum distance error
        - per_axis_rmse: (3,) RMSE for each axis independently
    """
    # Interpolate target positions at tracker timestamps
    target_interp = np.column_stack(
        [
            np.interp(tracker_times, target_times, target_positions[:, i])
            for i in range(3)
        ]
    )

    position_errors = tracker_positions - target_interp
    distance_errors = np.linalg.norm(position_errors, axis=1)
    per_axis_rmse = np.sqrt(np.mean(position_errors**2, axis=0))

    return {
        "position_errors": position_errors,
        "distance_errors": distance_errors,
        "rmse": float(np.sqrt(np.mean(distance_errors**2))),
        "mean_error": float(np.mean(distance_errors)),
        "max_error": float(np.max(distance_errors)),
        "per_axis_rmse": per_axis_rmse,
    }


def plot_trajectory_comparison(
    target_times: np.ndarray,
    target_positions: np.ndarray,
    tracker_times: np.ndarray,
    tracker_positions: np.ndarray,
    title: str = "Tracker vs Target Trajectory",
    show: bool = True,
    save_path: str | None = None,
):
    """
    3D plot comparing target ground truth and tracker trajectory.
    """
    errors = compute_tracking_errors(
        target_times, target_positions, tracker_times, tracker_positions
    )

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(title, fontsize=14)

    ax3d = fig.add_subplot(2, 3, 1, projection="3d")
    ax3d.plot(
        target_positions[:, 0],
        target_positions[:, 1],
        target_positions[:, 2],
        linewidth=1.5,
        color="blue",
        label="Target (GT)",
        alpha=0.8,
    )
    ax3d.plot(
        tracker_positions[:, 0],
        tracker_positions[:, 1],
        tracker_positions[:, 2],
        linewidth=1.5,
        color="orange",
        label="Tracker",
        alpha=0.8,
    )
    ax3d.scatter(*target_positions[0], marker="o", s=80, c="green", zorder=5)
    ax3d.set_xlabel("X (m)")
    ax3d.set_ylabel("Y (m)")
    ax3d.set_zlabel("Z (m)")
    ax3d.set_title("3D Paths")
    ax3d.legend()

    # --- Top-down (XY) view ---
    ax_xy = fig.add_subplot(2, 3, 2)
    ax_xy.plot(
        target_positions[:, 0],
        target_positions[:, 1],
        linewidth=1.5,
        color="blue",
        label="Target",
    )
    ax_xy.plot(
        tracker_positions[:, 0],
        tracker_positions[:, 1],
        linewidth=1.5,
        color="orange",
        label="Tracker",
    )
    ax_xy.set_xlabel("X (m)")
    ax_xy.set_ylabel("Y (m)")
    ax_xy.set_title("Top-Down View (XY)")
    ax_xy.legend()
    ax_xy.set_aspect("equal")
    ax_xy.grid(True, alpha=0.3)

    # --- Distance error over time ---
    ax_dist = fig.add_subplot(2, 3, 3)
    ax_dist.plot(tracker_times, errors["distance_errors"], color="red", linewidth=1.0)
    ax_dist.axhline(
        errors["rmse"],
        color="black",
        linestyle="--",
        linewidth=0.8,
        label=f"RMSE = {errors['rmse']:.2f} m",
    )
    ax_dist.axhline(
        errors["mean_error"],
        color="gray",
        linestyle=":",
        linewidth=0.8,
        label=f"Mean = {errors['mean_error']:.2f} m",
    )
    ax_dist.set_xlabel("Time (s)")
    ax_dist.set_ylabel("Distance Error (m)")
    ax_dist.set_title("Tracking Error")
    ax_dist.legend()
    ax_dist.grid(True, alpha=0.3)

    # --- Per-axis position comparison ---
    axis_names = ["X", "Y", "Z"]
    for i, name in enumerate(axis_names):
        ax = fig.add_subplot(2, 3, 4 + i)
        ax.plot(
            target_times,
            target_positions[:, i],
            linewidth=1.5,
            color="blue",
            label="Target",
            alpha=0.8,
        )
        ax.plot(
            tracker_times,
            tracker_positions[:, i],
            linewidth=1.5,
            color="orange",
            label="Tracker",
            alpha=0.8,
        )
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(f"{name} (m)")
        rmse_i = errors["per_axis_rmse"][i]
        ax.set_title(f"{name}-axis (RMSE = {rmse_i:.2f} m)")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    if show:
        plt.show()

    return fig, errors
