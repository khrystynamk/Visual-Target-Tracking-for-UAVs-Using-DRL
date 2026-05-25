import numpy as np

from vtt.metrics.scripted_trajectories import (
    CircularTrajectory,
    FigureEightTrajectory,
    SinusoidalTrajectory,
)


TRAJECTORY_TRAIN_FAMILIES = ("sinusoidal", "circular", "figure_eight")


def sample_train_trajectory(rng: np.random.Generator, origin: np.ndarray):
    """
    Sample a training trajectory uniformly across families with randomized params.
    """
    family = rng.choice(TRAJECTORY_TRAIN_FAMILIES)
    if family == "sinusoidal":
        return SinusoidalTrajectory.random(origin=origin, rng=rng)
    if family == "circular":
        return CircularTrajectory.random(origin=origin, rng=rng)
    return FigureEightTrajectory.random(origin=origin, rng=rng)


TRAJECTORY_EVAL_PRESETS = {
    "sinusoidal": lambda origin: SinusoidalTrajectory(
        amplitudes=(3.0, 2.0, 1.0),
        frequencies=(0.08, 0.1, 0.05),
        origin=origin,
    ),
    "helix": lambda origin: CircularTrajectory(
        radius=4.0,
        angular_speed=0.25,
        vertical_amplitude=1.5,
        vertical_frequency=0.04,
        origin=origin,
    ),
    "circular": lambda origin: CircularTrajectory(
        radius=4.0,
        angular_speed=0.25,
        origin=origin,
    ),
    "figure_eight": lambda origin: FigureEightTrajectory(
        size=4.0,
        speed=0.08,
        origin=origin,
    ),
}
