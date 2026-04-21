from vtt.metrics.scripted_trajectories import (
    CircularTrajectory,
    FigureEightTrajectory,
    SinusoidalTrajectory,
)


TRAJECTORY_TRAIN_PRESETS = [
    lambda origin: SinusoidalTrajectory(
        amplitudes=(1.5, 1.0, 0.5),
        frequencies=(0.05, 0.06, 0.04),
        origin=origin,
    ),
    lambda origin: SinusoidalTrajectory(
        amplitudes=(2.0, 1.5, 0.8),
        frequencies=(0.06, 0.07, 0.05),
        origin=origin,
    ),
    lambda origin: SinusoidalTrajectory(
        amplitudes=(3.0, 2.5, 0.5),
        frequencies=(0.08, 0.09, 0.04),
        origin=origin,
    ),
    lambda origin: SinusoidalTrajectory(
        amplitudes=(1.5, 1.0, 2.0),
        frequencies=(0.06, 0.05, 0.1),
        origin=origin,
    ),
    lambda origin: SinusoidalTrajectory(
        amplitudes=(3.0, 1.0, 0.8),
        frequencies=(0.1, 0.06, 0.05),
        origin=origin,
    ),
    lambda origin: SinusoidalTrajectory(
        amplitudes=(1.5, 1.5, 0.8),
        frequencies=(0.12, 0.14, 0.08),
        origin=origin,
    ),
    lambda origin: SinusoidalTrajectory(
        amplitudes=(2.5, 2.0, 1.0),
        frequencies=(0.1, 0.12, 0.08),
        origin=origin,
    ),
    lambda origin: SinusoidalTrajectory(
        amplitudes=(3.0, 2.5, 1.5),
        frequencies=(0.12, 0.15, 0.1),
        origin=origin,
    ),
]


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
