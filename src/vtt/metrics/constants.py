from vtt.metrics.scripted_trajectories import (
    CircularTrajectory,
    FigureEightTrajectory,
    SinusoidalTrajectory,
)


TRAJECTORY_PRESETS = {
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
