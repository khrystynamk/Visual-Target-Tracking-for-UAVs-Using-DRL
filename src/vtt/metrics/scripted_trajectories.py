"""
Scripted target trajectories for UAV tracking evaluation.
"""

from abc import ABC, abstractmethod

import numpy as np


class BaseTrajectory(ABC):
    """
    Abstract base for all scripted trajectories.
    """

    def __init__(self, origin: np.ndarray | None = None):
        self.origin = (
            np.asarray(origin, dtype=np.float64) if origin is not None else np.zeros(3)
        )

    @abstractmethod
    def position(self, t: float):
        """
        Return world-frame position at time t.
        """

    @abstractmethod
    def velocity(self, t: float):
        """
        Return world-frame velocity at time t.
        """

    @abstractmethod
    def acceleration(self, t: float):
        """
        Return world-frame acceleration at time t.
        """

    def sample(self, duration: float, dt: float = 0.05):
        """
        Sample the trajectory, returning an (N, 3) array of positions.
        """
        times = np.arange(0, duration, dt)
        return np.array([self.position(t) for t in times])


class SinusoidalTrajectory(BaseTrajectory):
    """
    Sinusoidal motion along each axis.
    p_i(t) = a_i * sin(w_i * t + phi_i) + origin_i
    """

    def __init__(
        self,
        amplitudes: tuple[float, float, float] = (2.0, 2.0, 1.0),
        frequencies: tuple[float, float, float] = (0.1, 0.15, 0.08),
        phases: tuple[float, float, float] = (0.0, 0.0, 0.0),
        origin: np.ndarray | None = None,
    ):
        super().__init__(origin)
        self.a = np.asarray(amplitudes, dtype=np.float64)
        self.w = 2 * np.pi * np.asarray(frequencies, dtype=np.float64)
        self.phi = np.asarray(phases, dtype=np.float64)

    @classmethod
    def random(
        cls,
        amp_range: tuple[float, float] = (0.5, 2.5),
        freq_range: tuple[float, float] = (0.05, 0.2),
        origin: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
    ):
        rng = rng or np.random.default_rng()
        amplitudes = tuple(rng.uniform(*amp_range, size=3))
        frequencies = tuple(rng.uniform(*freq_range, size=3))
        phases = tuple(rng.uniform(-np.pi / 2, np.pi / 2, size=3))
        return cls(
            amplitudes=amplitudes, frequencies=frequencies, phases=phases, origin=origin
        )

    def position(self, t: float):
        return (
            self.a * np.sin(self.w * t + self.phi)
            - self.a * np.sin(self.phi)
            + self.origin
        )

    def velocity(self, t: float):
        return self.a * self.w * np.cos(self.w * t + self.phi)

    def acceleration(self, t: float):
        return -self.a * self.w**2 * np.sin(self.w * t + self.phi)


class CircularTrajectory(BaseTrajectory):
    def __init__(
        self,
        radius: float = 3.0,
        angular_speed: float = 0.3,
        vertical_amplitude: float = 0.0,
        vertical_frequency: float = 0.1,
        phase: float = np.pi,
        origin: np.ndarray | None = None,
    ):
        super().__init__(origin)
        self.r = radius
        self.omega = angular_speed
        self.z_amp = vertical_amplitude
        self.z_w = 2 * np.pi * vertical_frequency
        self.phi = phase
        # offset so position(0) == origin
        self._x0 = self.r * np.cos(self.phi)
        self._y0 = self.r * np.sin(self.phi)

    def position(self, t: float):
        x = self.r * np.cos(self.omega * t + self.phi) - self._x0
        y = self.r * np.sin(self.omega * t + self.phi) - self._y0
        z = self.z_amp * np.sin(self.z_w * t)
        return np.array([x, y, z]) + self.origin

    def velocity(self, t: float):
        vx = -self.r * self.omega * np.sin(self.omega * t + self.phi)
        vy = self.r * self.omega * np.cos(self.omega * t + self.phi)
        vz = self.z_amp * self.z_w * np.cos(self.z_w * t)
        return np.array([vx, vy, vz])

    def acceleration(self, t: float):
        ax = -self.r * self.omega**2 * np.cos(self.omega * t + self.phi)
        ay = -self.r * self.omega**2 * np.sin(self.omega * t + self.phi)
        az = -self.z_amp * self.z_w**2 * np.sin(self.z_w * t)
        return np.array([ax, ay, az])


class FigureEightTrajectory(BaseTrajectory):
    """
    Figure-8 pattern in the XY plane.
    x(t) = size * sin(w * t)
    y(t) = size * sin(2 * w * t)
    """

    def __init__(
        self,
        size: float = 3.0,
        speed: float = 0.1,
        origin: np.ndarray | None = None,
    ):
        super().__init__(origin)
        self.size = size
        self.w = 2 * np.pi * speed

    def position(self, t: float):
        x = self.size * np.sin(self.w * t)
        y = self.size * np.sin(2 * self.w * t)
        return np.array([x, y, 0.0]) + self.origin

    def velocity(self, t: float):
        vx = self.size * self.w * np.cos(self.w * t)
        vy = self.size * 2 * self.w * np.cos(2 * self.w * t)
        return np.array([vx, vy, 0.0])

    def acceleration(self, t: float):
        ax = -self.size * self.w**2 * np.sin(self.w * t)
        ay = -self.size * (2 * self.w) ** 2 * np.sin(2 * self.w * t)
        return np.array([ax, ay, 0.0])
