import time
import threading

import airsim
import numpy as np

from vtt.metrics.scripted_trajectories import BaseTrajectory


class TrajectoryFollower:
    """
    Drives the target drone along a scripted trajectory in AirSim.
    """

    def __init__(
        self,
        trajectory: BaseTrajectory,
        vehicle_name: str = "Target",
        dt: float = 0.05,
        api_port: int = 41451,
    ):
        self.trajectory = trajectory
        self.vehicle_name = vehicle_name
        self.dt = dt
        self.api_port = api_port

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._t_elapsed = 0.0

        self._recorded_times: list[float] = []
        self._recorded_positions: list[np.ndarray] = []

    def start(self):
        self._stop_event.clear()
        self._t_elapsed = 0.0
        self._recorded_times.clear()
        self._recorded_positions.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    @property
    def elapsed_time(self):
        return self._t_elapsed

    def get_recorded_positions(self):
        if not self._recorded_positions:
            return np.empty((0, 3))
        return np.array(self._recorded_positions)

    def get_recorded_times(self):
        return np.array(self._recorded_times)

    def _run(self):
        # Each thread needs its own AirSim client — msgpack-rpc/tornado
        # IOLoop cannot be shared across threads.
        client = airsim.MultirotorClient(port=self.api_port)
        client.confirmConnection()

        while not self._stop_event.is_set():
            t0 = time.time()
            vel_zup = self.trajectory.velocity(self._t_elapsed)
            client.moveByVelocityAsync(
                float(vel_zup[0]),
                float(vel_zup[1]),
                float(-vel_zup[2]),
                self.dt,
                vehicle_name=self.vehicle_name,
            )

            pos_zup = self.trajectory.position(self._t_elapsed)
            self._recorded_times.append(self._t_elapsed)
            self._recorded_positions.append(pos_zup.copy())
            self._t_elapsed += self.dt

            elapsed = time.time() - t0
            sleep_time = self.dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
