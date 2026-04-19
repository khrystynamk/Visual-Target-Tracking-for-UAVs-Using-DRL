import threading

import airsim
from pynput import keyboard
import numpy as np
from scipy.spatial.transform import Rotation as ScipyRotation
import time

from vtt.constants import (
    TARGET_VEHICLE,
    ACCELERATION,
    MAX_VEL,
    MAX_YAW_RATE,
    TICK,
    FRICTION,
)


class DroneController:
    """
    Simple drone controller for manual drone navigation using a keyboard.
    """

    def __init__(self, vehicle_name: str = TARGET_VEHICLE):
        self.desired_velocity = np.zeros(3, dtype=np.float32)

        self._key_command_mapping = {
            keyboard.Key.up: "forward",
            keyboard.Key.down: "backward",
            keyboard.Key.left: "turn left",
            keyboard.Key.right: "turn right",
            keyboard.KeyCode.from_char("w"): "forward",
            keyboard.KeyCode.from_char("s"): "backward",
            keyboard.KeyCode.from_char("a"): "left",
            keyboard.KeyCode.from_char("d"): "right",
            keyboard.Key.space: "up",
            keyboard.Key.shift: "down",
        }

        self._active_commands = {
            command: False for command in self._key_command_mapping.values()
        }

        self._vehicle = vehicle_name

        self._client = airsim.MultirotorClient()
        self._client.confirmConnection()
        self._client.enableApiControl(True, self._vehicle)
        self._client.armDisarm(True, self._vehicle)
        self._client.takeoffAsync(vehicle_name=self._vehicle).join()

    def start(self):
        """Start keyboard control in a background thread."""
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop keyboard control."""
        self._stop.set()
        if hasattr(self, "_thread"):
            self._thread.join(timeout=2.0)

    def fly_by_keyboard(self):
        """Run keyboard control on the main thread (blocking)."""
        self._stop = threading.Event()
        self._run()

    def _run(self):
        with keyboard.Listener(
            on_press=self._on_press, on_release=self._on_release
        ) as listener:
            while not self._stop.is_set() and listener.running:
                self._handle_commands()
                time.sleep(TICK / 2.0)
            listener.stop()

    def move(self, velocity, yaw_rate):
        self._client.moveByVelocityAsync(
            velocity[0].item(),
            velocity[1].item(),
            velocity[2].item(),
            TICK,
            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
            yaw_mode=airsim.YawMode(True, yaw_rate),
            vehicle_name=self._vehicle,
        )

    def _on_press(self, key):
        if key in self._key_command_mapping.keys():
            self._active_commands[self._key_command_mapping[key]] = True
        elif key == keyboard.Key.esc:
            return False

    def _on_release(self, key):
        if key in self._key_command_mapping.keys():
            self._active_commands[self._key_command_mapping[key]] = False

    def _handle_commands(self):
        drone_orientation = ScipyRotation.from_quat(
            self._client.simGetVehiclePose(self._vehicle).orientation.to_numpy_array()
        )
        yaw = drone_orientation.as_euler("zyx")[0]
        forward_direction = np.array([np.cos(yaw), np.sin(yaw), 0])
        left_direction = np.array(
            [np.cos(yaw - np.deg2rad(90)), np.sin(yaw - np.deg2rad(90)), 0]
        )

        if self._active_commands["forward"] or self._active_commands["backward"]:
            forward_increment = forward_direction * TICK * ACCELERATION
            if self._active_commands["forward"]:
                self.desired_velocity += forward_increment
            else:
                self.desired_velocity -= forward_increment
        else:
            forward_component = (
                np.dot(self.desired_velocity, forward_direction) * forward_direction
            )
            self.desired_velocity -= FRICTION * forward_component

        if self._active_commands["up"] or self._active_commands["down"]:
            vertical_component = drone_orientation.apply(np.array([0.0, 0.0, -1.0]))
            vertical_component *= TICK * ACCELERATION
            if self._active_commands["up"]:
                self.desired_velocity += vertical_component
            else:
                self.desired_velocity -= vertical_component
        else:
            self.desired_velocity[2] *= FRICTION

        if self._active_commands["left"] or self._active_commands["right"]:
            side_increment = left_direction * TICK * ACCELERATION
            if self._active_commands["left"]:
                self.desired_velocity += side_increment
            else:
                self.desired_velocity -= side_increment
        else:
            left_component = (
                np.dot(self.desired_velocity, left_direction) * left_direction
            )
            self.desired_velocity -= FRICTION * left_component

        speed = np.linalg.norm(self.desired_velocity)
        if speed > MAX_VEL:
            self.desired_velocity = self.desired_velocity / speed * MAX_VEL

        yaw_rate = 0.0
        if self._active_commands["turn left"]:
            yaw_rate = -MAX_YAW_RATE
        elif self._active_commands["turn right"]:
            yaw_rate = MAX_YAW_RATE

        self.move(self.desired_velocity, yaw_rate)


if __name__ == "__main__":
    controller = DroneController()
    controller.fly_by_keyboard()
