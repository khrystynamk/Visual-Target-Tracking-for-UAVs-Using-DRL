"""
Gymnasium environment for DRL-based visual target tracking in AirSim.

Actor:  depth images only → CNN → MLP → action
Critic: relative target state (privileged) → MLP → Q-value
"""

import gymnasium as gym
import numpy as np
import cv2
import airsim

from vtt.constants import (
    TRACKER_VEHICLE,
    TARGET_VEHICLE,
    TRACKER_CAMERA,
    TS,
    IMAGE_SIZE,
    FRAME_STACK,
    MAX_EPISODE_STEPS,
    MAX_VEL,
    MAX_YAW_RATE,
    DESIRED_DISTANCE,
    MAX_DISTANCE,
    MIN_DISTANCE,
)
from vtt.metrics.scripted_trajectories import SinusoidalTrajectory
from vtt.target.trajectory_follower import TrajectoryFollower


def _quat_to_rotation_matrix(q) -> np.ndarray:
    """
    Convert AirSim quaternion to a rotation matrix.
    """
    q0, q1, q2, q3 = q.w_val, q.x_val, q.y_val, q.z_val
    return np.array(
        [
            [
                2 * (q0 * q0 + q1 * q1) - 1,
                2 * (q1 * q2 - q0 * q3),
                2 * (q1 * q3 + q0 * q2),
            ],
            [
                2 * (q1 * q2 + q0 * q3),
                2 * (q0 * q0 + q2 * q2) - 1,
                2 * (q2 * q3 - q0 * q1),
            ],
            [
                2 * (q1 * q3 - q0 * q2),
                2 * (q2 * q3 + q0 * q1),
                2 * (q0 * q0 + q3 * q3) - 1,
            ],
        ]
    )


class TrackingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        image_size: int = IMAGE_SIZE,
        frame_stack: int = FRAME_STACK,
        max_episode_steps: int = MAX_EPISODE_STEPS,
        max_vel: float = MAX_VEL,
        max_yaw_rate: float = MAX_YAW_RATE,
        desired_distance: float = DESIRED_DISTANCE,
        max_distance: float = MAX_DISTANCE,
        min_distance: float = MIN_DISTANCE,
        render_mode: str | None = None,
    ):
        super().__init__()

        self.image_size = image_size
        self.frame_stack = frame_stack
        self.max_episode_steps = max_episode_steps
        self.max_vel = max_vel
        self.max_yaw_rate = max_yaw_rate
        self.desired_distance = desired_distance
        self.max_distance = max_distance
        self.min_distance = min_distance
        self.render_mode = render_mode

        # (frame_stack, H, W) — stacked depth frames.
        self.observation_space = gym.spaces.Dict(
            {
                # Depth frames
                "image": gym.spaces.Box(
                    0.0,
                    1.0,
                    shape=(frame_stack, image_size, image_size),
                    dtype=np.float32,
                ),
                # Relative target state [pos(3), vel(3), acc(3)]
                "critic_state": gym.spaces.Box(
                    -np.inf,
                    np.inf,
                    shape=(9,),
                    dtype=np.float32,
                ),
            }
        )

        self.action_space = gym.spaces.Box(
            -1.0,
            1.0,
            shape=(4,),
            dtype=np.float32,
        )

        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()

        self._trajectory = None
        self._follower = None
        self._step_count = 0
        self._frame_buffer = None
        self._rng = np.random.default_rng()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        if self._follower is not None:
            self._follower.stop()
            self._follower = None

        self.client.reset()

        for vehicle in [TRACKER_VEHICLE, TARGET_VEHICLE]:
            self.client.enableApiControl(True, vehicle)
            self.client.armDisarm(True, vehicle)

        self.client.takeoffAsync(vehicle_name=TRACKER_VEHICLE).join()
        self.client.takeoffAsync(vehicle_name=TARGET_VEHICLE).join()

        target_pose = self.client.simGetVehiclePose(TARGET_VEHICLE)
        tp = target_pose.position
        origin = np.array([tp.x_val, tp.y_val, -tp.z_val])

        self._trajectory = SinusoidalTrajectory.random(
            amp_range=(1.0, 3.0),
            freq_range=(0.05, 0.15),
            origin=origin,
            rng=self._rng,
        )
        self._follower = TrajectoryFollower(self._trajectory, TARGET_VEHICLE, dt=TS)
        self._follower.start()

        self._step_count = 0

        frame = self._capture_image()
        self._frame_buffer = [frame] * self.frame_stack

        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        self._step_count += 1

        vx_body = float(action[0]) * self.max_vel
        vy_body = float(action[1]) * self.max_vel
        vz_ned = float(action[2]) * self.max_vel
        yaw_rate = float(action[3]) * self.max_yaw_rate

        tracker_state = self.client.getMultirotorState(vehicle_name=TRACKER_VEHICLE)
        _, _, yaw = airsim.to_eularian_angles(
            tracker_state.kinematics_estimated.orientation
        )
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)
        vx_world = vx_body * cos_y - vy_body * sin_y
        vy_world = vx_body * sin_y + vy_body * cos_y

        self.client.moveByVelocityAsync(
            vx_world,
            vy_world,
            vz_ned,
            TS,
            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
            yaw_mode=airsim.YawMode(True, yaw_rate),
            vehicle_name=TRACKER_VEHICLE,
        )

        frame = self._capture_image()
        self._frame_buffer.append(frame)
        self._frame_buffer.pop(0)

        obs = self._get_obs()

        rel_pos = obs["critic_state"][:3]
        reward, done = self._compute_reward(rel_pos)

        truncated = self._step_count >= self.max_episode_steps

        info = {
            "distance": float(np.linalg.norm(rel_pos)),
            "t_elapsed": self._follower.elapsed_time,
        }

        return obs, reward, done, truncated, info

    def _capture_image(self):
        responses = self.client.simGetImages(
            [
                airsim.ImageRequest(
                    TRACKER_CAMERA,
                    airsim.ImageType.DepthPerspective,
                    pixels_as_float=True,
                    compress=False,
                ),
            ],
            vehicle_name=TRACKER_VEHICLE,
        )

        if responses[0].width == 0 or responses[0].height == 0:
            return np.zeros((1, self.image_size, self.image_size), dtype=np.float32)

        depth = airsim.list_to_2d_float_array(
            responses[0].image_data_float,
            responses[0].width,
            responses[0].height,
        )
        depth = np.clip(depth, 0.0, self.max_distance) / self.max_distance
        depth = cv2.resize(depth, (self.image_size, self.image_size))
        return depth[np.newaxis].astype(np.float32)  # (1, H, W)

    def _get_obs(self):
        tracker = self.client.getMultirotorState(vehicle_name=TRACKER_VEHICLE)
        target_pose = self.client.simGetVehiclePose(TARGET_VEHICLE)
        target_state = self.client.getMultirotorState(vehicle_name=TARGET_VEHICLE)

        t_pos = np.array(
            [
                tracker.kinematics_estimated.position.x_val,
                tracker.kinematics_estimated.position.y_val,
                tracker.kinematics_estimated.position.z_val,
            ]
        )
        t_vel = np.array(
            [
                tracker.kinematics_estimated.linear_velocity.x_val,
                tracker.kinematics_estimated.linear_velocity.y_val,
                tracker.kinematics_estimated.linear_velocity.z_val,
            ]
        )
        tgt_pos = np.array(
            [
                target_pose.position.x_val,
                target_pose.position.y_val,
                target_pose.position.z_val,
            ]
        )
        tgt_vel = np.array(
            [
                target_state.kinematics_estimated.linear_velocity.x_val,
                target_state.kinematics_estimated.linear_velocity.y_val,
                target_state.kinematics_estimated.linear_velocity.z_val,
            ]
        )
        t_acc = np.array(
            [
                tracker.kinematics_estimated.linear_acceleration.x_val,
                tracker.kinematics_estimated.linear_acceleration.y_val,
                tracker.kinematics_estimated.linear_acceleration.z_val,
            ]
        )
        tgt_acc = np.array(
            [
                target_state.kinematics_estimated.linear_acceleration.x_val,
                target_state.kinematics_estimated.linear_acceleration.y_val,
                target_state.kinematics_estimated.linear_acceleration.z_val,
            ]
        )

        q = tracker.kinematics_estimated.orientation
        R = _quat_to_rotation_matrix(q).T

        rel_pos = R @ (tgt_pos - t_pos)
        rel_vel = R @ (tgt_vel - t_vel)
        rel_acc = R @ (tgt_acc - t_acc)

        # 9D relative target state
        critic_state = np.concatenate([rel_pos, rel_vel, rel_acc]).astype(np.float32)

        stacked = np.concatenate(self._frame_buffer, axis=0)

        return {
            "image": stacked,
            "critic_state": critic_state,
        }

    def _compute_reward(self, relative_pos):
        x, y, z = relative_pos
        dist = np.linalg.norm(relative_pos)

        if abs(x) < 0.01:
            x = 0.01

        fov_half = np.pi / 4
        # Angular errors
        y_err = abs(np.arctan(y / x) / fov_half)
        z_err = abs(np.arctan(z / x) / fov_half)
        x_err = abs(x - self.desired_distance)

        # Per-axis rewards
        y_rew = max(0, 1 - y_err)
        z_rew = max(0, 1 - z_err)
        x_rew = max(0, 1 - x_err)

        r_track = (x_rew * y_rew * z_rew) ** (1 / 3)
        reward = r_track * (300 / self.max_episode_steps)

        done = bool(dist > self.max_distance or dist < self.min_distance)
        if done:
            reward = -10.0 / (300 / self.max_episode_steps)

        return float(reward), done

    def close(self):
        if self._follower is not None:
            self._follower.stop()
        self.client.enableApiControl(False, TRACKER_VEHICLE)
        self.client.enableApiControl(False, TARGET_VEHICLE)
