"""
Gymnasium environment for DRL-based visual target tracking in AirSim.

Actor:  depth images + tracker state (velocity + orientation) → CNN + MLP → action
Critic: relative state + tracker state (privileged) → MLP → Q-value

Observation space:
  "image":        (C, 224, 224) — stacked depth/RGB frames
  "actor_state":  (6,) — tracker [vx, vy, vz, roll, pitch, yaw] in body frame
  "critic_state": (12,) — relative [pos(3), vel(3)] + tracker [pos(3), vel(3)]
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
        use_depth: bool = True,
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
        self.use_depth = use_depth
        self.render_mode = render_mode

        channels_per_frame = 1 if use_depth else 3
        total_channels = frame_stack * channels_per_frame

        self.observation_space = gym.spaces.Dict(
            {
                "image": gym.spaces.Box(
                    0.0,
                    1.0,
                    shape=(total_channels, image_size, image_size),
                    dtype=np.float32,
                ),
                # Tracker's [vx, vy, vz, roll, pitch, yaw]
                "actor_state": gym.spaces.Box(
                    -np.inf,
                    np.inf,
                    shape=(6,),
                    dtype=np.float32,
                ),
                # Privileged states: target [pos(3), vel(3)] + tracker [pos(3), vel(3)]
                "critic_state": gym.spaces.Box(
                    -np.inf,
                    np.inf,
                    shape=(12,),
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

        self.client.reset()

        self.client.enableApiControl(True, TRACKER_VEHICLE)
        self.client.armDisarm(True, TRACKER_VEHICLE)
        self.client.takeoffAsync(vehicle_name=TRACKER_VEHICLE).join()

        self.client.enableApiControl(True, TARGET_VEHICLE)
        self.client.armDisarm(True, TARGET_VEHICLE)
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
        rel_vel = obs["critic_state"][3:6]
        reward, done = self._compute_reward(rel_pos, rel_vel, action)

        truncated = self._step_count >= self.max_episode_steps

        info = {
            "distance": float(np.linalg.norm(rel_pos)),
            "t_elapsed": self._follower.elapsed_time,
        }

        return obs, reward, done, truncated, info

    def _capture_image(self) -> np.ndarray:
        if self.use_depth:
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
            depth = airsim.list_to_2d_float_array(
                responses[0].image_data_float,
                responses[0].width,
                responses[0].height,
            )
            depth = np.clip(depth, 0.0, self.max_distance) / self.max_distance
            depth = cv2.resize(depth, (self.image_size, self.image_size))
            return depth[np.newaxis].astype(np.float32)
        else:
            responses = self.client.simGetImages(
                [
                    airsim.ImageRequest(
                        TRACKER_CAMERA,
                        airsim.ImageType.Scene,
                        False,
                        False,
                    ),
                ],
                vehicle_name=TRACKER_VEHICLE,
            )
            img1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
            bgr = img1d.reshape(responses[0].height, responses[0].width, 3)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (self.image_size, self.image_size))
            return (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)

    def _get_obs(self) -> dict:
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
        roll, pitch, yaw = airsim.to_eularian_angles(
            tracker.kinematics_estimated.orientation
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

        cos_y, sin_y = np.cos(yaw), np.sin(yaw)
        R = np.array(
            [
                [cos_y, sin_y, 0],
                [-sin_y, cos_y, 0],
                [0, 0, 1],
            ]
        )

        rel_pos = R @ (tgt_pos - t_pos)
        rel_vel = R @ (tgt_vel - t_vel)
        body_vel = R @ t_vel

        actor_state = np.array(
            [
                body_vel[0],
                body_vel[1],
                body_vel[2],
                roll,
                pitch,
                yaw,
            ],
            dtype=np.float32,
        )

        critic_state = np.concatenate(
            [
                rel_pos,
                rel_vel,
                t_pos,
                t_vel,
            ]
        ).astype(np.float32)

        stacked = np.concatenate(self._frame_buffer, axis=0)

        return {
            "image": stacked,
            "actor_state": actor_state,
            "critic_state": critic_state,
        }

    def _compute_reward(self, relative_pos, relative_vel, action):
        x, y, z = relative_pos
        dist = np.linalg.norm(relative_pos)

        if abs(x) < 0.01:
            x = 0.01

        fov_half = np.pi / 4
        y_err = abs(np.arctan2(y, x)) / fov_half
        z_err = abs(np.arctan2(z, x)) / fov_half
        d_err = abs(dist - self.desired_distance) / self.desired_distance

        r_track = max(0, 1 - y_err) * max(0, 1 - z_err) * max(0, 1 - d_err)

        vel_mag = np.linalg.norm(relative_vel)
        act_mag = np.linalg.norm(action)
        r_vel = -0.3 * vel_mag / (1 + vel_mag)
        r_act = -0.1 * act_mag / (1 + act_mag)

        reward = r_track + r_vel + r_act

        done = bool(dist > self.max_distance or dist < self.min_distance)
        if done:
            reward = -10.0

        return float(reward), done

    def close(self):
        if self._follower is not None:
            self._follower.stop()
        self.client.enableApiControl(False, TRACKER_VEHICLE)
        self.client.enableApiControl(False, TARGET_VEHICLE)
