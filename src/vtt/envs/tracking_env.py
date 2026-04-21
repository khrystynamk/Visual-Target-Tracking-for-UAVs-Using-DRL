"""
Gymnasium environment for DRL-based visual target tracking in AirSim.

Actor:  depth images only → CNN → MLP → action
Critic: relative target state (privileged) → MLP → Q-value
"""

import gymnasium as gym
import numpy as np
import airsim

from vtt.constants import (
    TRACKER_VEHICLE,
    TARGET_VEHICLE,
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
from vtt.target.trajectory_follower import TrajectoryFollower
from vtt.utils.camera_helpers import (
    capture_depth,
    setup_detector,
    get_raw_camera_resolution,
    get_relative_bbox,
    render_depth_with_bbox,
)
from vtt.metrics.constants import TRAJECTORY_TRAIN_PRESETS


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
        show_cv: bool = False,
        render_mode: str | None = None,
    ):
        super().__init__()
        self.show_cv = show_cv

        self.image_size = image_size
        self.frame_stack = frame_stack
        self.max_episode_steps = max_episode_steps
        self.max_vel = max_vel
        self.max_yaw_rate = max_yaw_rate
        self.desired_distance = desired_distance
        self.max_distance = max_distance
        self.min_distance = min_distance
        self.render_mode = render_mode

        self.observation_space = gym.spaces.Dict(
            {
                # Stacked colormapped depth frames (frame_stack * 3, H, W)
                "image": gym.spaces.Box(
                    0.0,
                    1.0,
                    shape=(frame_stack * 3, image_size, image_size),
                    dtype=np.float32,
                ),
                # Relative bbox from oracle detector [cx, cy, w, h] normalized to [0,1]
                "bbox": gym.spaces.Box(
                    0.0,
                    1.0,
                    shape=(4,),
                    dtype=np.float32,
                ),
                # Relative target state [pos(3), vel(3), acc(3)] — critic only
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
        self._raw_w, self._raw_h = get_raw_camera_resolution(self.client)

        self._trajectory = None
        self._follower = None
        self._step_count = 0
        self._episode_count = 0
        self._frames_without_detection = 0
        self._max_lost_steps = 30
        self._frame_buffer = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self._follower is not None:
            self._follower.stop()
            self._follower = None

        self.client.reset()

        for vehicle in [TRACKER_VEHICLE, TARGET_VEHICLE]:
            self.client.enableApiControl(True, vehicle)
            self.client.armDisarm(True, vehicle)

        self.client.takeoffAsync(vehicle_name=TRACKER_VEHICLE).join()
        self.client.takeoffAsync(vehicle_name=TARGET_VEHICLE).join()

        setup_detector(self.client)

        target_pose = self.client.simGetVehiclePose(TARGET_VEHICLE)
        tp = target_pose.position
        origin = np.array([tp.x_val, tp.y_val, -tp.z_val])

        idx = self._episode_count % len(TRAJECTORY_TRAIN_PRESETS)
        self._trajectory = TRAJECTORY_TRAIN_PRESETS[idx](origin)
        self._episode_count += 1

        self._follower = TrajectoryFollower(self._trajectory, TARGET_VEHICLE, dt=TS)
        self._follower.start()

        self._step_count = 0

        self._frames_without_detection = 0

        frame = capture_depth(self.client)
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

        frame = capture_depth(self.client)
        self._frame_buffer.append(frame)
        self._frame_buffer.pop(0)

        obs = self._get_obs()

        rel_pos = obs["critic_state"][:3]
        reward, done = self._compute_reward(rel_pos)

        # Terminate if target lost for too long
        if self._frames_without_detection >= self._max_lost_steps:
            reward = -10.0
            done = True

        truncated = self._step_count >= self.max_episode_steps

        if self.show_cv:
            self._render_cv(obs, reward)

        info = {
            "distance": float(np.linalg.norm(rel_pos)),
            "t_elapsed": self._follower.elapsed_time,
        }

        return obs, reward, done, truncated, info

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
        bbox = self._get_bbox()

        return {
            "image": stacked,
            "bbox": bbox,
            "critic_state": critic_state,
        }

    def _get_bbox(self) -> np.ndarray:
        bbox, detected = get_relative_bbox(
            self.client, self._raw_w, self._raw_h, self.image_size
        )
        if detected:
            self._frames_without_detection = 0
        else:
            self._frames_without_detection += 1
        return bbox

    def _render_cv(self, obs, reward):
        dist = float(np.linalg.norm(obs["critic_state"][:3]))
        render_depth_with_bbox(
            obs["image"][-3:],
            obs["bbox"],
            self.image_size,
            reward=reward,
            dist=dist,
            frames_lost=self._frames_without_detection,
        )

    def _compute_reward(self, relative_pos):
        x, y, z = relative_pos
        dist = np.linalg.norm(relative_pos)

        if abs(x) < 0.01:
            x = 0.01

        fov_half = np.pi / 4
        # Angular errors
        y_err = abs(np.arctan(y / x) / fov_half)
        z_err = abs(np.arctan(z / x) / fov_half)
        x_err = abs(x - self.desired_distance) / self.desired_distance

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
