"""
Run a trained SAC agent against a scripted target trajectory.

Usage:
  python scripts/run_trained.py --model experiments/sac_depth/best/best_model.zip
  python scripts/run_trained.py --model experiments/sac_depth/best/best_model.zip --trajectory circular --duration 40
  python scripts/run_trained.py --model experiments/sac_depth/best/best_model.zip --trajectory figure_eight

Available trajectories:
  sinusoidal, helix, figure_eight, circular
"""

import argparse
import json
import os
import time

import airsim
import numpy as np
from stable_baselines3 import SAC

from vtt.constants import (
    TRACKER_VEHICLE,
    TARGET_VEHICLE,
    IMG_W,
    IMG_H,
    TS,
    FRAME_STACK,
    MAX_VEL,
    MAX_YAW_RATE,
)
from vtt.envs.tracking_env import TrackingEnv, _quat_to_rotation_matrix
from vtt.metrics.tracker_metrics import Metrics
from vtt.utils.camera_helpers import (
    get_raw_camera_resolution,
    capture_depth_raw,
    setup_detector,
    get_relative_bbox,
)
from vtt.target.trajectory_follower import TrajectoryFollower
from vtt.metrics.trajectory_comparison import (
    TrajectoryRecord,
    compute_tracking_errors,
    plot_trajectory_comparison,
)
from vtt.utils.common_utils import get_target_origin
from vtt.metrics.constants import TRAJECTORY_EVAL_PRESETS


def parse_args():
    parser = argparse.ArgumentParser(description="Run trained SAC agent")
    parser.add_argument(
        "--model", "-m", required=True, help="Path to trained SAC model .zip"
    )
    parser.add_argument(
        "--trajectory",
        "-t",
        choices=list(TRAJECTORY_EVAL_PRESETS.keys()),
        default="sinusoidal",
        help="Target trajectory preset",
    )
    parser.add_argument(
        "--duration",
        "-d",
        type=float,
        default=40.0,
        help="Duration in seconds",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip the comparison plot at the end",
    )
    parser.add_argument(
        "--save-dir",
        default="experiments/sac_runs",
        help="Directory to save metrics and plots",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # --- Load model ---
    env = TrackingEnv()
    model = SAC.load(args.model, env=env, buffer_size=1)
    print(f"Loaded SAC model from {args.model}")

    # Reuse the env's AirSim client to avoid RPC contention
    client = env.client
    print("Connected to AirSim.")

    client.enableApiControl(True, TRACKER_VEHICLE)
    client.armDisarm(True, TRACKER_VEHICLE)
    client.takeoffAsync(vehicle_name=TRACKER_VEHICLE).join()

    client.enableApiControl(True, TARGET_VEHICLE)
    client.armDisarm(True, TARGET_VEHICLE)
    client.takeoffAsync(vehicle_name=TARGET_VEHICLE).join()

    raw_w, raw_h = get_raw_camera_resolution(client)
    setup_detector(client)
    origin = get_target_origin(client)

    trajectory = TRAJECTORY_EVAL_PRESETS[args.trajectory](origin)
    print(f"Starting trajectory: {args.trajectory} (duration={args.duration}s)")

    # --- Initialize frame buffer (before follower to avoid RPC contention) ---
    print("Testing RPC: simGetVehiclePose...")
    print(client.simGetVehiclePose(TRACKER_VEHICLE))
    print("RPC works. Testing Scene image...")
    print(f"Scene image: {get_raw_camera_resolution(client)}")
    print("Scene works. Capturing depth frame...")
    frame = capture_depth_raw(client)[np.newaxis]  # (1, H, W)
    print("Frame captured.")

    follower = TrajectoryFollower(trajectory, TARGET_VEHICLE, dt=TS)
    follower.start()
    frame_buffer = [frame] * FRAME_STACK

    metrics = Metrics()
    tracker_record = TrajectoryRecord(name="SAC (Depth)")
    target_record = TrajectoryRecord(name="Target (actual)")
    t_start = time.time()

    while True:
        t0 = time.time()
        t_elapsed = t0 - t_start

        if t_elapsed >= args.duration:
            print(f"Duration {args.duration}s reached.")
            break

        # --- Build observation ---
        stacked = np.concatenate(frame_buffer, axis=0)  # (S, H, W)

        tracker_state = client.getMultirotorState(vehicle_name=TRACKER_VEHICLE)
        target_pose = client.simGetVehiclePose(TARGET_VEHICLE)
        target_state = client.getMultirotorState(vehicle_name=TARGET_VEHICLE)

        t_pos = np.array(
            [
                tracker_state.kinematics_estimated.position.x_val,
                tracker_state.kinematics_estimated.position.y_val,
                tracker_state.kinematics_estimated.position.z_val,
            ]
        )
        t_vel = np.array(
            [
                tracker_state.kinematics_estimated.linear_velocity.x_val,
                tracker_state.kinematics_estimated.linear_velocity.y_val,
                tracker_state.kinematics_estimated.linear_velocity.z_val,
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
                tracker_state.kinematics_estimated.linear_acceleration.x_val,
                tracker_state.kinematics_estimated.linear_acceleration.y_val,
                tracker_state.kinematics_estimated.linear_acceleration.z_val,
            ]
        )
        tgt_acc = np.array(
            [
                target_state.kinematics_estimated.linear_acceleration.x_val,
                target_state.kinematics_estimated.linear_acceleration.y_val,
                target_state.kinematics_estimated.linear_acceleration.z_val,
            ]
        )

        q = tracker_state.kinematics_estimated.orientation
        R = _quat_to_rotation_matrix(q).T
        rel_pos = R @ (tgt_pos - t_pos)
        rel_vel = R @ (tgt_vel - t_vel)
        rel_acc = R @ (tgt_acc - t_acc)
        critic_state = np.concatenate([rel_pos, rel_vel, rel_acc]).astype(np.float32)

        bbox, _ = get_relative_bbox(client, raw_w, raw_h)

        obs = {
            "image": stacked,
            "bbox": bbox,
            "critic_state": critic_state,
        }

        # --- Get action from policy ---
        print("Predicting....")
        action, _ = model.predict(obs, deterministic=True)

        vx_body = float(action[0]) * MAX_VEL
        vy_body = float(action[1]) * MAX_VEL
        vz_ned = float(action[2]) * MAX_VEL
        yaw_rate = float(action[3]) * MAX_YAW_RATE

        print(
            f"action: {action}, vx={vx_body:.2f}, vy={vy_body:.2f}, vz={vz_ned:.2f}, yaw={yaw_rate:.2f}"
        )

        _, _, yaw = airsim.to_eularian_angles(
            tracker_state.kinematics_estimated.orientation
        )
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)
        vx_world = vx_body * cos_y - vy_body * sin_y
        vy_world = vx_body * sin_y + vy_body * cos_y

        client.moveByVelocityAsync(
            vx_world,
            vy_world,
            vz_ned,
            TS,
            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
            yaw_mode=airsim.YawMode(True, yaw_rate),
            vehicle_name=TRACKER_VEHICLE,
        )

        # --- Update depth frame buffer ---
        frame = capture_depth_raw(client)[np.newaxis]  # (1, H, W)
        frame_buffer.append(frame)
        frame_buffer.pop(0)

        # --- Record metrics ---
        dist3d = float(np.linalg.norm(tgt_pos - t_pos))
        detected = bbox[2] > 0.0 and bbox[3] > 0.0  # w, h > 0 means detection
        if detected:
            cx_px = bbox[0] * IMG_W
            cy_px = bbox[1] * IMG_H
            metrics.update(True, cx_px - IMG_W / 2.0, cy_px - IMG_H / 2.0, dist3d)
        else:
            metrics.update(False, 0.0, 0.0, dist3d)

        tracker_record.append(t_elapsed, np.array([t_pos[0], t_pos[1], -t_pos[2]]))
        target_record.append(t_elapsed, np.array([tgt_pos[0], tgt_pos[1], -tgt_pos[2]]))

        elapsed = time.time() - t0
        if elapsed < TS:
            time.sleep(TS - elapsed)

    # --- Cleanup ---
    follower.stop()
    client.moveByVelocityAsync(0.0, 0.0, 0.0, 1.0, vehicle_name=TRACKER_VEHICLE)
    print("Stopped.\n")

    # --- Print metrics ---
    summary = metrics.summary()
    print("--- Tracking Metrics (SAC Agent) ---")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    target_t, target_pos = target_record.as_arrays()
    tracker_t, tracker_pos = tracker_record.as_arrays()

    if len(target_t) > 1 and len(tracker_t) > 1:
        errors = compute_tracking_errors(target_t, target_pos, tracker_t, tracker_pos)
        print("\n--- Trajectory Errors ---")
        print(f"  RMSE:       {errors['rmse']:.3f} m")
        print(f"  Mean error: {errors['mean_error']:.3f} m")
        print(f"  Max error:  {errors['max_error']:.3f} m")
        print(
            f"  Per-axis RMSE: X={errors['per_axis_rmse'][0]:.3f}, "
            f"Y={errors['per_axis_rmse'][1]:.3f}, Z={errors['per_axis_rmse'][2]:.3f} m"
        )

        summary["trajectory_rmse_m"] = errors["rmse"]
        summary["trajectory_mean_error_m"] = errors["mean_error"]
        summary["trajectory_max_error_m"] = errors["max_error"]

    # --- Save results ---
    os.makedirs(args.save_dir, exist_ok=True)
    run_id = f"sac_{args.trajectory}_{int(time.time())}"

    metrics_path = os.path.join(args.save_dir, f"{run_id}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")

    traj_path = os.path.join(args.save_dir, f"{run_id}_trajectories.npz")
    np.savez(
        traj_path,
        target_times=target_t,
        target_positions=target_pos,
        tracker_times=tracker_t,
        tracker_positions=tracker_pos,
    )
    print(f"Trajectories saved to {traj_path}")

    if not args.no_plot and len(target_t) > 1 and len(tracker_t) > 1:
        plot_path = os.path.join(args.save_dir, f"{run_id}_comparison.png")
        plot_trajectory_comparison(
            target_t,
            target_pos,
            tracker_t,
            tracker_pos,
            title=f"SAC (Depth) — {args.trajectory}",
            save_path=plot_path,
            show=True,
        )

    env.close()


if __name__ == "__main__":
    main()
