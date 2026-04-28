"""
Evaluate and compare the P-controller baseline and SAC agent on multiple trajectories (sinusoidal, circular, and figure-8).

Usage:
  python scripts/evaluate.py --model experiments/sac_depth/best/best_model.zip
  python scripts/evaluate.py --model experiments/sac_depth/final_model.zip --duration 20
"""

import argparse
import json
import os
import time

import numpy as np
import airsim
from stable_baselines3 import SAC

from vtt.constants import (
    TRACKER_VEHICLE,
    TARGET_VEHICLE,
    TS,
    FRAME_STACK,
    MAX_VEL,
    MAX_YAW_RATE,
)
from vtt.control.p_controller import compute_control
from vtt.utils.camera_helpers import (
    get_raw_camera_resolution,
    capture_frame,
    setup_detector,
    detect,
    capture_depth_raw,
    get_relative_bbox,
)
from vtt.target.trajectory_follower import TrajectoryFollower
from vtt.metrics.trajectory_comparison import (
    TrajectoryRecord,
    compute_tracking_errors,
    plot_trajectory_comparison,
)
from vtt.envs.tracking_env import TrackingEnv, _quat_to_rotation_matrix
from vtt.utils.common_utils import get_target_origin
from vtt.metrics.constants import TRAJECTORY_EVAL_PRESETS
from vtt.metrics.tracker_metrics import Metrics


_ALL_WEATHER_PARAMS = [
    airsim.WeatherParameter.Rain,
    airsim.WeatherParameter.Roadwetness,
    airsim.WeatherParameter.Snow,
    airsim.WeatherParameter.RoadSnow,
    airsim.WeatherParameter.MapleLeaf,
    airsim.WeatherParameter.RoadLeaf,
    airsim.WeatherParameter.Dust,
    airsim.WeatherParameter.Fog,
]

WEATHER_CONDITIONS = {
    "default": (None, {}),
    "rain": (None, {airsim.WeatherParameter.Rain: 0.5}),
    "fog": (None, {airsim.WeatherParameter.Fog: 0.4}),
    "dust": (None, {airsim.WeatherParameter.Dust: 0.5}),
    "dawn": ("2024-06-15 05:30:00", {}),
    "dawn_rain": ("2024-06-15 05:30:00", {airsim.WeatherParameter.Rain: 0.5}),
}


def apply_condition(client, condition: str):
    """Apply a weather and/or lighting condition."""
    time_of_day, weather = WEATHER_CONDITIONS[condition]

    # Lighting
    if time_of_day is not None:
        client.simSetTimeOfDay(True, start_datetime=time_of_day, move_sun=True)
    else:
        client.simSetTimeOfDay(False)

    # Weather
    if weather:
        client.simEnableWeather(True)
        for param in _ALL_WEATHER_PARAMS:
            client.simSetWeatherParameter(param, 0.0)
        for param, val in weather.items():
            client.simSetWeatherParameter(param, val)
    else:
        client.simEnableWeather(False)


def setup_airsim(client, condition: str = "default"):
    client.simPause(True)
    client.reset()

    for vehicle in [TRACKER_VEHICLE, TARGET_VEHICLE]:
        client.enableApiControl(True, vehicle)
        client.armDisarm(True, vehicle)

    client.simPause(False)
    client.takeoffAsync(vehicle_name=TRACKER_VEHICLE).join()
    client.takeoffAsync(vehicle_name=TARGET_VEHICLE).join()
    # Move both drones to a consistent altitude (negative = up)
    client.moveToZAsync(-2.0, 1.0, vehicle_name=TRACKER_VEHICLE)
    client.moveToZAsync(-2.0, 1.0, vehicle_name=TARGET_VEHICLE).join()

    apply_condition(client, condition)


def run_baseline(client, trajectory, duration, raw_w, raw_h):
    """
    Run P-controller baseline, return (tracker_record, target_record, metrics).
    """
    follower = TrajectoryFollower(trajectory, TARGET_VEHICLE, dt=TS)
    follower.start()

    tracker_record = TrajectoryRecord(name="Baseline (P)")
    target_record = TrajectoryRecord(name="Target")
    metrics = Metrics()
    t_start = time.time()

    while True:
        t0 = time.time()
        t_elapsed = t0 - t_start
        if t_elapsed >= duration:
            break

        capture_frame(client)
        det = detect(client, raw_w, raw_h)

        vx_body = vz = 0.0
        yaw_rate = 0.0
        if det is not None:
            cx, cy, area, _ = det
            vx_body, vz, yaw_rate = compute_control(cx, cy, area)

        tracker_state = client.getMultirotorState(vehicle_name=TRACKER_VEHICLE)
        _, _, yaw = airsim.to_eularian_angles(
            tracker_state.kinematics_estimated.orientation
        )
        vx_world = vx_body * np.cos(yaw)
        vy_world = vx_body * np.sin(yaw)

        client.moveByVelocityAsync(
            vx_world,
            vy_world,
            vz,
            TS,
            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
            yaw_mode=airsim.YawMode(True, yaw_rate),
            vehicle_name=TRACKER_VEHICLE,
        )

        sp = tracker_state.kinematics_estimated.position
        tp_pos = client.simGetVehiclePose(TARGET_VEHICLE).position
        dist3d = float(
            np.sqrt(
                (sp.x_val - tp_pos.x_val) ** 2
                + (sp.y_val - tp_pos.y_val) ** 2
                + (sp.z_val - tp_pos.z_val) ** 2
            )
        )

        if det is not None:
            from vtt.constants import IMG_W, IMG_H

            metrics.update(True, cx - IMG_W / 2.0, cy - IMG_H / 2.0, dist3d)
        else:
            metrics.update(False, 0.0, 0.0, dist3d)

        tracker_record.append(t_elapsed, np.array([sp.x_val, sp.y_val, -sp.z_val]))
        target_record.append(
            t_elapsed, np.array([tp_pos.x_val, tp_pos.y_val, -tp_pos.z_val])
        )

        elapsed = time.time() - t0
        if elapsed < TS:
            time.sleep(TS - elapsed)

    follower.stop()
    client.moveByVelocityAsync(0, 0, 0, 1, vehicle_name=TRACKER_VEHICLE)
    return tracker_record, target_record, metrics


def run_sac_agent(client, model, trajectory, duration, raw_w, raw_h):
    """
    Run trained SAC agent, return (tracker_record, target_record, metrics).
    """
    follower = TrajectoryFollower(trajectory, TARGET_VEHICLE, dt=TS)
    follower.start()

    tracker_record = TrajectoryRecord(name="SAC (Depth)")
    target_record = TrajectoryRecord(name="Target")
    metrics = Metrics()

    frame = capture_depth_raw(client)[np.newaxis]  # (1, H, W)
    frame_buffer = [frame] * FRAME_STACK

    t_start = time.time()

    while True:
        t0 = time.time()
        t_elapsed = t0 - t_start
        if t_elapsed >= duration:
            break
        stacked = np.concatenate(frame_buffer, axis=0)  # (S, H, W)

        tracker = client.getMultirotorState(vehicle_name=TRACKER_VEHICLE)
        target_pose = client.simGetVehiclePose(TARGET_VEHICLE)
        target_state = client.getMultirotorState(vehicle_name=TARGET_VEHICLE)

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
        critic_state = np.concatenate([rel_pos, rel_vel, rel_acc]).astype(np.float32)

        bbox, _ = get_relative_bbox(client, raw_w, raw_h)
        obs = {"image": stacked, "bbox": bbox, "critic_state": critic_state}
        action, _ = model.predict(obs, deterministic=True)

        vx_body = float(action[0]) * MAX_VEL
        vy_body = float(action[1]) * MAX_VEL
        vz_ned = float(action[2]) * MAX_VEL
        yaw_rate = float(action[3]) * MAX_YAW_RATE

        _, _, yaw = airsim.to_eularian_angles(tracker.kinematics_estimated.orientation)
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

        frame = capture_depth_raw(client)[np.newaxis]  # (1, H, W)
        frame_buffer.append(frame)
        frame_buffer.pop(0)

        # Metrics: use oracle detector to check if target is in frame
        det = detect(client, raw_w, raw_h)
        dist3d = float(np.linalg.norm(tgt_pos - t_pos))
        if det is not None:
            from vtt.constants import IMG_W, IMG_H

            cx, cy, _, _ = det
            metrics.update(True, cx - IMG_W / 2.0, cy - IMG_H / 2.0, dist3d)
        else:
            metrics.update(False, 0.0, 0.0, dist3d)

        tracker_record.append(t_elapsed, np.array([t_pos[0], t_pos[1], -t_pos[2]]))
        target_record.append(t_elapsed, np.array([tgt_pos[0], tgt_pos[1], -tgt_pos[2]]))

        elapsed = time.time() - t0
        if elapsed < TS:
            time.sleep(TS - elapsed)

    follower.stop()
    client.moveByVelocityAsync(0, 0, 0, 1, vehicle_name=TRACKER_VEHICLE)
    return tracker_record, target_record, metrics


def main():
    valid_trajectories = list(TRAJECTORY_EVAL_PRESETS.keys())
    parser = argparse.ArgumentParser(description="Evaluate baseline vs SAC agent")
    parser.add_argument("--model", required=True, help="Path to trained SAC model .zip")
    parser.add_argument(
        "--trajectory",
        required=True,
        choices=valid_trajectories,
        help="Trajectory to evaluate",
    )
    parser.add_argument(
        "--duration", type=float, default=15.0, help="Episode duration (s)"
    )
    parser.add_argument(
        "--condition",
        default="default",
        choices=list(WEATHER_CONDITIONS.keys()),
        help="Weather/lighting condition",
    )
    parser.add_argument("--runs", type=int, default=5, help="Number of runs per method")
    parser.add_argument(
        "--save-dir", default="experiments/evaluation", help="Output dir"
    )
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    client = airsim.MultirotorClient()
    client.confirmConnection()

    env = TrackingEnv()
    model = SAC.load(args.model, env=env)
    print(f"Loaded SAC model from {args.model}")

    raw_w, raw_h = get_raw_camera_resolution(client)
    setup_detector(client)

    traj_name = args.trajectory
    condition = args.condition
    n_runs = args.runs
    METRIC_KEYS = [
        "rmse",
        "mean_error",
        "max_error",
        "detect_rate_%",
        "mean_dist3d_m",
        "loss_events",
    ]

    print(f"\n{'=' * 60}")
    print(f"Trajectory: {traj_name}  |  Condition: {condition}  |  Runs: {n_runs}")
    print(f"{'=' * 60}")

    all_runs = {"baseline": [], "sac": []}

    for run_idx in range(n_runs):
        print(f"\n--- Run {run_idx + 1}/{n_runs} ---")

        # --- Baseline ---
        print("Running P-controller baseline...")
        setup_airsim(client, condition)
        origin = get_target_origin(client)
        trajectory = TRAJECTORY_EVAL_PRESETS[traj_name](origin)

        baseline_tracker, baseline_target, baseline_metrics = run_baseline(
            client, trajectory, args.duration, raw_w, raw_h
        )

        bt, bp = baseline_tracker.as_arrays()
        tt, tp = baseline_target.as_arrays()
        if len(bt) > 1:
            baseline_errors = compute_tracking_errors(tt, tp, bt, bp)
            baseline_summary = baseline_metrics.summary()
            run_result = {
                **baseline_summary,
                "rmse": baseline_errors["rmse"],
                "mean_error": baseline_errors["mean_error"],
                "max_error": baseline_errors["max_error"],
                "per_axis_rmse": baseline_errors["per_axis_rmse"].tolist(),
            }
            all_runs["baseline"].append(run_result)
            print(
                f"Baseline RMSE: {baseline_errors['rmse']:.3f} m, "
                f"detect_rate: {baseline_summary['detect_rate_%']}%"
            )

            # Save trajectory plot for best run (first run)
            if run_idx == 0:
                plot_trajectory_comparison(
                    tt,
                    tp,
                    bt,
                    bp,
                    title=f"Baseline (P) — {traj_name} [{condition}]",
                    show=False,
                    save_path=os.path.join(
                        args.save_dir, f"baseline_{traj_name}_{condition}.png"
                    ),
                )

        # --- SAC Agent ---
        print("Running SAC agent...")
        setup_airsim(client, condition)
        setup_detector(client)
        origin = get_target_origin(client)
        trajectory = TRAJECTORY_EVAL_PRESETS[traj_name](origin)

        sac_tracker, sac_target, sac_metrics = run_sac_agent(
            client, model, trajectory, args.duration, raw_w, raw_h
        )

        st, sp_sac = sac_tracker.as_arrays()
        stt, stp = sac_target.as_arrays()
        if len(st) > 1:
            sac_errors = compute_tracking_errors(stt, stp, st, sp_sac)
            sac_summary = sac_metrics.summary()
            run_result = {
                **sac_summary,
                "rmse": sac_errors["rmse"],
                "mean_error": sac_errors["mean_error"],
                "max_error": sac_errors["max_error"],
                "per_axis_rmse": sac_errors["per_axis_rmse"].tolist(),
            }
            all_runs["sac"].append(run_result)
            print(
                f"SAC RMSE: {sac_errors['rmse']:.3f} m, "
                f"detect_rate: {sac_summary['detect_rate_%']}%"
            )

            if run_idx == 0:
                plot_trajectory_comparison(
                    stt,
                    stp,
                    st,
                    sp_sac,
                    title=f"SAC (Depth) — {traj_name} [{condition}]",
                    show=False,
                    save_path=os.path.join(
                        args.save_dir, f"sac_{traj_name}_{condition}.png"
                    ),
                )

    print(f"\n{'=' * 60}")
    print(f"SUMMARY — {traj_name} | {condition} ({n_runs} runs)")
    print(f"{'=' * 60}")
    header = f"{'Method':<15} {'RMSE (m)':<18} {'Mean (m)':<18} {'Max (m)':<18} {'Detect%':<18} {'Dist3D (m)':<18} {'Losses':<14}"
    print(header)
    print("-" * len(header))

    aggregated = {}
    for method, label in [("baseline", "P-controller"), ("sac", "SAC (Depth)")]:
        runs = all_runs[method]
        if not runs:
            continue
        agg = {}
        for key in METRIC_KEYS:
            vals = [r[key] for r in runs if key in r]
            agg[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

        aggregated[method] = {"per_run": runs, "aggregated": agg}
        print(
            f"{label:<15} "
            f"{agg['rmse']['mean']:.3f}±{agg['rmse']['std']:.3f}  "
            f"{agg['mean_error']['mean']:.3f}±{agg['mean_error']['std']:.3f}  "
            f"{agg['max_error']['mean']:.3f}±{agg['max_error']['std']:.3f}  "
            f"{agg['detect_rate_%']['mean']:.1f}±{agg['detect_rate_%']['std']:.1f}    "
            f"{agg['mean_dist3d_m']['mean']:.3f}±{agg['mean_dist3d_m']['std']:.3f}  "
            f"{agg['loss_events']['mean']:.2f}±{agg['loss_events']['std']:.2f}"
        )

    metrics_path = os.path.join(args.save_dir, f"eval_{traj_name}_{condition}.json")
    with open(metrics_path, "w") as f:
        json.dump({traj_name: aggregated}, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")

    env.close()


if __name__ == "__main__":
    main()
