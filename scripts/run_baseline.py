"""
Run the baseline tracker against a scripted target trajectory.

Usage:
  python scripts/run_baseline_p.py --trajectory sinusoidal --duration 40
  python scripts/run_baseline_p.py --trajectory helix --duration 60
  python scripts/run_baseline_p.py --trajectory figure_eight --duration 40
  python scripts/run_baseline_p.py --trajectory keyboard  # manual control

Available trajectories:
  sinusoidal, helix, figure_eight, circular, keyboard
"""

import argparse
import json
import os
import time

import airsim
import cv2
import numpy as np

from vtt.constants import TRACKER_VEHICLE, TARGET_VEHICLE, IMG_W, IMG_H, TS
from vtt.metrics.tracker_metrics import Metrics
from vtt.target.keyboard_controller import DroneController
from vtt.utils.camera_helpers import (
    get_raw_camera_resolution,
    capture_frame,
    setup_detector,
    detect,
)
from vtt.control.p_controller import compute_control
from vtt.metrics.scripted_trajectories import (
    SinusoidalTrajectory,
    CircularTrajectory,
    FigureEightTrajectory,
)
from vtt.target.trajectory_follower import TrajectoryFollower
from vtt.metrics.trajectory_comparison import (
    TrajectoryRecord,
    compute_tracking_errors,
    plot_trajectory_comparison,
)


def _get_target_origin(client) -> np.ndarray:
    pose = client.simGetVehiclePose("Target")
    p = pose.position
    return np.array([p.x_val, p.y_val, -p.z_val])


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


def parse_args():
    parser = argparse.ArgumentParser(description="Run P-controller baseline tracker")
    parser.add_argument(
        "--trajectory",
        "-t",
        choices=list(TRAJECTORY_PRESETS.keys()) + ["keyboard"],
        default="sinusoidal",
        help="Target trajectory preset or 'keyboard' for manual control",
    )
    parser.add_argument(
        "--duration",
        "-d",
        type=float,
        default=40.0,
        help="Duration in seconds (ignored for keyboard mode)",
    )
    parser.add_argument(
        "--no-img",
        type=bool,
        default=True,
        help="Do not show additional image with detected target drone",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip the comparison plot at the end",
    )
    parser.add_argument(
        "--save-dir",
        default="experiments/baseline_p",
        help="Directory to save metrics and plots",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    use_keyboard = args.trajectory == "keyboard"

    client = airsim.MultirotorClient()
    client.confirmConnection()
    print("Connected to AirSim.")

    client.enableApiControl(True, TRACKER_VEHICLE)
    client.armDisarm(True, TRACKER_VEHICLE)
    client.takeoffAsync(vehicle_name=TRACKER_VEHICLE).join()

    raw_w, raw_h = get_raw_camera_resolution(client)
    setup_detector(client)

    target_ctrl = None
    follower = None

    if use_keyboard:
        target_ctrl = DroneController()
        target_ctrl.start()
    else:
        client.enableApiControl(True, TARGET_VEHICLE)
        client.armDisarm(True, TARGET_VEHICLE)
        client.takeoffAsync(vehicle_name=TARGET_VEHICLE).join()
        origin = _get_target_origin(client)

        trajectory = TRAJECTORY_PRESETS[args.trajectory](origin)
        print(f"Starting trajectory: {args.trajectory} (duration={args.duration}s)")
        follower = TrajectoryFollower(trajectory, TARGET_VEHICLE, dt=TS)
        follower.start()

    time.sleep(0.5)

    metrics = Metrics()
    tracker_record = TrajectoryRecord(name="Tracker")
    target_record = TrajectoryRecord(name="Target (actual)")
    t_start = time.time()

    while True:
        t0 = time.time()
        t_elapsed = t0 - t_start

        if not use_keyboard and t_elapsed >= args.duration:
            print(f"Duration {args.duration}s reached.")
            break

        frame = capture_frame(client)
        det = detect(client, raw_w, raw_h)
        vx_body = vz = 0.0
        yaw_rate = 0.0

        if det is not None:
            cx, cy, area, (x1, y1, x2, y2) = det
            vx_body, vz, yaw_rate = compute_control(cx, cy, area)

            if not args.no_img:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, (int(cx), int(cy)), 5, (0, 0, 255), -1)
                cv2.putText(
                    frame,
                    f"err: ({cx - IMG_W / 2:.0f}, {cy - IMG_H / 2:.0f})  area: {area:.0f}",
                    (4, 16),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 255, 0),
                    1,
                )
        else:
            if not args.no_img:
                cv2.putText(
                    frame,
                    "NO DETECTION",
                    (4, 16),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1,
                )

        tracker_state = client.getMultirotorState(vehicle_name=TRACKER_VEHICLE)
        _, _, yaw = airsim.to_eularian_angles(
            tracker_state.kinematics_estimated.orientation
        )
        vx_world = vx_body * np.cos(yaw)
        vy_world = vx_body * np.sin(yaw)

        sp = tracker_state.kinematics_estimated.position
        tracker_record.append(t_elapsed, np.array([sp.x_val, sp.y_val, -sp.z_val]))

        target_pose = client.simGetVehiclePose(TARGET_VEHICLE)
        tp = target_pose.position
        target_record.append(t_elapsed, np.array([tp.x_val, tp.y_val, -tp.z_val]))

        dist3d = float(
            np.sqrt(
                (sp.x_val - tp.x_val) ** 2
                + (sp.y_val - tp.y_val) ** 2
                + (sp.z_val - tp.z_val) ** 2
            )
        )
        if det is not None:
            metrics.update(True, cx - IMG_W / 2.0, cy - IMG_H / 2.0, dist3d)
        else:
            metrics.update(False, 0.0, 0.0, dist3d)

        client.moveByVelocityAsync(
            vx_world,
            vy_world,
            vz,
            TS,
            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
            yaw_mode=airsim.YawMode(True, yaw_rate),
            vehicle_name=TRACKER_VEHICLE,
        )

        if not args.no_img:
            cv2.imshow("Tracker (P-control)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        elapsed = time.time() - t0
        if elapsed < TS:
            time.sleep(TS - elapsed)

    if target_ctrl:
        target_ctrl.stop()
    if follower:
        follower.stop()
    client.moveByVelocityAsync(0.0, 0.0, 0.0, 1.0, vehicle_name=TRACKER_VEHICLE)
    cv2.destroyAllWindows()
    print("Stopped.\n")

    summary = metrics.summary()
    print("--- Tracking Metrics (P-controller) ---")
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

    os.makedirs(args.save_dir, exist_ok=True)
    run_id = f"p_{args.trajectory}_{int(time.time())}"

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
            title=f"P-Controller Baseline — {args.trajectory}",
            save_path=plot_path,
            show=True,
        )


if __name__ == "__main__":
    main()
