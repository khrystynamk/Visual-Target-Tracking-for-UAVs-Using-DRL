"""
Shared constants for the VTT project.

Some of them must match the AirSim settings in configs/airsim/settings.json.
"""

TRACKER_VEHICLE = "Drone1"
TARGET_VEHICLE = "Target"
TARGET_MESH_NAME = "Target"

TRACKER_CAMERA = "front_center"
DETECTION_RADIUS_CM = 20000  # 200 m

IMG_W = 480
IMG_H = 480

TS = 0.05  # control loop timestep [s] (20 Hz)

ACCELERATION = 4.0  # m/s^2
MAX_SPEED = 8.0  # m/s
YAW_RATE = 90.0  # deg/s
TICK = TS * 8  # command duration [s] (0.4s at TS=0.05)
FRICTION = 0.5
