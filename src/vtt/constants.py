"""
Shared constants for the VTT project.

Some of them must match the AirSim settings in configs/airsim/settings.json and sac_depth.yaml.
"""

TRACKER_VEHICLE = "Drone1"
TARGET_VEHICLE = "Target"
TARGET_MESH_NAME = "Target"

TRACKER_CAMERA = "front_center"
DETECTION_RADIUS_CM = 20000  # 200 m

IMG_W = 224
IMG_H = 224

TS = 0.05  # control loop timestep [s] (20 Hz)

ACCELERATION = 4.0  # m/s^2
MAX_VEL = 5.0  # m/s
MAX_YAW_RATE = 229.0  # deg/s  (~4 rad/s)
TICK = TS * 8
FRICTION = 0.5

IMAGE_SIZE = 224
FRAME_STACK = 3
MAX_EPISODE_STEPS = 300  # 15s at 20Hz
DESIRED_DISTANCE = 2.0
MAX_DISTANCE = 8.0
MIN_DISTANCE = 0.3
