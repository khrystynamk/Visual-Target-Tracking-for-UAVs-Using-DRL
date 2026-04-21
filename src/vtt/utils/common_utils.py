import numpy as np
from vtt.constants import TARGET_VEHICLE


def get_target_origin(client) -> np.ndarray:
    pose = client.simGetVehiclePose(TARGET_VEHICLE)
    p = pose.position
    return np.array([p.x_val, p.y_val, -p.z_val])
