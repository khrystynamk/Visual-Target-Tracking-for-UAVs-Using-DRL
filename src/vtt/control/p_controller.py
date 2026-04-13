"""
Baseline visual tracker: AirSim oracle bbox + P-controller.

Control strategy:
  1. Yaw — rotate to keep target centered horizontally (cx error)
  2. Forward — move forward/back to maintain distance (bbox area error)
  3. Vertical — move up/down to keep target centered vertically (cy error)
"""

import numpy as np

from vtt.constants import IMG_W, IMG_H, MAX_SPEED

TARGET_AREA = (IMG_W * 0.2) * (IMG_H * 0.2)

KP_YAW = 0.5
KP_FWD = MAX_SPEED / TARGET_AREA
KP_VERT = MAX_SPEED / (IMG_H / 2)

# Ignore small corrections from bbox
FWD_DEADBAND = TARGET_AREA * 0.10


def compute_control(cx: float, cy: float, area: float):
    """
    Args:
      cx: detected bbox center x [px]
      cy: detected bbox center y [px]
      area: detected bbox area [px^2]

    Returns:
      (vx_body, vz_ned, yaw_rate) — forward speed, vertical speed, yaw rate
    """
    cx_err = cx - IMG_W / 2.0
    cy_err = cy - IMG_H / 2.0
    area_err = TARGET_AREA - area

    yaw_rate = KP_YAW * cx_err

    if abs(area_err) < FWD_DEADBAND:
        vx_body = 0.0
    else:
        vx_body = np.clip(KP_FWD * area_err, -MAX_SPEED, MAX_SPEED)

    vz_ned = np.clip(KP_VERT * cy_err, -MAX_SPEED, MAX_SPEED)

    return float(vx_body), float(vz_ned), float(yaw_rate)
