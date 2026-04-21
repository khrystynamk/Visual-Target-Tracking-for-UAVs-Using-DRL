import airsim
import numpy as np
import cv2

from vtt.constants import (
    IMAGE_SIZE,
    MAX_DISTANCE,
    TRACKER_VEHICLE,
    TARGET_MESH_NAME,
    TRACKER_CAMERA,
    DETECTION_RADIUS_CM,
    IMG_W,
    IMG_H,
)


def get_raw_camera_resolution(client: airsim.MultirotorClient):
    resp = client.simGetImages(
        [airsim.ImageRequest(TRACKER_CAMERA, airsim.ImageType.Scene, False, False)],
        vehicle_name=TRACKER_VEHICLE,
    )[0]
    return resp.width, resp.height


def capture_frame(client: airsim.MultirotorClient):
    responses = client.simGetImages(
        [airsim.ImageRequest(TRACKER_CAMERA, airsim.ImageType.Scene, False, False)],
        vehicle_name=TRACKER_VEHICLE,
    )
    img1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
    frame = img1d.reshape(responses[0].height, responses[0].width, 3)
    return cv2.resize(frame, (IMG_W, IMG_H))


def capture_depth(client) -> np.ndarray:
    """
    Capture a single depth frame for the SAC agent. Returns (1, H, W).
    """
    responses = client.simGetImages(
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
    depth = np.clip(depth, 0.0, MAX_DISTANCE) / MAX_DISTANCE
    depth = cv2.resize(depth, (IMAGE_SIZE, IMAGE_SIZE))
    return depth[np.newaxis].astype(np.float32)


def setup_detector(client: airsim.MultirotorClient):
    client.simSetDetectionFilterRadius(
        TRACKER_CAMERA,
        airsim.ImageType.Scene,
        DETECTION_RADIUS_CM,
        vehicle_name=TRACKER_VEHICLE,
    )
    client.simAddDetectionFilterMeshName(
        TRACKER_CAMERA,
        airsim.ImageType.Scene,
        TARGET_MESH_NAME,
        vehicle_name=TRACKER_VEHICLE,
    )


def detect(client: airsim.MultirotorClient, raw_w: int, raw_h: int):
    detections = client.simGetDetections(
        TRACKER_CAMERA,
        airsim.ImageType.Scene,
        vehicle_name=TRACKER_VEHICLE,
    )
    if not detections:
        return None

    best = max(
        detections,
        key=lambda d: (
            (d.box2D.max.x_val - d.box2D.min.x_val)
            * (d.box2D.max.y_val - d.box2D.min.y_val)
        ),
    )

    sx = IMG_W / raw_w
    sy = IMG_H / raw_h

    x1 = int(best.box2D.min.x_val * sx)
    y1 = int(best.box2D.min.y_val * sy)
    x2 = int(best.box2D.max.x_val * sx)
    y2 = int(best.box2D.max.y_val * sy)

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    area = float((x2 - x1) * (y2 - y1))
    return cx, cy, area, (x1, y1, x2, y2)
