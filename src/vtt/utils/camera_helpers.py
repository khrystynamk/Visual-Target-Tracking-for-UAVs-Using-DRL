import airsim
import numpy as np
import cv2

from vtt.constants import (
    IMAGE_SIZE,
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


def capture_depth_raw(client) -> np.ndarray:
    """
    Capture raw depth in meters from AirSim. Returns (H, W) float32.
    No normalization — DeFM's preprocess_depth_image handles that.
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

    if responses[0].width == 0 or responses[0].height == 0:
        return np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)

    depth = airsim.list_to_2d_float_array(
        responses[0].image_data_float,
        responses[0].width,
        responses[0].height,
    )
    return depth.astype(np.float32)  # raw meters, (H, W)


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


def get_relative_bbox(
    client: airsim.MultirotorClient,
    raw_w: int,
    raw_h: int,
    image_size: int = IMAGE_SIZE,
) -> np.ndarray:
    det = detect(client, raw_w, raw_h)
    if det is None:
        return np.zeros(4, dtype=np.float32), False

    cx, cy, _, (x1, y1, x2, y2) = det
    rel_cx = cx / image_size
    rel_cy = cy / image_size
    rel_w = (x2 - x1) / image_size
    rel_h = (y2 - y1) / image_size

    return np.array([rel_cx, rel_cy, rel_w, rel_h], dtype=np.float32).clip(
        0.0, 1.0
    ), True


def render_depth_with_bbox(
    depth_frame: np.ndarray,
    bbox: np.ndarray,
    image_size: int,
    reward: float = 0.0,
    dist: float = 0.0,
    frames_lost: int = 0,
):
    if depth_frame.ndim == 3 and depth_frame.shape[0] == 3:
        img = (depth_frame.transpose(1, 2, 0) * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        if depth_frame.ndim == 1:
            return
        d = depth_frame.squeeze()  # ensure (H, W)
        d = np.log1p(d) / np.log1p(100)  # log scale to [0, 1]
        d = (d * 255).clip(0, 255).astype(np.uint8)
        img = cv2.applyColorMap(d, cv2.COLORMAP_INFERNO)

    cx, cy, w, h = bbox
    if w > 0 and h > 0:
        x1 = int((cx - w / 2) * image_size)
        y1 = int((cy - h / 2) * image_size)
        x2 = int((cx + w / 2) * image_size)
        y2 = int((cy + h / 2) * image_size)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(
            img, (int(cx * image_size), int(cy * image_size)), 3, (0, 0, 255), -1
        )

    cv2.putText(
        img,
        f"r={reward:.2f} d={dist:.1f}m lost={frames_lost}",
        (3, 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.3,
        (0, 255, 0),
        1,
    )

    cv2.imshow("Tracking (depth + bbox)", img)
    cv2.waitKey(1)
