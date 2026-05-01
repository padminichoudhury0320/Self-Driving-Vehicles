import cv2
import numpy as np

def detect_lanes(image):
    height, width, _ = image.shape

    line_image = np.zeros_like(image)
    lane_area = np.zeros_like(image)

    left_bottom = (int(width * 0.32), height)
    right_bottom = (int(width * 0.62), height)

    left_top = (int(width * 0.42), int(height * 0.55))
    right_top = (int(width * 0.55), int(height * 0.55))

    cv2.line(line_image, left_bottom, left_top, (0, 180, 0), 5)
    cv2.line(line_image, right_bottom, right_top, (0, 180, 0), 5)

    pts = np.array([[left_bottom, left_top, right_top, right_bottom]], np.int32)
    cv2.fillPoly(lane_area, pts, (0, 180, 0))

    edges = np.zeros((height, width), dtype=np.uint8)
    roi = edges.copy()

    return line_image, edges, roi, lane_area