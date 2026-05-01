import cv2
import numpy as np
from lane_detection import detect_lanes
from object_detection import detect_objects

image = cv2.imread("road5.jpg")

line_image, edges, roi, lane_area = detect_lanes(image.copy())
object_image, danger, slow = detect_objects(image.copy(), lane_area)

output = image.copy()
 
# =====================================================
# LANE SHADING (NO OVER-BRIGHT)
# =====================================================

lane_mask = lane_area[:, :, 1] > 0
height = output.shape[0]

for y in range(height):
    row_mask = lane_mask[y]
    alpha = (y / height) * 0.6

    output[y][row_mask] = (
        (1 - alpha) * output[y][row_mask] +
        alpha * np.array([0, 255, 0])
    ).astype(np.uint8)

# Draw lanes
output = cv2.addWeighted(output, 1, line_image, 1, 0)

# Draw objects
mask_obj = object_image != image
output[mask_obj] = object_image[mask_obj]

# =====================================================
# 🔥 STABLE DECISION SYSTEM
# =====================================================

if danger:
    text = "STOP"
    color = (0, 0, 255)
elif slow:
    text = "SLOW"
    color = (0, 140, 255)
else:
    text = "GO"
    color = (0, 255, 0)

cv2.putText(output, text, (40, 70),
            cv2.FONT_HERSHEY_SIMPLEX, 1.6, color, 4)

# =====================================================
# DISPLAY
# =====================================================

cv2.imshow("Final Output", output)
cv2.waitKey(0)
cv2.destroyAllWindows()