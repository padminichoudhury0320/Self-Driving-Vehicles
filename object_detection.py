import cv2
import numpy as np

print("Loading YOLO model... (this may take 30+ seconds)")
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
print("YOLO model loaded!")

classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

def detect_objects(image, lane_area):
    height, width, _ = image.shape

    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []

    # 🔥 Track closest object INSIDE lane
    closest_cy = -1
    closest_box = None

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    danger = False
    slow = False

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = classes[class_ids[i]]

            if label in ["car", "bus", "truck", "person"]:

                # 🔥 Ignore very small (far) objects
                if w < width * 0.03:
                    continue

                # Bottom center (realistic)
                cx = x + w // 2
                cy = y + int(h * 0.9)

                cx = min(cx, width - 1)
                cy = min(cy, height - 1)

                # 🔥 Center driving region
                lane_left = int(width * 0.42)
                lane_right = int(width * 0.72)

                inside_lane = lane_left < cx < lane_right

                # Default color
                color = (0, 255, 0)

                # 🔥 Track closest ONLY inside lane
                if inside_lane:
                    if cy > closest_cy:
                        closest_cy = cy
                        closest_box = (x, y, w, h)

                # Draw all objects
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

    # 🔥 Decision based on closest object ONLY
    if closest_box is not None:
        x, y, w, h = closest_box

        if closest_cy > height * 0.65:
            danger = True
            color = (0, 0, 255)

        elif closest_cy > height * 0.45:
            slow = True
            color = (0, 140, 255)

        else:
            color = (0, 255, 0)

        # Highlight closest object
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 4)

    return image, danger, slow