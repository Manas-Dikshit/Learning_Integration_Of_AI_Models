import cv2
import numpy as np
import torch

# Load YOLOv8 pose model (local)
model_path = "./fingertracking/yolov8n-pose"

# Ultralytics loader (works with HF weights)
from ultralytics import YOLO
model = YOLO(model_path)

cap = cv2.VideoCapture(0)

canvas = None
prev_x, prev_y = 0, 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    if canvas is None:
        canvas = np.zeros_like(frame)

    # Run inference
    results = model(frame)

    if results[0].keypoints is not None:
        keypoints = results[0].keypoints.xy.cpu().numpy()

        for person in keypoints:
            # Use wrist (index 10 or 9 depending on side)
            x, y = int(person[10][0]), int(person[10][1])

            cv2.circle(frame, (x, y), 8, (0, 255, 0), -1)

            if prev_x == 0 and prev_y == 0:
                prev_x, prev_y = x, y

            cv2.line(canvas, (prev_x, prev_y), (x, y), (255, 0, 0), 5)

            prev_x, prev_y = x, y
    else:
        prev_x, prev_y = 0, 0

    # Merge canvas
    gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_canvas, 20, 255, cv2.THRESH_BINARY_INV)
    thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    frame = cv2.bitwise_and(frame, thresh)
    frame = cv2.bitwise_or(frame, canvas)

    cv2.imshow("HF Finger Drawing", frame)

    key = cv2.waitKey(1)

    if key == 27:  # ESC
        break
    elif key == ord('c'):
        canvas = np.zeros_like(frame)

cap.release()
cv2.destroyAllWindows()