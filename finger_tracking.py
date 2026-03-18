import cv2
import numpy as np
from ultralytics import YOLO

# Load model
model = YOLO("./fingertracking/yolov8n-pose")

cap = cv2.VideoCapture(0)

canvas = None
prev_x, prev_y = 0, 0

draw_mode = True
color = (255, 0, 0)  # default blue

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    if canvas is None:
        canvas = np.zeros_like(frame)

    h, w, _ = frame.shape

    # Draw UI buttons
    cv2.rectangle(frame, (0, 0), (100, 60), (255, 0, 0), -1)   # Blue
    cv2.rectangle(frame, (100, 0), (200, 60), (0, 255, 0), -1) # Green
    cv2.rectangle(frame, (200, 0), (300, 60), (0, 0, 255), -1) # Red
    cv2.rectangle(frame, (300, 0), (400, 60), (0, 0, 0), -1)   # Clear

    results = model(frame)

    if results[0].keypoints is not None:
        keypoints = results[0].keypoints.xy.cpu().numpy()

        for person in keypoints:
            x, y = int(person[10][0]), int(person[10][1])  # wrist

            cv2.circle(frame, (x, y), 8, (0, 255, 255), -1)

            # 🧠 Gesture logic (simulated)
            # If hand is at top → selection mode
            if y < 60:
                prev_x, prev_y = 0, 0

                if x < 100:
                    color = (255, 0, 0)
                elif x < 200:
                    color = (0, 255, 0)
                elif x < 300:
                    color = (0, 0, 255)
                elif x < 400:
                    canvas = np.zeros_like(frame)

                draw_mode = False

            else:
                draw_mode = True

                if draw_mode:
                    if prev_x == 0 and prev_y == 0:
                        prev_x, prev_y = x, y

                    cv2.line(canvas, (prev_x, prev_y), (x, y), color, 5)

                    prev_x, prev_y = x, y

    else:
        prev_x, prev_y = 0, 0

    # Merge canvas
    gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_canvas, 20, 255, cv2.THRESH_BINARY_INV)
    thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    frame = cv2.bitwise_and(frame, thresh)
    frame = cv2.bitwise_or(frame, canvas)

    cv2.imshow("Air Drawing AI", frame)

    key = cv2.waitKey(1)

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()