import os
import time
import cv2
import torch
import numpy as np
from transformers import AutoImageProcessor, SiglipForImageClassification
from PIL import Image as PILImage

# Classification model (your transformer model)
# Use local model folder. Prefer workspace-relative folder name without leading './'
model_path = "sign_model"

print("Loading classifier model...")
try:
    processor = AutoImageProcessor.from_pretrained(model_path, local_files_only=True)
    model = SiglipForImageClassification.from_pretrained(model_path, local_files_only=True)
    model.eval()
except Exception as e:
    raise RuntimeError(
        f"Failed to load transformer model from local path '{model_path}': {e}\n"
        "If your model is in a different folder, set `model_path` to that folder, or ensure the files are present."
    )

# Try MediaPipe Tasks hand landmarker (requires a TFLite model file)
HAND_LANDMARKER_MODEL = os.environ.get("HAND_LANDMARKER_MODEL", None)
use_tasks = False
use_solutions = False
hand_detector = None
mp_draw_utils = None
hand_connections = None

if HAND_LANDMARKER_MODEL and os.path.exists(HAND_LANDMARKER_MODEL):
    try:
        from mediapipe.tasks.python.vision import (
            HandLandmarker,
            HandLandmarkerOptions,
            HandLandmarksConnections,
            drawing_utils,
            RunningMode,
        )
        from mediapipe.tasks.python.core.base_options import BaseOptions
        from mediapipe.tasks.python.vision.core import image as image_lib

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=HAND_LANDMARKER_MODEL),
            running_mode=RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_tracking_confidence=0.7,
        )

        hand_detector = HandLandmarker.create_from_options(options)
        mp_draw_utils = drawing_utils
        hand_connections = HandLandmarksConnections
        image_wrapper = image_lib.Image
        use_tasks = True
        print("Using MediaPipe Tasks HandLandmarker with model:", HAND_LANDMARKER_MODEL)
    except Exception as e:
        print("Failed to initialize MediaPipe Tasks HandLandmarker:", e)

if not use_tasks:
    # Try legacy mediapipe solutions API
    try:
        import mediapipe as mp
        if hasattr(mp, "solutions"):
            mp_hands = mp.solutions.hands
            mp_draw = mp.solutions.drawing_utils
            hands = mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.7,
            )
            mp_draw_utils = mp_draw
            hand_connections = mp_hands.HAND_CONNECTIONS
            use_solutions = True
            print("Using legacy MediaPipe solutions Hands API")
    except Exception:
        use_solutions = False

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

print("Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    hand_crop = None

    if use_tasks:
        # Convert to RGB and wrap as MediaPipe Image
        mp_img = image_wrapper(image_format=image_wrapper.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        timestamp_ms = int(time.time() * 1000)
        try:
            result = hand_detector.detect_for_video(mp_img, timestamp_ms)
            if result.hand_landmarks:
                landmarks = result.hand_landmarks[0]
                x_coords = [lm.x for lm in landmarks]
                y_coords = [lm.y for lm in landmarks]
                xmin = int(min(x_coords) * w) - 20
                xmax = int(max(x_coords) * w) + 20
                ymin = int(min(y_coords) * h) - 20
                ymax = int(max(y_coords) * h) + 20
                xmin = max(0, xmin)
                ymin = max(0, ymin)
                xmax = min(w, xmax)
                ymax = min(h, ymax)
                hand_crop = frame[ymin:ymax, xmin:xmax]
                mp_draw_utils.draw_landmarks(frame, landmarks, hand_connections.HAND_CONNECTIONS)
        except Exception:
            pass

    elif use_solutions:
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw_utils.draw_landmarks(frame, hand_landmarks, hand_connections)
                x_coords = [lm.x for lm in hand_landmarks.landmark]
                y_coords = [lm.y for lm in hand_landmarks.landmark]
                xmin = int(min(x_coords) * w) - 20
                xmax = int(max(x_coords) * w) + 20
                ymin = int(min(y_coords) * h) - 20
                ymax = int(max(y_coords) * h) + 20
                xmin = max(0, xmin)
                ymin = max(0, ymin)
                xmax = min(w, xmax)
                ymax = min(h, ymax)
                hand_crop = frame[ymin:ymax, xmin:xmax]

    # If no hand detected, use whole frame
    if hand_crop is None or hand_crop.size == 0:
        hand_crop = frame

    # Run classifier on cropped region
    pil_image = PILImage.fromarray(cv2.cvtColor(hand_crop, cv2.COLOR_BGR2RGB))
    inputs = processor(images=pil_image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_class = outputs.logits.argmax(-1).item()
    label = model.config.id2label[predicted_class]

    cv2.putText(
        frame,
        f"Prediction: {label}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    cv2.imshow("Sign Language Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()