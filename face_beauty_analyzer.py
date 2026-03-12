import cv2
import torch
import numpy as np
from PIL import Image
from transformers import AutoModelForImageClassification, AutoImageProcessor
import face_alignment

# load emotion model
model_path = "./emotion_model"

emotion_model = AutoModelForImageClassification.from_pretrained(model_path)
processor = AutoImageProcessor.from_pretrained(model_path)

emotion_model.eval()

# load landmark detector
fa = face_alignment.FaceAlignment(
    face_alignment.LandmarksType.TWO_D,
    device="cpu"
)

# face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:

        face = frame[y:y+h, x:x+w]

        # emotion detection
        image = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
        inputs = processor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = emotion_model(**inputs)

        logits = outputs.logits
        predicted = logits.argmax(-1).item()
        mood = emotion_model.config.id2label[predicted]

        # landmark detection
        landmarks = fa.get_landmarks(frame)

        suggestion = "Looking good"

        if mood == "sad":
            suggestion = "Try smiling 🙂"

        if mood == "angry":
            suggestion = "Relax facial muscles"

        if landmarks is not None:
            suggestion = suggestion + " | Keep good posture"

        # draw face box
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

        cv2.putText(
            frame,
            f"Mood: {mood}",
            (x, y-30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,255,0),
            2
        )

        cv2.putText(
            frame,
            f"Tip: {suggestion}",
            (x, y-5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255,255,0),
            2
        )

    cv2.imshow("AI Face Beauty Analyzer", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()