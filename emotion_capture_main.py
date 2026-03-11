import cv2
import torch
from PIL import Image
from transformers import AutoModelForImageClassification, AutoImageProcessor

# load local model
model_path = "./emotion_model"

model = AutoModelForImageClassification.from_pretrained(model_path)
processor = AutoImageProcessor.from_pretrained(model_path)

model.eval()

# face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# start webcam
cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:

        face = frame[y:y+h, x:x+w]

        image = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))

        inputs = processor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits
        predicted_class_id = logits.argmax(-1).item()
        label = model.config.id2label[predicted_class_id]

        # draw face box
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

        # show emotion
        cv2.putText(
            frame,
            f"Mood: {label}",
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0,255,0),
            2
        )

    cv2.imshow("Emotion Detector", frame)

    # press q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()