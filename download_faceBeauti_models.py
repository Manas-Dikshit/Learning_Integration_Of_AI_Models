from transformers import AutoModelForImageClassification, AutoImageProcessor
import face_alignment
import torch

print("Downloading emotion model...")

emotion_model_name = "trpakov/vit-face-expression"
emotion_path = "./emotion_model"

model = AutoModelForImageClassification.from_pretrained(emotion_model_name)
processor = AutoImageProcessor.from_pretrained(emotion_model_name)

model.save_pretrained(emotion_path)
processor.save_pretrained(emotion_path)

print("Emotion model saved.")

print("Downloading landmark model...")

# this automatically downloads the landmark model
fa = face_alignment.FaceAlignment(
    face_alignment.LandmarksType.TWO_D,
    device="cpu"
)

print("Landmark model downloaded.")

print("All models ready.")