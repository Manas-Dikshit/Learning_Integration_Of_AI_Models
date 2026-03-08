# download_model.py

from transformers import AutoImageProcessor, AutoModelForImageClassification
import os

MODEL_NAME = "microsoft/resnet-50"
SAVE_DIR = "./models/resnet50"

os.makedirs(SAVE_DIR, exist_ok=True)

print("Downloading model...")

processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)

processor.save_pretrained(SAVE_DIR)
model.save_pretrained(SAVE_DIR)

print("Model downloaded and saved to:", SAVE_DIR)