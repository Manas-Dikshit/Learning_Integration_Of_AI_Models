import os
import urllib.request

# folder to save model
save_dir = "./eye_state"
os.makedirs(save_dir, exist_ok=True)

model_url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"

model_path = os.path.join(save_dir, "face_landmarker.task")

print("Downloading face landmark model...")

urllib.request.urlretrieve(model_url, model_path)

print("Download complete!")
print("Model saved at:", model_path)