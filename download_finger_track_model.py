from huggingface_hub import snapshot_download
import os

save_dir = "./fingertracking/yolov8n-pose"

os.makedirs(save_dir, exist_ok=True)

print("[INFO] Downloading model from Hugging Face...")

snapshot_download(
    repo_id="Xenova/yolov8n-pose",
    local_dir=save_dir,
    local_dir_use_symlinks=False
)

print(f"[SUCCESS] Model saved to: {save_dir}")