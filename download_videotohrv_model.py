from huggingface_hub import snapshot_download
import os

model_id = "PulsePals/VideoToHRV"
save_dir = "./videotohrv_model"

os.makedirs(save_dir, exist_ok=True)

print("Downloading VideoToHRV repository...")

snapshot_download(
    repo_id=model_id,
    local_dir=save_dir,
    local_dir_use_symlinks=False
)

print("Download completed!")
print("Saved in:", save_dir)