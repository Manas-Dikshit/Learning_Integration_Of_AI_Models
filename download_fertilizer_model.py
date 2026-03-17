from huggingface_hub import hf_hub_download, list_repo_files
import joblib
import os

MODEL_REPO = "Briankabiru/FertiliserAdvisor"
SAVE_DIR = "./fertilizer"


def find_model_file():
    files = list_repo_files(MODEL_REPO)
    print("Files in repo:", files)

    for f in files:
        if f.endswith(".pkl") or f.endswith(".joblib"):
            return f

    raise Exception("No model file found!")


def download_model():
    os.makedirs(SAVE_DIR, exist_ok=True)

    model_file = find_model_file()
    print("Downloading:", model_file)

    model_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=model_file,
        local_dir=SAVE_DIR,          # ✅ saves in ./fertilizer
        local_dir_use_symlinks=False
    )

    print("Saved at:", model_path)
    return model_path


if __name__ == "__main__":
    download_model()