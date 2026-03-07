from transformers import AutoImageProcessor, SiglipForImageClassification

model_name = "prithivMLmods/Alphabet-Sign-Language-Detection"
save_path = "./sign_model"

print("Downloading model...")

model = SiglipForImageClassification.from_pretrained(model_name)
processor = AutoImageProcessor.from_pretrained(model_name)

model.save_pretrained(save_path)
processor.save_pretrained(save_path)

print("Model saved to", save_path)