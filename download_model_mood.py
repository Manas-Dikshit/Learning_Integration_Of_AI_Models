from transformers import AutoModelForImageClassification, AutoImageProcessor

model_name = "trpakov/vit-face-expression"
save_path = "./emotion_model"

print("Downloading model...")

model = AutoModelForImageClassification.from_pretrained(model_name)
processor = AutoImageProcessor.from_pretrained(model_name)

model.save_pretrained(save_path)
processor.save_pretrained(save_path)

print("Model downloaded and saved to:", save_path)