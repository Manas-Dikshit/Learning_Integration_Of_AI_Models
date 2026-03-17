import joblib
import os

MODEL_PATH = "./fertilizer"


def load_model():
    # find model file inside folder
    for file in os.listdir(MODEL_PATH):
        if file.endswith(".pkl") or file.endswith(".joblib"):
            model_file = os.path.join(MODEL_PATH, file)
            print("Loading:", model_file)
            return joblib.load(model_file)

    raise Exception("No model found in ./fertilizer/")


def get_user_input():
    print("\nEnter details:\n")

    nitrogen = float(input("Nitrogen: "))
    phosphorus = float(input("Phosphorus: "))
    potassium = float(input("Potassium: "))
    temperature = float(input("Temperature: "))
    humidity = float(input("Humidity: "))
    moisture = float(input("Moisture: "))
    ph = float(input("Soil pH: "))   # ✅ NEW FEATURE

    soil = input("Soil (Sandy/Loamy/Clay): ")
    crop = input("Crop (Rice/Wheat/Maize): ")

    soil_map = {"Sandy": 0, "Loamy": 1, "Clay": 2}
    crop_map = {"Rice": 0, "Wheat": 1, "Maize": 2}

    return [[
        temperature,
        humidity,
        moisture,
        ph,   # ✅ added
        soil_map.get(soil, 1),
        crop_map.get(crop, 0),
        nitrogen,
        phosphorus,
        potassium
    ]]


def predict():
    model = load_model()
    data = get_user_input()

    result = model.predict(data)

    print("\n🌾 Fertilizer Recommendation:")
    print(result)


if __name__ == "__main__":
    predict()