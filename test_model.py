from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# -----------------------------------
# LOAD TAMPER MODEL
# -----------------------------------

model = load_model("models/tamper_model.h5")

print("\nTamper Model Loaded Successfully")

# -----------------------------------
# CLASS NAMES (MUST MATCH TRAINING)
# -----------------------------------

class_names = [
    "ORIGINAL",
    "BLACKOUT_ATTACK",
    "BLUR_ATTACK",
    "GAUSSIAN_NOISE",
    "FAKE_PNEUMONIA"
]

# -----------------------------------
# TEST DATASETS
# -----------------------------------

test_folders = {
    "ORIGINAL": "training/original",
    "BLACKOUT_ATTACK": "training/tampered/blackout_attack",
    "BLUR_ATTACK": "training/tampered/blur_attack",
    "GAUSSIAN_NOISE": "training/tampered/gaussian_noise",
    "FAKE_PNEUMONIA": "training/tampered/fake_pneumonia"
}

# -----------------------------------
# TEST LOOP
# -----------------------------------

total = 0
correct = 0

for true_label, folder_path in test_folders.items():

    print("\n===============================")
    print(f"Testing: {true_label}")
    print("===============================\n")

    if not os.path.exists(folder_path):
        print("Folder missing:", folder_path)
        continue

    files = os.listdir(folder_path)

    for file in files[:10]:

        path = os.path.join(folder_path, file)

        try:
            img = image.load_img(path, target_size=(128, 128))
            img_array = image.img_to_array(img)
            img_array = img_array / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            prediction = model.predict(img_array, verbose=0)

            pred_index = np.argmax(prediction)
            confidence = np.max(prediction)

            predicted_label = class_names[pred_index]

            total += 1
            if predicted_label == true_label:
                correct += 1

            print(f"File: {file}")
            print(f"True Label: {true_label}")
            print(f"Predicted: {predicted_label}")
            print(f"Confidence: {confidence:.2f}")
            print("------------------------")

        except Exception as e:
            print("Error:", file)
            print(e)

# -----------------------------------
# FINAL ACCURACY
# -----------------------------------

if total > 0:
    accuracy = (correct / total) * 100
    print("\n===================================")
    print(f"Tamper Model Accuracy: {accuracy:.2f}%")
    print("===================================")

print("\nTesting completed successfully")