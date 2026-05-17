from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

from sklearn.metrics import confusion_matrix, classification_report

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

# -----------------------------------
# CREATE RESULTS FOLDER
# -----------------------------------

os.makedirs("results", exist_ok=True)

# -----------------------------------
# LOAD TAMPER MODEL
# -----------------------------------

model = load_model("models/tamper_model.h5")

print("\nTamper Model Loaded Successfully")

# -----------------------------------
# CLASS NAMES (MUST MATCH TRAINING ORDER)
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
# STORAGE
# -----------------------------------

y_true = []
y_pred = []

# -----------------------------------
# IMAGE PREDICTION FUNCTION
# -----------------------------------

def predict_image(path):

    try:
        img = image.load_img(path, target_size=(128, 128))
        img_array = image.img_to_array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array, verbose=0)
        return np.argmax(prediction)

    except Exception as e:
        print("Error:", path)
        print(e)
        return None

# -----------------------------------
# PROCESS DATASETS
# -----------------------------------

for true_label, folder_path in test_folders.items():

    print(f"\nProcessing: {true_label}")

    if not os.path.exists(folder_path):
        print("Missing folder:", folder_path)
        continue

    files = [
        f for f in os.listdir(folder_path)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    for file in files:

        path = os.path.join(folder_path, file)

        pred = predict_image(path)

        if pred is not None:
            y_true.append(class_names.index(true_label))
            y_pred.append(pred)

# -----------------------------------
# CONFUSION MATRIX
# -----------------------------------

cm = confusion_matrix(y_true, y_pred)

print("\nConfusion Matrix:\n")
print(cm)

# -----------------------------------
# CLASSIFICATION REPORT
# -----------------------------------

print("\nClassification Report:\n")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names
    )
)

# -----------------------------------
# PLOT HEATMAP
# -----------------------------------

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.title("Tamper Detection Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

# -----------------------------------
# SAVE FIGURE
# -----------------------------------

save_path = os.path.join("results", "confusion_matrix.png")
plt.savefig(save_path)

print(f"\nSaved at: {save_path}")

plt.show()

print("\nConfusion matrix generated successfully")