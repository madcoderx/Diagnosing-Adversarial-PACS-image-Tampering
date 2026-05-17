from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

import matplotlib.pyplot as plt
import numpy as np
import os

# -----------------------------------
# RESULTS FOLDER
# -----------------------------------

os.makedirs("results", exist_ok=True)

# -----------------------------------
# LOAD TAMPER MODEL ONLY
# -----------------------------------

model = load_model("models/tamper_model.h5")

print("\nTamper Model Loaded Successfully")

# -----------------------------------
# CLASS NAMES (TAMPER MODEL ONLY)
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
y_scores = []

# -----------------------------------
# IMAGE PREDICTION
# -----------------------------------

def predict_image(path):

    try:
        img = image.load_img(path, target_size=(128, 128))
        img_array = image.img_to_array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array, verbose=0)

        return prediction[0]

    except Exception as e:
        print("Error:", path)
        print(e)
        return None

# -----------------------------------
# PROCESS DATA
# -----------------------------------

for label, folder_path in test_folders.items():

    print(f"\nProcessing: {label}")

    if not os.path.exists(folder_path):
        continue

    files = os.listdir(folder_path)

    for file in files:

        path = os.path.join(folder_path, file)

        pred = predict_image(path)

        if pred is not None:
            y_true.append(class_names.index(label))
            y_scores.append(pred)

# -----------------------------------
# CONVERT
# -----------------------------------

y_true = np.array(y_true)
y_scores = np.array(y_scores)

y_true_bin = label_binarize(y_true, classes=list(range(len(class_names))))

# -----------------------------------
# PLOT ROC CURVES
# -----------------------------------

plt.figure(figsize=(8, 6))

for i in range(len(class_names)):

    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC = {roc_auc:.2f})")

# -----------------------------------
# RANDOM BASELINE
# -----------------------------------

plt.plot([0, 1], [0, 1], linestyle="--")

# -----------------------------------
# LABELS
# -----------------------------------

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Tamper Detection ROC Curve")
plt.legend(loc="lower right")
plt.grid(True)

# -----------------------------------
# SAVE
# -----------------------------------

save_path = os.path.join("results", "roc_curve_tamper.png")
plt.savefig(save_path)

plt.show()

print("\nROC curve saved at:", save_path)