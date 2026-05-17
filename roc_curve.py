from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.preprocessing import label_binarize

import matplotlib.pyplot as plt

import numpy as np
import os

# -----------------------------------
# Create Results Folder
# -----------------------------------

os.makedirs(
    "results",
    exist_ok=True
)

# -----------------------------------
# Load Trained Model
# -----------------------------------

model = load_model(
    "models/pacs_model.h5"
)

print("\nModel loaded successfully")

# -----------------------------------
# Class Names
# -----------------------------------

class_names = [

    "BLACKOUT_ATTACK",

    "BLUR_ATTACK",

    "FAKE_PNEUMONIA",

    "GAUSSIAN_NOISE",

    "ORIGINAL"

]

# -----------------------------------
# Folder Paths
# -----------------------------------

test_folders = {

    "BLACKOUT_ATTACK":
    "training/tampered/blackout_attack",

    "BLUR_ATTACK":
    "training/tampered/blur_attack",

    "FAKE_PNEUMONIA":
    "training/tampered/fake_pneumonia",

    "GAUSSIAN_NOISE":
    "training/tampered/gaussian_noise",

    "ORIGINAL":
    "training/original"

}

# -----------------------------------
# Label Mapping
# -----------------------------------

label_map = {

    "BLACKOUT_ATTACK": 0,

    "BLUR_ATTACK": 1,

    "FAKE_PNEUMONIA": 2,

    "GAUSSIAN_NOISE": 3,

    "ORIGINAL": 4

}

# -----------------------------------
# Lists
# -----------------------------------

y_true = []

y_scores = []

# -----------------------------------
# Prediction Function
# -----------------------------------

def predict_image(path):

    try:

        # Load image
        img = image.load_img(
            path,
            target_size=(128, 128)
        )

        # Convert to array
        img_array = image.img_to_array(img)

        # Normalize
        img_array = img_array / 255.0

        # Add batch dimension
        img_array = np.expand_dims(
            img_array,
            axis=0
        )

        # Predict probabilities
        prediction = model.predict(
            img_array,
            verbose=0
        )

        return prediction[0]

    except Exception as e:

        print("Error:", path)

        print(e)

        return None

# -----------------------------------
# Process Images
# -----------------------------------

for folder_label, folder_path in test_folders.items():

    print(
        f"\nProcessing: {folder_label}"
    )

    files = [

        f for f in os.listdir(folder_path)

        if f.lower().endswith(
            (".png", ".jpg", ".jpeg")
        )

    ]

    for file in files:

        path = os.path.join(
            folder_path,
            file
        )

        prediction = predict_image(path)

        if prediction is not None:

            y_true.append(
                label_map[folder_label]
            )

            y_scores.append(prediction)

# -----------------------------------
# Convert to NumPy Arrays
# -----------------------------------

y_true = np.array(y_true)

y_scores = np.array(y_scores)

# -----------------------------------
# Binarize Labels
# -----------------------------------

y_true_bin = label_binarize(
    y_true,
    classes=[0, 1, 2, 3, 4]
)

# -----------------------------------
# Plot ROC Curves
# -----------------------------------

plt.figure(figsize=(8, 6))

for i in range(len(class_names)):

    fpr, tpr, thresholds = roc_curve(
        y_true_bin[:, i],
        y_scores[:, i]
    )

    roc_auc = auc(fpr, tpr)

    plt.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"{class_names[i]} "
              f"(AUC = {roc_auc:.2f})"
    )

# -----------------------------------
# Random Classifier Line
# -----------------------------------

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

# -----------------------------------
# Labels and Title
# -----------------------------------

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "Multiclass ROC Curve"
)

plt.legend(
    loc="lower right"
)

plt.grid(True)

# -----------------------------------
# Save Figure
# -----------------------------------

save_path = os.path.join(
    "results",
    "roc_curve.png"
)

plt.savefig(save_path)

print(
    f"\nROC curve saved at:\n{save_path}"
)

# -----------------------------------
# Show Plot
# -----------------------------------

plt.show()

print("\nROC curve generated successfully")