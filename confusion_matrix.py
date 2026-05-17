from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import seaborn as sns
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
# Load Model
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

y_pred = []

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

        # Predict
        prediction = model.predict(
            img_array,
            verbose=0
        )

        predicted_class = np.argmax(
            prediction
        )

        return predicted_class

    except Exception as e:

        print("Error:", path)

        print(e)

        return None

# -----------------------------------
# Process Each Folder
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

        pred = predict_image(path)

        if pred is not None:

            y_true.append(
                label_map[folder_label]
            )

            y_pred.append(pred)

# -----------------------------------
# Confusion Matrix
# -----------------------------------

cm = confusion_matrix(
    y_true,
    y_pred
)

print("\nConfusion Matrix:\n")

print(cm)

# -----------------------------------
# Classification Report
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
# Plot Heatmap
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

plt.title(
    "Multiclass Confusion Matrix"
)

plt.xlabel("Predicted Label")

plt.ylabel("True Label")

# -----------------------------------
# Save Figure
# -----------------------------------

save_path = os.path.join(
    "results",
    "confusion_matrix.png"
)

plt.savefig(save_path)

print(
    f"\nConfusion matrix saved at:\n{save_path}"
)

# -----------------------------------
# Show Plot
# -----------------------------------

plt.show()

print("\nConfusion matrix generated successfully")