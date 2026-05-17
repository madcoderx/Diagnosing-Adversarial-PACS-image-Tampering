from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

import numpy as np
import os

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
# Test Folders
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
# Loop Through Each Folder
# -----------------------------------

for folder_label, folder_path in test_folders.items():

    print("\n===================================")

    print(f"Testing Folder: {folder_label}")

    print("===================================\n")

    # Read files
    files = os.listdir(folder_path)

    # Test first 10 images
    for file in files[:10]:

        path = os.path.join(
            folder_path,
            file
        )

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

            # -----------------------------------
            # Predict
            # -----------------------------------

            prediction = model.predict(
                img_array,
                verbose=0
            )

            predicted_index = np.argmax(
                prediction
            )

            confidence = np.max(
                prediction
            )

            predicted_label = class_names[
                predicted_index
            ]

            # -----------------------------------
            # Print Result
            # -----------------------------------

            print(

                f"{file}"

                f"\nPredicted: "
                f"{predicted_label}"

                f"\nConfidence: "
                f"{confidence:.2f}"

                f"\n"

            )

        except Exception as e:

            print(
                "Error processing:",
                file
            )

            print(e)

print("\nTesting completed successfully")