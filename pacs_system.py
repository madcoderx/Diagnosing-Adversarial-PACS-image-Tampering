import os
import cv2
import pandas as pd
from datetime import datetime

# -----------------------------------
# PACS Main Folder
# -----------------------------------

pacs_folder = "pacs_storage"

# -----------------------------------
# PACS Disease Folders
# -----------------------------------

normal_pacs_folder = os.path.join(
    pacs_folder,
    "NORMAL"
)

pneumonia_pacs_folder = os.path.join(
    pacs_folder,
    "PNEUMONIA"
)

# -----------------------------------
# TAMPERED PACS FOLDERS (ADD THIS)
# -----------------------------------

tampered_pacs_folder = os.path.join(pacs_folder, "TAMPERED")

blackout_folder = os.path.join(tampered_pacs_folder, "BLACKOUT_ATTACK")

blur_folder = os.path.join(tampered_pacs_folder, "BLUR_ATTACK")

noise_folder = os.path.join(tampered_pacs_folder, "GAUSSIAN_NOISE")

fake_pneumonia_folder = os.path.join(tampered_pacs_folder, "FAKE_PNEUMONIA")

# -----------------------------------
# Create PACS Folders
# -----------------------------------

os.makedirs(normal_pacs_folder, exist_ok=True)
os.makedirs(pneumonia_pacs_folder, exist_ok=True)

# create tampered folders
os.makedirs(tampered_pacs_folder, exist_ok=True)
os.makedirs(blackout_folder, exist_ok=True)
os.makedirs(blur_folder, exist_ok=True)
os.makedirs(noise_folder, exist_ok=True)
os.makedirs(fake_pneumonia_folder, exist_ok=True)

# -----------------------------------
# Dataset Folders
# -----------------------------------

normal_dataset_folder = (
    "dataset/NORMAL"
)

pneumonia_dataset_folder = (
    "dataset/PNEUMONIA"
)

# -----------------------------------
# Metadata List
# -----------------------------------

data = []

# -----------------------------------
# Patient Counter
# -----------------------------------

patient_id = 1

# -----------------------------------
# Store Images Function
# -----------------------------------

def store_images(

    dataset_folder,

    disease_name,

    pacs_save_folder

):

    global patient_id

    # Read image files
    files = [

        f for f in os.listdir(dataset_folder)

        if f.lower().endswith(
            (
                ".png",
                ".jpg",
                ".jpeg"
            )
        )

    ]

    print(
        f"\nProcessing {disease_name} images..."
    )

    # Process first 100 images
    for file in files[:100]:

        image_path = os.path.join(
            dataset_folder,
            file
        )

        # Read image
        image = cv2.imread(
            image_path
        )

        # Skip broken image
        if image is None:

            print(
                f"Skipping corrupted image: {file}"
            )

            continue

        try:

            # Resize image
            image = cv2.resize(
                image,
                (256, 256)
            )

            # Create PACS filename
            new_name = (

                f"{disease_name}_"

                f"{patient_id}.png"

            )

            # Save path
            save_path = os.path.join(
                pacs_save_folder,
                new_name
            )

            # Save image
            cv2.imwrite(
                save_path,
                image
            )

            # -----------------------------------
            # Metadata
            # -----------------------------------

            data.append({

                "Patient_ID":
                patient_id,

                "Image_Name":
                new_name,

                "Diagnosis":
                disease_name,

                "PACS_Path":
                save_path,

                "Image_Size":
                "256x256",

                "Upload_Time":
                datetime.now()

            })

            print(
                f"Stored: {new_name}"
            )

            # Increment patient ID
            patient_id += 1

        except Exception as e:

            print(
                f"\nError processing: {file}"
            )

            print(e)

# -----------------------------------
# Store NORMAL Images
# -----------------------------------

store_images(

    normal_dataset_folder,

    "NORMAL",

    normal_pacs_folder

)

# -----------------------------------
# Store PNEUMONIA Images
# -----------------------------------

store_images(

    pneumonia_dataset_folder,

    "PNEUMONIA",

    pneumonia_pacs_folder

)

# -----------------------------------
# Save Metadata CSV
# -----------------------------------

metadata_path = os.path.join(
    pacs_folder,
    "metadata.csv"
)

df = pd.DataFrame(data)

df.to_csv(
    metadata_path,
    index=False
)

# -----------------------------------
# Final Output
# -----------------------------------

print(
    "\nPACS storage created successfully"
)

print(
    f"\nMetadata saved at:\n{metadata_path}"
)

print(
    "\nFolder Structure:"
)

print(
    """
pacs_storage/

    NORMAL/

    PNEUMONIA/

    metadata.csv
"""
)