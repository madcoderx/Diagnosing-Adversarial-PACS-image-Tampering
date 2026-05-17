import os
import cv2
import numpy as np
import pandas as pd

# -----------------------------------
# INPUT PACS FOLDERS
# -----------------------------------

input_folders = [
    "pacs_storage/NORMAL",
    "pacs_storage/PNEUMONIA"
]

# -----------------------------------
# OUTPUT FOLDERS (TAMPER MODEL DATASET)
# -----------------------------------

original_folder = "training/original"

tampered_folder = "training/tampered"

noise_folder = os.path.join(tampered_folder, "gaussian_noise")
blur_folder = os.path.join(tampered_folder, "blur_attack")
blackout_folder = os.path.join(tampered_folder, "blackout_attack")
fake_pneumonia_folder = os.path.join(tampered_folder, "fake_pneumonia")

# -----------------------------------
# CREATE FOLDERS
# -----------------------------------

os.makedirs(original_folder, exist_ok=True)
os.makedirs(noise_folder, exist_ok=True)
os.makedirs(blur_folder, exist_ok=True)
os.makedirs(blackout_folder, exist_ok=True)
os.makedirs(fake_pneumonia_folder, exist_ok=True)

# -----------------------------------
# METADATA
# -----------------------------------

attack_data = []

# -----------------------------------
# PROCESS IMAGES
# -----------------------------------

for input_folder in input_folders:

    print(f"\nReading Folder: {input_folder}")

    if not os.path.exists(input_folder):
        print("Folder not found")
        continue

    files = [
        f for f in os.listdir(input_folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    print(f"Images Found: {len(files)}")

    for file in files:

        path = os.path.join(input_folder, file)
        image = cv2.imread(path)

        if image is None:
            print(f"Skipping: {file}")
            continue

        image = cv2.resize(image, (128, 128))

        # -----------------------------------
        # ORIGINAL IMAGE
        # -----------------------------------

        cv2.imwrite(
            os.path.join(original_folder, file),
            image
        )

        attack_data.append({
            "Image_Name": file,
            "Attack_Type": "ORIGINAL"
        })

        h, w, _ = image.shape

        # ===================================
        # 1. GAUSSIAN NOISE
        # ===================================

        noise = np.random.normal(0, 25, image.shape)
        noisy_image = image.astype(np.float32) + noise
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

        noise_name = "noise_" + file

        cv2.imwrite(
            os.path.join(noise_folder, noise_name),
            noisy_image
        )

        attack_data.append({
            "Image_Name": noise_name,
            "Attack_Type": "GAUSSIAN_NOISE"
        })

        # ===================================
        # 2. BLUR ATTACK
        # ===================================

        blurred_image = cv2.GaussianBlur(image, (15, 15), 0)

        blur_name = "blur_" + file

        cv2.imwrite(
            os.path.join(blur_folder, blur_name),
            blurred_image
        )

        attack_data.append({
            "Image_Name": blur_name,
            "Attack_Type": "BLUR_ATTACK"
        })

        # ===================================
        # 3. BLACKOUT ATTACK
        # ===================================

        blackout_image = image.copy()
        blackout_image[h//3:h//2, w//3:w//2] = 0

        blackout_name = "blackout_" + file

        cv2.imwrite(
            os.path.join(blackout_folder, blackout_name),
            blackout_image
        )

        attack_data.append({
            "Image_Name": blackout_name,
            "Attack_Type": "BLACKOUT_ATTACK"
        })

        # ===================================
        # 4. FAKE PNEUMONIA ATTACK (KEEP THIS)
        # ===================================

        fake_img = image.copy()
        overlay = fake_img.copy()

        center_x = np.random.randint(w//3, 2*w//3)
        center_y = np.random.randint(h//3, 2*h//3)

        axis_x = np.random.randint(20, 40)
        axis_y = np.random.randint(30, 50)

        cv2.ellipse(
            overlay,
            (center_x, center_y),
            (axis_x, axis_y),
            angle=np.random.randint(0, 180),
            startAngle=0,
            endAngle=360,
            color=(255, 255, 255),
            thickness=-1
        )

        overlay = cv2.GaussianBlur(overlay, (41, 41), 0)

        fake_img = cv2.addWeighted(overlay, 0.25, fake_img, 0.75, 0)

        fake_name = "fake_pneumonia_" + file

        cv2.imwrite(
            os.path.join(fake_pneumonia_folder, fake_name),
            fake_img
        )

        attack_data.append({
            "Image_Name": fake_name,
            "Attack_Type": "FAKE_PNEUMONIA"
        })

# -----------------------------------
# SAVE METADATA
# -----------------------------------

df = pd.DataFrame(attack_data)

df.to_csv(
    "training/attack_metadata.csv",
    index=False
)

# -----------------------------------
# VERIFICATION
# -----------------------------------

print("\nVerification")

print("Original:", len(os.listdir(original_folder)))
print("Gaussian Noise:", len(os.listdir(noise_folder)))
print("Blur Attack:", len(os.listdir(blur_folder)))
print("Blackout Attack:", len(os.listdir(blackout_folder)))
print("Fake Pneumonia:", len(os.listdir(fake_pneumonia_folder)))

print("\nAll tamper dataset generated successfully (FINAL VERSION)")