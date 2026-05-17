import cv2
import matplotlib.pyplot as plt
import os
import sys

# -----------------------------------
# Select Attack Type
# -----------------------------------

attack_type = "fake_pneumonia"

# Available:
# gaussian_noise
# blur_attack
# blackout_attack
# fake_pneumonia

# -----------------------------------
# Original Image Folder
# -----------------------------------

original_folder = "training/original"

# -----------------------------------
# Tampered Folder
# -----------------------------------

tampered_folder = os.path.join(
    "training/tampered",
    attack_type
)

# -----------------------------------
# Get Image Files
# -----------------------------------

original_files = [

    f for f in os.listdir(original_folder)

    if f.lower().endswith(
        (".png", ".jpg", ".jpeg")
    )

]

tampered_files = [

    f for f in os.listdir(tampered_folder)

    if f.lower().endswith(
        (".png", ".jpg", ".jpeg")
    )

]

# -----------------------------------
# Check Files
# -----------------------------------

if len(original_files) == 0:

    print("\nNo original images found")

    sys.exit()

if len(tampered_files) == 0:

    print("\nNo tampered images found")

    sys.exit()

# -----------------------------------
# Select First Images
# -----------------------------------

original_path = os.path.join(

    original_folder,

    original_files[0]

)

tampered_path = os.path.join(

    tampered_folder,

    tampered_files[0]

)

print("\nOriginal Image:")
print(original_path)

print("\nTampered Image:")
print(tampered_path)

# -----------------------------------
# Read Images
# -----------------------------------

original = cv2.imread(
    original_path
)

tampered = cv2.imread(
    tampered_path
)

# -----------------------------------
# Validate Images
# -----------------------------------

if original is None:

    print(
        "\nError loading original image"
    )

    sys.exit()

if tampered is None:

    print(
        "\nError loading tampered image"
    )

    sys.exit()

# -----------------------------------
# Convert BGR -> RGB
# -----------------------------------

original = cv2.cvtColor(

    original,

    cv2.COLOR_BGR2RGB

)

tampered = cv2.cvtColor(

    tampered,

    cv2.COLOR_BGR2RGB

)

# -----------------------------------
# Difference Map
# -----------------------------------

difference = cv2.absdiff(
    original,
    tampered
)

# -----------------------------------
# Create Results Folder
# -----------------------------------

os.makedirs(
    "results",
    exist_ok=True
)

# -----------------------------------
# Save Difference Image
# -----------------------------------

difference_bgr = cv2.cvtColor(

    difference,

    cv2.COLOR_RGB2BGR

)

cv2.imwrite(

    "results/difference_map.png",

    difference_bgr

)

print(
    "\nDifference map saved successfully"
)

# -----------------------------------
# Visualization
# -----------------------------------

plt.figure(figsize=(15, 5))

# -----------------------------------
# Original Image
# -----------------------------------

plt.subplot(1, 3, 1)

plt.imshow(original)

plt.title("Original Image")

plt.axis("off")

# -----------------------------------
# Tampered Image
# -----------------------------------

plt.subplot(1, 3, 2)

plt.imshow(tampered)

plt.title(
    f"Tampered Image\n({attack_type})"
)

plt.axis("off")

# -----------------------------------
# Difference Map
# -----------------------------------

plt.subplot(1, 3, 3)

plt.imshow(difference)

plt.title("Difference Map")

plt.axis("off")

plt.tight_layout()

plt.show()

print(
    "\nVisualization completed successfully"
)