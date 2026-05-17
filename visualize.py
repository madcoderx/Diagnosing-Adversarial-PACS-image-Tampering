import cv2
import matplotlib.pyplot as plt
import os
import sys

# -----------------------------------
# CONFIG
# -----------------------------------

IMG_SIZE = 128

attack_type = "fake_pneumonia"

# -----------------------------------
# FOLDERS
# -----------------------------------

original_folder = "training/original"
tampered_folder = os.path.join("training/tampered", attack_type)

# -----------------------------------
# GET FILES
# -----------------------------------

original_files = [
    f for f in os.listdir(original_folder)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
]

tampered_files = [
    f for f in os.listdir(tampered_folder)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
]

if not original_files or not tampered_files:
    print("No images found")
    sys.exit()

# -----------------------------------
# LOAD FIRST PAIR
# -----------------------------------

orig_path = os.path.join(original_folder, original_files[0])
tamp_path = os.path.join(tampered_folder, tampered_files[0])

print("\nOriginal:", orig_path)
print("Tampered:", tamp_path)

orig = cv2.imread(orig_path)
tamp = cv2.imread(tamp_path)

if orig is None or tamp is None:
    print("Image load error")
    sys.exit()

# -----------------------------------
# RESIZE (IMPORTANT FOR FAIR COMPARISON)
# -----------------------------------

orig = cv2.resize(orig, (IMG_SIZE, IMG_SIZE))
tamp = cv2.resize(tamp, (IMG_SIZE, IMG_SIZE))

# -----------------------------------
# CONVERT RGB
# -----------------------------------

orig_rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
tamp_rgb = cv2.cvtColor(tamp, cv2.COLOR_BGR2RGB)

# -----------------------------------
# DIFFERENCE MAP (ENHANCED)
# -----------------------------------

diff = cv2.absdiff(orig, tamp)
diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

# Normalize for visualization
diff_norm = cv2.normalize(diff_gray, None, 0, 255, cv2.NORM_MINMAX)

# -----------------------------------
# SAVE OUTPUT
# -----------------------------------

os.makedirs("results", exist_ok=True)

cv2.imwrite("results/difference_map.png", diff_norm)

print("\nSaved difference map")

# -----------------------------------
# VISUALIZATION
# -----------------------------------

plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.imshow(orig_rgb)
plt.title("Original PACS Image")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(tamp_rgb)
plt.title(f"Tampered Image ({attack_type})")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(diff_norm, cmap="gray")
plt.title("Difference Heatmap")
plt.axis("off")

plt.tight_layout()
plt.show()

print("\nVisualization Completed")