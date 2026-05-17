import os
import cv2
import matplotlib.pyplot as plt

# -----------------------------------
# CREATE RESULTS FOLDER
# -----------------------------------
os.makedirs("results", exist_ok=True)

# -----------------------------------
# 1. BAR GRAPH - DATASET DISTRIBUTION
# -----------------------------------

folders = {
    "NORMAL": "training/original",
    "BLACKOUT": "training/tampered/blackout_attack",
    "BLUR": "training/tampered/blur_attack",
    "GAUSSIAN_NOISE": "training/tampered/gaussian_noise",
    "FAKE_PNEUMONIA": "training/tampered/fake_pneumonia"
}

counts = {}

for label, path in folders.items():
    if os.path.exists(path):
        counts[label] = len([
            f for f in os.listdir(path)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])
    else:
        counts[label] = 0

plt.figure(figsize=(8, 5))
plt.bar(counts.keys(), counts.values())
plt.title("Dataset Distribution")
plt.xlabel("Class")
plt.ylabel("Number of Images")
plt.xticks(rotation=30)
plt.tight_layout()

plt.savefig("results/bar_dataset_distribution.png")
plt.show()

# -----------------------------------
# 2. HISTOGRAM - PIXEL INTENSITY
# -----------------------------------

sample_path = None

for folder in folders.values():
    if os.path.exists(folder):
        files = os.listdir(folder)
        if files:
            sample_path = os.path.join(folder, files[0])
            break

if sample_path:
    img = cv2.imread(sample_path, 0)  # grayscale

    plt.figure(figsize=(6, 4))
    plt.hist(img.ravel(), bins=256, range=(0, 256))
    plt.title("Pixel Intensity Histogram")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")

    plt.savefig("results/pixel_histogram.png")
    plt.show()
else:
    print("No image found for histogram")

# -----------------------------------
# 3. SCATTER PLOT - ORIGINAL VS TAMPERED
# -----------------------------------

orig_path = None
tamp_path = None

if os.path.exists("training/original"):
    orig_files = os.listdir("training/original")
    if orig_files:
        orig_path = os.path.join("training/original", orig_files[0])

if os.path.exists("training/tampered/blur_attack"):
    tamp_files = os.listdir("training/tampered/blur_attack")
    if tamp_files:
        tamp_path = os.path.join("training/tampered/blur_attack", tamp_files[0])

if orig_path and tamp_path:

    img1 = cv2.imread(orig_path, 0)
    img2 = cv2.imread(tamp_path, 0)

    img1 = cv2.resize(img1, (64, 64)).flatten()
    img2 = cv2.resize(img2, (64, 64)).flatten()

    plt.figure(figsize=(6, 5))
    plt.scatter(img1[:1000], img2[:1000], alpha=0.5)
    plt.title("Original vs Tampered Scatter Plot")
    plt.xlabel("Original Pixels")
    plt.ylabel("Tampered Pixels")

    plt.savefig("results/scatter_plot.png")
    plt.show()

else:
    print("Not enough images for scatter plot")

# -----------------------------------
print("\nAll visualizations completed successfully")
print("Saved in: results/")