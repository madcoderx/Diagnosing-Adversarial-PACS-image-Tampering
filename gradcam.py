import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# -----------------------------------
# CONFIG
# -----------------------------------

IMG_SIZE = 128

# -----------------------------------
# LOAD MODEL
# -----------------------------------

model = load_model("models/tamper_model.h5", compile=False)

print("\nModel Loaded")

# FORCE BUILD (IMPORTANT)
dummy = tf.zeros((1, IMG_SIZE, IMG_SIZE, 3))
_ = model(dummy)

# -----------------------------------
# CLASS LABELS
# -----------------------------------

class_names = [
    "ORIGINAL",
    "BLACKOUT_ATTACK",
    "BLUR_ATTACK",
    "GAUSSIAN_NOISE",
    "FAKE_PNEUMONIA"
]

# -----------------------------------
# FIND LAST CONV LAYER
# -----------------------------------

last_conv_layer = None
for layer in reversed(model.layers):
    if isinstance(layer, tf.keras.layers.Conv2D):
        last_conv_layer = layer
        break

print("Last Conv Layer:", last_conv_layer.name)

# -----------------------------------
# IMAGE LOAD
# -----------------------------------

test_folder = "training/tampered/fake_pneumonia"
file = os.listdir(test_folder)[0]
img_path = os.path.join(test_folder, file)

img = cv2.imread(img_path)
original = img.copy()

img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
img_array = img_resized.astype(np.float32) / 255.0
img_array = np.expand_dims(img_array, axis=0)

img_tensor = tf.convert_to_tensor(img_array)

# -----------------------------------
# GRADIENT TAPE ONLY (NO MODEL GRAPH)
# -----------------------------------

with tf.GradientTape() as tape:
    tape.watch(img_tensor)

    # Forward pass manually
    x = img_tensor

    for layer in model.layers:
        x = layer(x)

        if layer.name == last_conv_layer.name:
            conv_outputs = x

    predictions = x

    pred_index = tf.argmax(predictions[0])
    loss = predictions[:, pred_index]

# -----------------------------------
# GRADIENTS
# -----------------------------------

grads = tape.gradient(loss, conv_outputs)

if grads is None:
    raise ValueError("Gradient failed")

# -----------------------------------
# HEATMAP
# -----------------------------------

pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
conv_outputs = conv_outputs[0]

heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

heatmap = tf.maximum(heatmap, 0)
heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)

heatmap = heatmap.numpy()

# -----------------------------------
# RESIZE HEATMAP
# -----------------------------------

heatmap = cv2.resize(
    heatmap,
    (original.shape[1], original.shape[0])
)

heatmap = np.uint8(255 * heatmap)
heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

overlay = cv2.addWeighted(original, 0.6, heatmap_color, 0.4, 0)

# -----------------------------------
# FINAL PREDICTION
# -----------------------------------

final_pred = model.predict(img_array, verbose=0)

idx = np.argmax(final_pred)
conf = np.max(final_pred)

label = class_names[idx]

print("\nPrediction:", label)
print("Confidence:", conf)

# -----------------------------------
# SAVE
# -----------------------------------

os.makedirs("results", exist_ok=True)

cv2.imwrite("results/heatmap.png", heatmap_color)
cv2.imwrite("results/overlay.png", overlay)

# -----------------------------------
# SHOW
# -----------------------------------

plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
plt.title("Original")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB))
plt.title("Heatmap")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
plt.title(f"{label} ({conf:.2f})")
plt.axis("off")

plt.tight_layout()
plt.show()

print("\nGradCAM SUCCESS")