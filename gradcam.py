import cv2
import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model

# -----------------------------------
# Constants
# -----------------------------------

IMG_SIZE = 128

# -----------------------------------
# Load Trained Model
# -----------------------------------

model = load_model(
    "models/pacs_model.h5",
    compile=False
)

print("\nModel Loaded Successfully")

# -----------------------------------
# Force Model Build
# -----------------------------------

input_shape = (1, 128, 128, 3)

dummy_input = np.zeros(
    input_shape,
    dtype=np.float32
)

# Call model once
_ = model(dummy_input)

print("\nModel Initialized Successfully")

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
# Find Last Conv Layer
# -----------------------------------

last_conv_layer = None

for layer in reversed(model.layers):

    if isinstance(
        layer,
        tf.keras.layers.Conv2D
    ):

        last_conv_layer = layer.name

        break

print(
    "\nLast Conv Layer:",
    last_conv_layer
)

# -----------------------------------
# Select Image Folder
# -----------------------------------

test_folder = (
    "training/tampered/"
    "fake_pneumonia"
)

# -----------------------------------
# Select First Image
# -----------------------------------

files = [

    f for f in os.listdir(test_folder)

    if f.lower().endswith(
        (".png", ".jpg", ".jpeg")
    )

]

if len(files) == 0:

    print("\nNo images found")

    sys.exit()

file_name = files[0]

image_path = os.path.join(
    test_folder,
    file_name
)

print(
    "\nLoading Image:",
    image_path
)

# -----------------------------------
# Read Image
# -----------------------------------

image = cv2.imread(image_path)

if image is None:

    print("\nError loading image")

    sys.exit()

# Original Image
original = image.copy()

# -----------------------------------
# Preprocess Image
# -----------------------------------

image = cv2.resize(
    image,
    (IMG_SIZE, IMG_SIZE)
)

image = image.astype(
    "float32"
) / 255.0

image = np.expand_dims(
    image,
    axis=0
)

image_tensor = tf.convert_to_tensor(
    image
)

print(
    "\nImage Shape:",
    image_tensor.shape
)

# -----------------------------------
# Create New Input Layer
# -----------------------------------

inputs = tf.keras.Input(
    shape=(128, 128, 3)
)

# -----------------------------------
# Forward Pass Through Layers
# -----------------------------------

x = inputs

last_conv_output = None

for layer in model.layers:

    x = layer(x)

    # Save last conv output
    if layer.name == last_conv_layer:

        last_conv_output = x

# Final predictions
outputs = x

# -----------------------------------
# Create GradCAM Model
# -----------------------------------

grad_model = tf.keras.models.Model(

    inputs=inputs,

    outputs=[

        last_conv_output,

        outputs

    ]

)

print("\nGradCAM model created")

# -----------------------------------
# Compute Gradients
# -----------------------------------

with tf.GradientTape() as tape:

    conv_outputs, predictions = grad_model(
        image_tensor
    )

    predicted_class = tf.argmax(
        predictions[0]
    )

    loss = predictions[
        :,
        predicted_class
    ]

# -----------------------------------
# Compute Gradients
# -----------------------------------

grads = tape.gradient(
    loss,
    conv_outputs
)

if grads is None:

    print(
        "\nERROR: Gradients are None"
    )

    sys.exit()

# -----------------------------------
# Compute Channel Importance
# -----------------------------------

pooled_grads = tf.reduce_mean(

    grads,

    axis=(0, 1, 2)

)

# Remove Batch Dimension
conv_outputs = conv_outputs[0]

# -----------------------------------
# Generate Heatmap
# -----------------------------------

heatmap = tf.reduce_sum(

    conv_outputs * pooled_grads,

    axis=-1

)

# Remove Negative Values
heatmap = tf.maximum(
    heatmap,
    0
)

# Normalize Heatmap
heatmap = heatmap / (
    tf.reduce_max(heatmap) + 1e-8
)

heatmap = heatmap.numpy()

# -----------------------------------
# Resize Heatmap
# -----------------------------------

heatmap = cv2.resize(

    heatmap,

    (
        original.shape[1],
        original.shape[0]
    )

)

# Convert to uint8
heatmap = np.uint8(
    255 * heatmap
)

# -----------------------------------
# Apply Color Map
# -----------------------------------

heatmap_color = cv2.applyColorMap(

    heatmap,

    cv2.COLORMAP_JET

)

# -----------------------------------
# Overlay Heatmap
# -----------------------------------

superimposed = cv2.addWeighted(

    original,
    0.6,

    heatmap_color,
    0.4,

    0

)

# -----------------------------------
# Predicted Class
# -----------------------------------

prediction = model.predict(
    image_tensor,
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

print(
    f"\nPrediction: "
    f"{predicted_label}"
)

print(
    f"Confidence: "
    f"{confidence:.2f}"
)

# -----------------------------------
# Create Results Folder
# -----------------------------------

os.makedirs(
    "results",
    exist_ok=True
)

# -----------------------------------
# Save Outputs
# -----------------------------------

cv2.imwrite(

    "results/gradcam_heatmap.png",

    heatmap_color

)

cv2.imwrite(

    "results/gradcam_output.png",

    superimposed

)

print(
    "\nGradCAM images saved successfully"
)

# -----------------------------------
# Display Results
# -----------------------------------

plt.figure(figsize=(15, 5))

# -----------------------------------
# Original Image
# -----------------------------------

plt.subplot(1, 3, 1)

plt.imshow(

    cv2.cvtColor(
        original,
        cv2.COLOR_BGR2RGB
    )

)

plt.title("Original Image")

plt.axis("off")

# -----------------------------------
# Heatmap
# -----------------------------------

plt.subplot(1, 3, 2)

plt.imshow(

    cv2.cvtColor(
        heatmap_color,
        cv2.COLOR_BGR2RGB
    )

)

plt.title("GradCAM Heatmap")

plt.axis("off")

# -----------------------------------
# Overlay
# -----------------------------------

plt.subplot(1, 3, 3)

plt.imshow(

    cv2.cvtColor(
        superimposed,
        cv2.COLOR_BGR2RGB
    )

)

plt.title(

    f"{predicted_label}\n"
    f"Confidence: {confidence:.2f}"

)

plt.axis("off")

plt.tight_layout()

plt.show()

print(
    "\nGradCAM visualization completed"
)