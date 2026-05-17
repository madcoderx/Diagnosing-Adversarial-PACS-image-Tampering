import tensorflow as tf
import numpy as np
import cv2
import os
import sys

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# -----------------------------------
# Constants
# -----------------------------------

IMG_SIZE = 128

# -----------------------------------
# Load Model
# -----------------------------------

model = load_model(
    "models/pacs_model.h5",
    compile=False
)

# Initialize model properly
dummy = np.zeros(
    (1, IMG_SIZE, IMG_SIZE, 3),
    dtype=np.float32
)

_ = model(dummy)

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
# Load Image Function
# -----------------------------------

def load_image(path):

    img = image.load_img(
        path,
        target_size=(IMG_SIZE, IMG_SIZE)
    )

    img = image.img_to_array(img)

    img = img / 255.0

    img = np.expand_dims(
        img,
        axis=0
    )

    return tf.convert_to_tensor(
        img,
        dtype=tf.float32
    )

# -----------------------------------
# FGSM Attack Function
# -----------------------------------

def fgsm_attack(

    image_tensor,

    true_label,

    epsilon=0.02

):

    loss_object = (
        tf.keras.losses.CategoricalCrossentropy()
    )

    with tf.GradientTape() as tape:

        tape.watch(image_tensor)

        prediction = model(
            image_tensor,
            training=False
        )

        loss = loss_object(
            true_label,
            prediction
        )

    # Compute gradients
    gradient = tape.gradient(
        loss,
        image_tensor
    )

    # Sign of gradients
    signed_grad = tf.sign(
        gradient
    )

    # Create adversarial image
    adv_image = (

        image_tensor

        + epsilon * signed_grad

    )

    # Clip pixel values
    adv_image = tf.clip_by_value(
        adv_image,
        0,
        1
    )

    return adv_image, signed_grad

# -----------------------------------
# Test Folder
# -----------------------------------

test_folder = (
    "training/tampered/"
    "fake_pneumonia"
)

# -----------------------------------
# Get Image Files
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

# -----------------------------------
# Select First Image
# -----------------------------------

img_path = os.path.join(
    test_folder,
    files[0]
)

print("\nUsing Image:")
print(img_path)

# -----------------------------------
# Load Original Image
# -----------------------------------

original = load_image(
    img_path
)

# -----------------------------------
# Initial Prediction
# -----------------------------------

prediction = model.predict(
    original,
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
    "\nOriginal Prediction:"
)

print(
    f"{predicted_label}"
)

print(
    f"Confidence: "
    f"{confidence:.2f}"
)

# -----------------------------------
# Create True Label Tensor
# -----------------------------------

true_label = tf.one_hot(
    predicted_index,
    depth=len(class_names)
)

true_label = tf.reshape(
    true_label,
    (1, len(class_names))
)

# -----------------------------------
# Generate FGSM Attack
# -----------------------------------

adv_img, noise = fgsm_attack(

    original,

    true_label,

    epsilon=0.02

)

print("\nFGSM attack generated")

# -----------------------------------
# Predict Adversarial Image
# -----------------------------------

adv_prediction = model.predict(
    adv_img,
    verbose=0
)

adv_index = np.argmax(
    adv_prediction
)

adv_confidence = np.max(
    adv_prediction
)

adv_label = class_names[
    adv_index
]

print(
    "\nAdversarial Prediction:"
)

print(
    f"{adv_label}"
)

print(
    f"Confidence: "
    f"{adv_confidence:.2f}"
)

# -----------------------------------
# Convert Images
# -----------------------------------

orig = original.numpy()[0]

adv = adv_img.numpy()[0]

noise = noise.numpy()[0]

# Convert to uint8
orig = (
    orig * 255
).astype(np.uint8)

adv = (
    adv * 255
).astype(np.uint8)

# Normalize noise
noise = (

    (noise - noise.min())

    /

    (noise.max() - noise.min() + 1e-8)

    * 255

).astype(np.uint8)

# RGB -> BGR
orig = cv2.cvtColor(
    orig,
    cv2.COLOR_RGB2BGR
)

adv = cv2.cvtColor(
    adv,
    cv2.COLOR_RGB2BGR
)

noise = cv2.cvtColor(
    noise,
    cv2.COLOR_RGB2BGR
)

# -----------------------------------
# Create Results Folder
# -----------------------------------

os.makedirs(
    "results",
    exist_ok=True
)

# -----------------------------------
# Save Images
# -----------------------------------

cv2.imwrite(
    "results/original_fgsm.png",
    orig
)

cv2.imwrite(
    "results/adversarial_fgsm.png",
    adv
)

cv2.imwrite(
    "results/fgsm_noise.png",
    noise
)

print("\nImages saved successfully")

print("\nSaved Files:")

print(
    "1. results/original_fgsm.png"
)

print(
    "2. results/adversarial_fgsm.png"
)

print(
    "3. results/fgsm_noise.png"
)

print(
    "\nFGSM attack completed successfully"
)