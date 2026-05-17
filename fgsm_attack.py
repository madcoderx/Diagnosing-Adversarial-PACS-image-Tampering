import tensorflow as tf
import numpy as np
import cv2
import os

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# -----------------------------------
# CONFIG
# -----------------------------------

IMG_SIZE = 128

# -----------------------------------
# LOAD MODEL
# -----------------------------------

model = load_model("models/tamper_model.h5", compile=False)

_ = model(tf.zeros((1, IMG_SIZE, IMG_SIZE, 3)))

print("\nModel Loaded")

# -----------------------------------
# CLASS LABELS
# -----------------------------------

class_names = [
    "BLACKOUT_ATTACK",
    "BLUR_ATTACK",
    "FAKE_PNEUMONIA",
    "GAUSSIAN_NOISE",
    "ORIGINAL"
]

# -----------------------------------
# IMAGE LOADER
# -----------------------------------

def load_image(path):
    img = image.load_img(path, target_size=(IMG_SIZE, IMG_SIZE))
    img = image.img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return tf.convert_to_tensor(img, dtype=tf.float32)

# -----------------------------------
# FGSM FUNCTION (FIXED)
# -----------------------------------

def fgsm_attack(image_tensor, epsilon=0.02):

    loss_fn = tf.keras.losses.CategoricalCrossentropy()

    with tf.GradientTape() as tape:
        tape.watch(image_tensor)

        prediction = model(image_tensor, training=False)

        # 🔥 FIX: use predicted class but STOP gradients from collapsing
        pred_label = tf.argmax(prediction, axis=1)

        true_label = tf.one_hot(pred_label, depth=len(class_names))

        loss = loss_fn(true_label, prediction)

    gradient = tape.gradient(loss, image_tensor)

    # 🔥 FIX: gradient fallback
    if gradient is None:
        raise ValueError("Gradient computation failed")

    signed_grad = tf.sign(gradient)

    adv_image = image_tensor + epsilon * signed_grad
    adv_image = tf.clip_by_value(adv_image, 0, 1)

    return adv_image, signed_grad

# -----------------------------------
# TEST IMAGE
# -----------------------------------

test_folder = "training/tampered/fake_pneumonia"
files = os.listdir(test_folder)

img_path = os.path.join(test_folder, files[0])

print("\nUsing:", img_path)

# -----------------------------------
# ORIGINAL IMAGE
# -----------------------------------

original = load_image(img_path)

pred = model.predict(original, verbose=0)

idx = np.argmax(pred)
conf = np.max(pred)

print("\nOriginal Prediction:", class_names[idx])
print("Confidence:", conf)

# -----------------------------------
# FGSM ATTACK
# -----------------------------------

adv_img, noise = fgsm_attack(original, epsilon=0.02)

adv_pred = model.predict(adv_img, verbose=0)

adv_idx = np.argmax(adv_pred)
adv_conf = np.max(adv_pred)

print("\nAdversarial Prediction:", class_names[adv_idx])
print("Confidence:", adv_conf)

# -----------------------------------
# CONVERT IMAGES SAFELY
# -----------------------------------

orig = (original.numpy()[0] * 255).astype(np.uint8)
adv = (adv_img.numpy()[0] * 255).astype(np.uint8)

noise = noise.numpy()[0]

# 🔥 FIX: safe normalization
noise_min = noise.min()
noise_max = noise.max()

if noise_max - noise_min == 0:
    noise = np.zeros_like(noise)
else:
    noise = (noise - noise_min) / (noise_max - noise_min)

noise = (noise * 255).astype(np.uint8)

# RGB → BGR
orig = cv2.cvtColor(orig, cv2.COLOR_RGB2BGR)
adv = cv2.cvtColor(adv, cv2.COLOR_RGB2BGR)
noise = cv2.cvtColor(noise, cv2.COLOR_RGB2BGR)

# -----------------------------------
# SAVE
# -----------------------------------

os.makedirs("results", exist_ok=True)

cv2.imwrite("results/original_fgsm.png", orig)
cv2.imwrite("results/adversarial_fgsm.png", adv)
cv2.imwrite("results/fgsm_noise.png", noise)

print("\nFGSM Completed Successfully")