import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout

from tensorflow.keras.utils import to_categorical

# -----------------------------------
# Constants
# -----------------------------------

IMG_SIZE = 128

# -----------------------------------
# Dataset Paths
# -----------------------------------

dataset_paths = {

    "ORIGINAL":
    "training/original",

    "BLACKOUT_ATTACK":
    "training/tampered/blackout_attack",

    "BLUR_ATTACK":
    "training/tampered/blur_attack",

    "FAKE_PNEUMONIA":
    "training/tampered/fake_pneumonia",

    "GAUSSIAN_NOISE":
    "training/tampered/gaussian_noise"

}

# -----------------------------------
# Store Data
# -----------------------------------

X = []

y = []

# -----------------------------------
# Load Images
# -----------------------------------

for label, folder_path in dataset_paths.items():

    print(f"\nLoading Class: {label}")

    # Folder check
    if not os.path.exists(folder_path):

        print(f"Folder NOT found: {folder_path}")

        continue

    # Read files
    files = [

        f for f in os.listdir(folder_path)

        if f.lower().endswith(
            (
                ".png",
                ".jpg",
                ".jpeg"
            )
        )

    ]

    print(f"Images Found: {len(files)}")

    # Process images
    for file in files:

        path = os.path.join(
            folder_path,
            file
        )

        try:

            img = cv2.imread(path)

            # Broken image check
            if img is None:

                print(f"Skipping: {file}")

                continue

            # Resize
            img = cv2.resize(
                img,
                (IMG_SIZE, IMG_SIZE)
            )

            # Normalize
            img = img / 255.0

            # Save data
            X.append(img)

            y.append(label)

        except Exception as e:

            print(f"Error: {file}")

            print(e)

# -----------------------------------
# Convert to NumPy Arrays
# -----------------------------------

X = np.array(X)

y = np.array(y)

# -----------------------------------
# Dataset Check
# -----------------------------------

print("\nDataset Summary")

print("Total Images:", len(X))

print("Total Labels:", len(y))

# Empty dataset check
if len(X) == 0 or len(y) == 0:

    print("\nERROR: Dataset is empty")

    sys.exit()

# -----------------------------------
# Encode Labels
# -----------------------------------

encoder = LabelEncoder()

y_encoded = encoder.fit_transform(y)

y_categorical = to_categorical(
    y_encoded
)

# -----------------------------------
# Save Class Names
# -----------------------------------

class_names = encoder.classes_

print("\nClasses:")

for i, name in enumerate(class_names):

    print(i, "=", name)

# -----------------------------------
# Train Test Split
# -----------------------------------

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y_categorical,

    test_size=0.2,

    random_state=42,

    stratify=y_encoded

)

print("\nTraining Samples:", len(X_train))

print("Testing Samples:", len(X_test))

# -----------------------------------
# CNN Model
# -----------------------------------

model = Sequential()

# -----------------------------------
# First Conv Block
# -----------------------------------

model.add(

    Conv2D(

        32,

        (3, 3),

        activation='relu',

        input_shape=(
            IMG_SIZE,
            IMG_SIZE,
            3
        )

    )

)

model.add(

    MaxPooling2D(
        pool_size=(2, 2)
    )

)

# -----------------------------------
# Second Conv Block
# -----------------------------------

model.add(

    Conv2D(

        64,

        (3, 3),

        activation='relu'

    )

)

model.add(

    MaxPooling2D(
        pool_size=(2, 2)
    )

)

# -----------------------------------
# Third Conv Block
# -----------------------------------

model.add(

    Conv2D(

        128,

        (3, 3),

        activation='relu'

    )

)

model.add(

    MaxPooling2D(
        pool_size=(2, 2)
    )

)

# -----------------------------------
# Flatten Layer
# -----------------------------------

model.add(Flatten())

# -----------------------------------
# Dense Layer
# -----------------------------------

model.add(

    Dense(

        128,

        activation='relu'

    )

)

# -----------------------------------
# Dropout
# -----------------------------------

model.add(

    Dropout(0.5)

)

# -----------------------------------
# Output Layer
# -----------------------------------

model.add(

    Dense(

        len(class_names),

        activation='softmax'

    )

)

# -----------------------------------
# Compile Model
# -----------------------------------

model.compile(

    optimizer='adam',

    loss='categorical_crossentropy',

    metrics=['accuracy']

)

# -----------------------------------
# Model Summary
# -----------------------------------

print("\nModel Summary:\n")

model.summary()

# -----------------------------------
# Train Model
# -----------------------------------

print("\nTraining Started...\n")

history = model.fit(

    X_train,

    y_train,

    validation_data=(
        X_test,
        y_test
    ),

    epochs=15,

    batch_size=32

)

# -----------------------------------
# Create Models Folder
# -----------------------------------

os.makedirs(
    "models",
    exist_ok=True
)

# -----------------------------------
# Save Model
# -----------------------------------

model.save(
    "models/pacs_model.h5"
)

print("\nModel Saved Successfully")

# -----------------------------------
# Evaluate Model
# -----------------------------------

loss, accuracy = model.evaluate(

    X_test,

    y_test

)

print(
    f"\nTest Accuracy: "
    f"{accuracy * 100:.2f}%"
)

# -----------------------------------
# Predictions
# -----------------------------------

predictions = model.predict(
    X_test
)

predicted_labels = np.argmax(
    predictions,
    axis=1
)

true_labels = np.argmax(
    y_test,
    axis=1
)

# -----------------------------------
# Classification Report
# -----------------------------------

print("\nClassification Report:\n")

print(

    classification_report(

        true_labels,

        predicted_labels,

        target_names=class_names

    )

)

# -----------------------------------
# Confusion Matrix
# -----------------------------------

cm = confusion_matrix(

    true_labels,

    predicted_labels

)

print("\nConfusion Matrix:\n")

print(cm)

# -----------------------------------
# Accuracy Graph
# -----------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    history.history['accuracy'],
    label='Training Accuracy'
)

plt.plot(
    history.history['val_accuracy'],
    label='Validation Accuracy'
)

plt.title("Model Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend()

plt.grid(True)

# -----------------------------------
# Save Accuracy Graph
# -----------------------------------

os.makedirs(
    "results",
    exist_ok=True
)

plt.savefig(
    "results/accuracy_graph.png"
)

plt.show()

# -----------------------------------
# Loss Graph
# -----------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    history.history['loss'],
    label='Training Loss'
)

plt.plot(
    history.history['val_loss'],
    label='Validation Loss'
)

plt.title("Model Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

plt.grid(True)

# -----------------------------------
# Save Loss Graph
# -----------------------------------

plt.savefig(
    "results/loss_graph.png"
)

plt.show()

print("\nTraining completed successfully")