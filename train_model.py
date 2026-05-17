import os
import sys
import cv2
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# -----------------------------------
# CONFIG
# -----------------------------------

IMG_SIZE = 128

# -----------------------------------
# TAMPER MODEL DATASET ONLY
# -----------------------------------

dataset_paths = {

    "ORIGINAL": "training/original",

    "BLACKOUT_ATTACK": "training/tampered/blackout_attack",

    "BLUR_ATTACK": "training/tampered/blur_attack",

    "GAUSSIAN_NOISE": "training/tampered/gaussian_noise",

    "FAKE_PNEUMONIA": "training/tampered/fake_pneumonia"
}

# -----------------------------------
# LOAD DATA
# -----------------------------------

X = []
y = []

for label, folder_path in dataset_paths.items():

    print(f"\nLoading Class: {label}")

    if not os.path.exists(folder_path):
        print(f"Missing folder: {folder_path}")
        continue

    files = [
        f for f in os.listdir(folder_path)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    print("Images:", len(files))

    for file in files:

        path = os.path.join(folder_path, file)
        img = cv2.imread(path)

        if img is None:
            continue

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img / 255.0

        X.append(img)
        y.append(label)

# -----------------------------------
# VALIDATION CHECK
# -----------------------------------

X = np.array(X)
y = np.array(y)

print("\nTotal Images:", len(X))

if len(X) == 0:
    print("Dataset empty")
    sys.exit()

# -----------------------------------
# ENCODING
# -----------------------------------

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

class_names = encoder.classes_

print("\nClasses:", class_names)

# -----------------------------------
# SPLIT DATA
# -----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_categorical,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

# -----------------------------------
# MODEL
# -----------------------------------

model = Sequential()

model.add(Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(128, (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(len(class_names), activation='softmax'))

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# -----------------------------------
# TRAIN
# -----------------------------------

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=15,
    batch_size=32
)

# -----------------------------------
# SAVE MODEL
# -----------------------------------

os.makedirs("models", exist_ok=True)

model.save("models/tamper_model.h5")

print("\nTamper model saved successfully")

# -----------------------------------
# EVALUATION
# -----------------------------------

loss, acc = model.evaluate(X_test, y_test)
print(f"\nAccuracy: {acc*100:.2f}%")

pred = model.predict(X_test)

y_pred = np.argmax(pred, axis=1)
y_true = np.argmax(y_test, axis=1)

print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=class_names))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_true, y_pred))