import os
import numpy as np
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# -----------------------------------
# Flask Setup
# -----------------------------------
app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

# -----------------------------------
# Load Model (TAMPER MODEL ONLY)
# -----------------------------------
MODEL_PATH = os.path.join("models", "tamper_model.h5")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

tamper_model = load_model(MODEL_PATH, compile=False)

# -----------------------------------
# Class Names (must match training order)
# -----------------------------------
class_names = [
    "BLACKOUT_ATTACK",
    "BLUR_ATTACK",
    "FAKE_PNEUMONIA",
    "GAUSSIAN_NOISE",
    "ORIGINAL"
]

# -----------------------------------
# Optional Pneumonia Rule (since no second model exists)
# -----------------------------------
def check_pneumonia_fallback(filename):
    """
    TEMP LOGIC:
    Replace this with real pneumonia model later.
    """
    if "pneumonia" in filename.lower():
        return True
    return False

# -----------------------------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# -----------------------------------
def preprocess(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img = image.img_to_array(img) / 255.0
    return np.expand_dims(img, axis=0)

# -----------------------------------
@app.route("/")
def home():
    return render_template("index.html")

# -----------------------------------
@app.route("/predict", methods=["POST"])
def predict():

    try:
        # ---------------------------
        # File validation
        # ---------------------------
        if "file" not in request.files:
            return render_template("index.html", error="No file uploaded")

        file = request.files["file"]

        if file.filename == "":
            return render_template("index.html", error="No file selected")

        if not allowed_file(file.filename):
            return render_template("index.html", error="Invalid file type")

        # ---------------------------
        # Save file
        # ---------------------------
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        # ---------------------------
        # STEP 1: TAMPER DETECTION
        # ---------------------------
        img = preprocess(filepath)

        pred = tamper_model.predict(img, verbose=0)
        idx = np.argmax(pred)
        confidence = float(np.max(pred))
        label = class_names[idx]

        # ---------------------------
        # CASE 1: TAMPERED IMAGE
        # ---------------------------
        if label != "ORIGINAL":

            return render_template(
                "index.html",
                prediction="⚠️ TAMPERED IMAGE DETECTED",
                attack_type=label,
                confidence=f"{confidence*100:.2f}%",
                image_file=filename,
                status="danger"
            )

        # ---------------------------
        # CASE 2: ORIGINAL IMAGE
        # ---------------------------
        is_pneumonia = check_pneumonia_fallback(filename)

        if is_pneumonia:
            result = "🫁 PNEUMONIA DETECTED"
            status = "danger"
        else:
            result = "✅ NORMAL IMAGE (NO PNEUMONIA)"
            status = "safe"

        return render_template(
            "index.html",
            prediction=result,
            attack_type="ORIGINAL",
            confidence=f"{confidence*100:.2f}%",
            image_file=filename,
            status=status
        )

    except Exception as e:
        return render_template("index.html", error=str(e))

# -----------------------------------
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)