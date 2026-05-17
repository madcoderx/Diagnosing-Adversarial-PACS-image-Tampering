import os
import numpy as np

from flask import Flask
from flask import render_template
from flask import request

from werkzeug.utils import secure_filename

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# -----------------------------------
# Flask App
# -----------------------------------

app = Flask(__name__)

# -----------------------------------
# Upload Folder
# -----------------------------------

UPLOAD_FOLDER = "static/uploads"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

# -----------------------------------
# Allowed Extensions
# -----------------------------------

ALLOWED_EXTENSIONS = {
    "png",
    "jpg",
    "jpeg"
}

# -----------------------------------
# Load Model
# -----------------------------------

model = load_model(
    "models/pacs_model.h5"
)

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
# Allowed File Check
# -----------------------------------

def allowed_file(filename):

    return (

        "." in filename

        and

        filename.rsplit(".", 1)[1].lower()

        in ALLOWED_EXTENSIONS

    )

# -----------------------------------
# Home Page
# -----------------------------------

@app.route("/")

def home():

    return render_template(
        "index.html"
    )

# -----------------------------------
# Predict Route
# -----------------------------------

@app.route(
    "/predict",
    methods=["POST"]
)

def predict():

    try:

        # Check file upload
        if "file" not in request.files:

            return render_template(

                "index.html",

                error="No file uploaded"

            )

        file = request.files["file"]

        # Empty file
        if file.filename == "":

            return render_template(

                "index.html",

                error="No file selected"

            )

        # Invalid extension
        if not allowed_file(file.filename):

            return render_template(

                "index.html",

                error="Invalid file type"

            )

        # -----------------------------------
        # Save File
        # -----------------------------------

        filename = secure_filename(
            file.filename
        )

        filepath = os.path.join(

            app.config["UPLOAD_FOLDER"],

            filename

        )

        file.save(filepath)

        # -----------------------------------
        # Image Preprocessing
        # -----------------------------------

        img = image.load_img(

            filepath,

            target_size=(128, 128)

        )

        img_array = image.img_to_array(
            img
        )

        img_array = img_array / 255.0

        img_array = np.expand_dims(

            img_array,

            axis=0

        )

        # -----------------------------------
        # Prediction
        # -----------------------------------

        prediction = model.predict(

            img_array,

            verbose=0

        )

        predicted_index = np.argmax(
            prediction
        )

        confidence = np.max(
            prediction
        )

        predicted_class = class_names[
            predicted_index
        ]

        # -----------------------------------
        # Result
        # -----------------------------------

        if predicted_class == "ORIGINAL":

            result = (
                "✅ ORIGINAL PACS IMAGE"
            )

            status = "safe"

        else:

            result = (
                "⚠️ TAMPERED IMAGE DETECTED"
            )

            status = "danger"

        confidence_text = (
            f"{confidence*100:.2f}%"
        )

        # -----------------------------------
        # Return Result
        # -----------------------------------

        return render_template(

            "index.html",

            prediction=result,

            attack_type=predicted_class,

            confidence=confidence_text,

            image_file=filename,

            status=status

        )

    except Exception as e:

        return render_template(

            "index.html",

            error=str(e)

        )

# -----------------------------------
# Run App
# -----------------------------------

if __name__ == "__main__":

    app.run(

        host="127.0.0.1",

        port=5000,

        debug=False

    )