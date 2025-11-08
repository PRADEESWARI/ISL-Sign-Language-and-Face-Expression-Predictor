import os
import numpy as np
import cv2
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Paths
FER_MODEL_PATH = "face_cnn_model.h5"   # Your trained FER model
ISL_MODEL_PATH = "isl_cnn_model.h5"    # Your trained ISL model
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Parameters
IMG_SIZE_FER = (48, 48)
IMG_SIZE_ISL = (64, 64)

# Load models
fer_model = load_model(FER_MODEL_PATH)
isl_model = load_model(ISL_MODEL_PATH)

# Class names
FER_CLASSES = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# Load ISL class indices (assuming folders = class names)
ISL_CLASSES = sorted(os.listdir("D:/EE - sem_v/isl_dataset/train"))

# Initialize Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    uploaded_img = None

    if request.method == "POST":
        gesture_type = request.form.get("gesture_type")  # user selection (face / hand)
        file = request.files["file"]

        if file and gesture_type:
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(img_path)
            uploaded_img = img_path

            if gesture_type == "face":
                # --- Face Expression Prediction ---
                img_cv = cv2.imread(img_path)
                gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

                if len(faces) > 0:
                    (x, y, w, h) = faces[0]
                    face = gray[y:y+h, x:x+w]
                    img = cv2.resize(face, IMG_SIZE_FER)
                    img = img / 255.0
                    img = np.expand_dims(img, axis=(0, -1))
                    prediction_idx = np.argmax(fer_model.predict(img))
                    prediction = f"Face Emotion: {FER_CLASSES[prediction_idx]}"
                else:
                    prediction = "⚠️ No face detected in the image."

            elif gesture_type == "hand":
                # --- Hand Gesture Prediction ---
                img = image.load_img(img_path, target_size=IMG_SIZE_ISL)
                img_array = image.img_to_array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                prediction_idx = np.argmax(isl_model.predict(img_array))
                prediction = f"Hand Gesture: {ISL_CLASSES[prediction_idx]}"

    return render_template("index.html", prediction=prediction, uploaded_img=uploaded_img)


if __name__ == "__main__":
    app.run(debug=True)
