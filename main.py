import os
import requests
import tensorflow as tf
import numpy as np
from flask import Flask, request, jsonify
from PIL import Image
from io import BytesIO

app = Flask(__name__)

# ====== CONFIG ======
MODEL_PATH = "combined_model.h5"

# 🔽 Use your actual model file ID from Google Drive (shared as public)
GOOGLE_DRIVE_FILE_ID = "1Ta8VqtUEguXfzs0iN7pNefyW2gKHi0z3"
MODEL_URL = f"https://drive.google.com/file/d/1Ta8VqtUEguXfzs0iN7pNefyW2gKHi0z3/view?usp=drive_link"

# ====== DOWNLOAD MODEL ======
def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model...")
        response = requests.get(MODEL_URL)
        with open(MODEL_PATH, "wb") as f:
            f.write(response.content)
        print("Model downloaded.")

download_model()

# ====== LOAD MODEL ======
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded.")

# ====== PREDICTION FUNCTION ======
def prepare_image(image_bytes):
    image = Image.open(BytesIO(image_bytes)).convert('RGB')
    image = image.resize((224, 224))  # adjust size to match your model input
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# ====== ROUTES ======
@app.route('/')
def index():
    return "Oral Cancer Detection API is Running!"

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    img_bytes = file.read()

    try:
        processed = prepare_image(img_bytes)
        prediction = model.predict(processed)
        result = prediction.tolist()

        return jsonify({'prediction': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ====== RUN LOCALLY ======
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
