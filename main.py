import os
import requests
import tensorflow as tf
import numpy as np
from flask import Flask, request, jsonify
from PIL import Image
from io import BytesIO

app = Flask(__name__)

# ====== CONFIGURATION ======
MODEL_PATH = "combined_model.h5"
GOOGLE_DRIVE_FILE_ID = "1Ta8VqtUEguXfzs0iN7pNefyW2gKHi0z3"
MODEL_URL = f"https://drive.google.com/uc?export=download&id={GOOGLE_DRIVE_FILE_ID}"

# ====== DOWNLOAD MODEL IF NOT EXISTS ======
def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model from Google Drive...")
        response = requests.get(MODEL_URL)
        if response.status_code == 200:
            with open(MODEL_PATH, "wb") as f:
                f.write(response.content)
            print("✅ Model downloaded successfully.")
        else:
            print(f"❌ Failed to download model. Status Code: {response.status_code}")
            exit(1)

download_model()

# ====== LOAD MODEL ======
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    exit(1)

# ====== IMAGE PREPROCESSING ======
def prepare_image(image_bytes):
    try:
        image = Image.open(BytesIO(image_bytes)).convert('RGB')
        image = image.resize((224, 224))  # Match model input size
        image = np.array(image) / 255.0   # Normalize
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        return image
    except Exception as e:
        raise ValueError(f"Image processing error: {str(e)}")

# ====== ROUTES ======
@app.route('/')
def index():
    return "✅ Oral Cancer Detection API is Running!"

@app.route('/health')
def health():
    return jsonify({'status': 'ok'})

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded. Use form-data key: image'}), 400

    file = request.files['image']
    img_bytes = file.read()

    try:
        processed_img = prepare_image(img_bytes)
        prediction = model.predict(processed_img)
        result = prediction.tolist()

        return jsonify({'prediction': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ====== LOCAL SERVER RUN ======
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
