from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Load Keras .h5 model
model = tf.keras.models.load_model('combined_model.h5')

# Modify according to your model's expected input size
IMG_SIZE = (224, 224)

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize(IMG_SIZE)
    image_np = np.array(image).astype(np.float32) / 255.0
    image_np = np.expand_dims(image_np, axis=0)  # Add batch dimension
    return image_np

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    file = request.files['image']
    image_bytes = file.read()
    input_data = preprocess_image(image_bytes)

    predictions = model.predict(input_data)[0]
    predicted_class = int(np.argmax(predictions))
    probability = float(np.max(predictions))

    return jsonify({
        "prediction": f"Class {predicted_class}",
        "probability": round(probability * 100, 2)
    })

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
