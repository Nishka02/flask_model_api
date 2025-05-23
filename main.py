import os
import numpy as np
from flask import Flask, request, jsonify
from PIL import Image
from io import BytesIO
import tensorflow as tf  # or use `tflite-runtime` for lightweight environments

app = Flask(__name__)

# ====== CONFIGURATION ======
MODEL_PATH = "model.tflite"  # Model file in the same directory

# ====== LOAD TFLITE MODEL ======
try:
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("✅ TFLite model loaded successfully.")
except Exception as e:
    print(f"❌ Failed to load TFLite model: {e}")
    exit(1)

# ====== IMAGE PREPROCESSING ======
def prepare_image(image_bytes):
    try:
        image = Image.open(BytesIO(image_bytes)).convert('RGB')
        image = image.resize((224, 224))  # Match your model input size
        image = np.array(image, dtype=np.float32) / 255.0  # Normalize
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        return image
    except Exception as e:
        raise ValueError(f"Image processing error: {str(e)}")

# ====== ROUTES ======
@app.route('/')
def index():
    return "✅ Oral Cancer Detection TFLite API is Running!"

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

        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], processed_img)

        # Run inference
        interpreter.invoke()

        # Get output tensor
        output_data = interpreter.get_tensor(output_details[0]['index'])
        result = output_data.tolist()

        return jsonify({'prediction': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ====== LOCAL SERVER RUN ======
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
