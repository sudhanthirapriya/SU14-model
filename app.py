from flask import Flask, render_template, request, jsonify
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
import numpy as np
import base64
import io
from PIL import Image

# Initialize Flask app
app = Flask(__name__)

# Load the MobileNetV2 model
model = MobileNetV2(weights='imagenet')

def process_image(image_bytes):
    # Open image from bytes
    img = Image.open(io.BytesIO(image_bytes))
    # Resize to model input size
    img = img.resize((224, 224))
    # Convert image to numpy array
    img = np.array(img)
    # Expand dimensions and preprocess
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    if 'file' not in data:
        return jsonify({'error': 'No image provided'}), 400

    # Extract Base64 string and decode
    img_data = base64.b64decode(data['file'].split(',')[1])

    try:
        # Process the image
        img = process_image(img_data)
        # Make prediction
        prediction = model.predict(img)
        decoded_prediction = decode_predictions(prediction, top=3)[0]

        # Get the top prediction
        top_prediction = decoded_prediction[0]
        class_name = top_prediction[1]
        confidence = round(top_prediction[2] * 100, 2)

        # Dummy values for age and emotion (replace with actual models if needed)
        age = "Unknown"
        emotion = "Unknown"

        result = {
            'class_name': class_name,
            'confidence': confidence,
            'age': age,
            'emotion': emotion
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
