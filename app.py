from flask import Flask, render_template, request, jsonify
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
import numpy as np
import cv2
from PIL import Image
import io
import os

# Initialize the Flask app
app = Flask(__name__)

# Load the pretrained MobileNetV2 model
model = MobileNetV2(weights='imagenet')

def process_image(img_data):
    img = Image.open(io.BytesIO(img_data))
    img = img.resize((224, 224))
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get image data from the request
    img_data = request.files['file'].read()

    # Process the image
    img = process_image(img_data)

    # Predict using the model
    prediction = model.predict(img)
    decoded_prediction = decode_predictions(prediction, top=3)[0]

    # Extract the top prediction
    top_prediction = decoded_prediction[0]
    class_name = top_prediction[1]
    confidence = round(top_prediction[2] * 100, 2)

    # Dummy age and emotion detection (Replace with actual model)
    age = "Unknown"
    emotion = "Unknown"

    # Respond with prediction results
    result = {
        'class_name': class_name,
        'confidence': confidence,
        'age': age,
        'emotion': emotion
    }
    
    return jsonify(result)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port, debug=True)
