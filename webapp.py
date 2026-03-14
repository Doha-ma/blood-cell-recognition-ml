from flask import Flask, request, render_template_string, send_from_directory
import os
from pathlib import Path
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
import json

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Paths
import os
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "blood_cell_classifier.keras")
CLASS_MAP_PATH = os.path.join(os.path.dirname(__file__), "models", "blood_cell_classifier_classes.json")
IMG_SIZE = (128, 128)

# Load model and map
model = load_model(MODEL_PATH)
with open(CLASS_MAP_PATH, "r") as f:
    class_map = json.load(f)
class_map_inv = {v: k for k, v in class_map.items()}

def predict_image(image_path):
    image = load_img(image_path, target_size=IMG_SIZE)
    array = img_to_array(image) / 255.0
    array = np.expand_dims(array, axis=0)
    predictions = model.predict(array)
    predicted_class_idx = np.argmax(predictions)
    predicted_class = class_map_inv[predicted_class_idx]
    confidence = predictions[0][predicted_class_idx]
    return predicted_class, confidence

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Blood Cell Classifier</title>
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }

        .container {
            background: white;
            padding: 40px;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
            width: 100%;
            max-width: 600px;
            text-align: center;
        }

        h1 {
            color: #333;
            margin-bottom: 10px;
            font-size: 2.5rem;
            font-weight: 700;
        }

        .subtitle {
            color: #666;
            margin-bottom: 40px;
            font-size: 1.1rem;
        }

        .upload-area {
            border: 2px dashed #667eea;
            border-radius: 15px;
            padding: 40px;
            margin-bottom: 30px;
            transition: all 0.3s ease;
            background: #f8f9ff;
        }

        .upload-area:hover {
            border-color: #764ba2;
            background: #f0f2ff;
            transform: translateY(-2px);
        }

        .upload-icon {
            font-size: 3rem;
            color: #667eea;
            margin-bottom: 15px;
        }

        .file-input-wrapper {
            position: relative;
            display: inline-block;
        }

        .file-input {
            display: none;
        }

        .upload-btn {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            padding: 15px 40px;
            border: none;
            border-radius: 50px;
            cursor: pointer;
            font-size: 1.1rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
            transition: all 0.3s ease;
            box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
        }

        .upload-btn:hover {
            transform: translateY(-3px);
            box-shadow: 0 15px 25px rgba(102, 126, 234, 0.4);
        }

        .submit-btn {
            background: #28a745;
            border: none;
            padding: 12px 30px;
            border-radius: 25px;
            color: white;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            margin-top: 20px;
        }

        .submit-btn:hover {
            background: #218838;
            transform: translateY(-2px);
        }

        .submit-btn:disabled {
            background: #6c757d;
            cursor: not-allowed;
            transform: none;
        }

        .file-name {
            margin-top: 20px;
            font-weight: 600;
            color: #667eea;
            font-size: 1.1rem;
        }

        .result-box {
            background: #e8f5e9;
            border: 2px solid #28a745;
            border-radius: 15px;
            padding: 25px;
            margin-top: 30px;
            text-align: left;
            display: none;
        }

        .result-box.show {
            display: block;
            animation: fadeIn 0.5s ease-in;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .prediction-title {
            color: #2e7d32;
            font-size: 1.5rem;
            margin-bottom: 15px;
            font-weight: 700;
        }

        .confidence-bar {
            width: 100%;
            height: 20px;
            background-color: #e0e0e0;
            border-radius: 10px;
            overflow: hidden;
            margin: 15px 0;
        }

        .confidence-fill {
            height: 100%;
            background: linear-gradient(90deg, #4caf50, #81c784);
            width: 0%;
            transition: width 1s ease-in-out;
        }

        .image-preview {
            max-width: 100%;
            max-height: 300px;
            border-radius: 10px;
            margin-top: 20px;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
        }

        .footer {
            margin-top: 30px;
            color: #666;
            font-size: 0.9rem;
        }

        @media (max-width: 600px) {
            .container {
                padding: 20px;
            }
            
            h1 {
                font-size: 2rem;
            }
            
            .upload-area {
                padding: 25px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🩸 Blood Cell Classifier</h1>
        <p class="subtitle">Upload a blood cell image for AI-powered classification</p>
        
        <div class="upload-area">
            <div class="upload-icon">📁</div>
            <h3>Choose Image</h3>
            <p style="color: #666; margin-bottom: 20px;">Supported formats: PNG, JPG, JPEG, BMP</p>
            
            <form method="POST" enctype="multipart/form-data" id="uploadForm">
                <div class="file-input-wrapper">
                    <input type="file" name="file" accept="image/*" id="fileInput" class="file-input" required>
                    <label for="fileInput" class="upload-btn">Select File</label>
                </div>
                
                <div id="fileName" class="file-name" style="display: none;"></div>
                
                <button type="submit" class="submit-btn" id="submitBtn" disabled>Classify Image</button>
            </form>
        </div>

        {% if prediction %}
        <div class="result-box show">
            <h3 class="prediction-title">Analysis Results</h3>
            <p><strong>Prediction:</strong> <span style="color: #667eea; font-weight: 700; text-transform: capitalize;">{{ prediction }}</span></p>
            <p><strong>Confidence:</strong> <span style="color: #2e7d32; font-weight: 700;">{{ "%.2f"|format(confidence * 100) }}%</span></p>
            
            <div class="confidence-bar">
                <div class="confidence-fill" style="width: {{ confidence * 100 }}%"></div>
            </div>
            
            <img src="/uploads/{{ filename }}" alt="Uploaded Image" class="image-preview">
        </div>
        {% endif %}
        
        <div class="footer">
            Powered by Deep Learning • Model trained on blood cell dataset
        </div>
    </div>

    <script>
        const fileInput = document.getElementById('fileInput');
        const fileName = document.getElementById('fileName');
        const submitBtn = document.getElementById('submitBtn');

        fileInput.addEventListener('change', function(e) {
            if (this.files && this.files[0]) {
                const file = this.files[0];
                fileName.textContent = 'Selected: ' + file.name;
                fileName.style.display = 'block';
                submitBtn.disabled = false;
                submitBtn.style.background = 'linear-gradient(135deg, #667eea, #764ba2)';
            }
        });
    </script>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    confidence = None
    filename = None
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)
            prediction, confidence = predict_image(filepath)
            filename = file.filename
    return render_template_string(HTML_TEMPLATE, prediction=prediction, confidence=confidence, filename=filename)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)