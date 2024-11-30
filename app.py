import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import pandas as pd

# Load your model
model = load_model(r'./eye_disease_detection_model.h5')
  # Update with your model file

# Initialize Flask app
app = Flask(__name__)

# Route for homepage
@app.route('/')
def index():
    return render_template('form.html')

# Route for file upload
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})

    try:
        # Preprocess the image
        img = Image.open(file).resize((224, 224))  # Update dimensions based on your model
        img_array = np.array(img) / 255.0  # Normalize if your model requires it
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Make prediction
        predictions = model.predict(img_array)
        disease = np.argmax(predictions, axis=1)[0]  # Update based on your model's output

        # Map prediction to disease label
        disease_labels = ['The patient is suffering from Bulging_Eyes', 'The patient is suffering from Central_Serous_Chorioretinopathy', 'The patient is suffering from Crossed_Eyes', 'The patient is suffering from Disc_Edema','The patient is suffering from Macular_Scar', 'The patient is suffering from Myopia','The patient is suffering from Pterygium','The patient is suffering from Retinal_Detachment', 'The patient is suffering from Retinitis_Pigmentosa','The patient is suffering from Uveitis','The patient is suffering from cataract_eye','The patient is suffering from diabetic_retinopathy_eye', 'The patient is suffering from glaucoma_eye','The patient have no problem normal_eye']  # Update with your labels
        result = disease_labels[disease]

        return jsonify({'disease': result})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
