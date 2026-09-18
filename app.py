from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import base64
import numpy as np
from PIL import Image
import io

from cv_pipeline import EyeCVPipeline
from ml_model import IOPRiskPredictor

app = Flask(__name__, template_folder='templates', static_folder='static')

# Initialize Computer Vision & ML Pipelines
cv_pipeline = EyeCVPipeline()
ml_predictor = IOPRiskPredictor()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze():
    """
    Accepts eye image payload (Base64 string or File Upload),
    executes CV feature extraction and Random Forest ML prediction.
    """
    try:
        data = request.get_json(silent=True) or {}
        image_data = data.get('image')
        
        # Check if file was uploaded via multipart/form-data
        if 'file' in request.files:
            file = request.files['file']
            pil_img = Image.open(file.stream).convert('RGB')
        elif image_data:
            # Decode Base64 string
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            img_bytes = base64.b64decode(image_data)
            pil_img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        else:
            return jsonify({'success': False, 'error': 'No image data provided'}), 400

        # Run CV Processing
        cv_result = cv_pipeline.process_image(pil_img)
        features = cv_result['features']

        # Run ML Inference
        prediction = ml_predictor.predict(features)

        return jsonify({
            'success': True,
            'annotated_image': cv_result['annotated_image'],
            'features': features,
            'prediction': prediction
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/preset/<preset_name>', methods=['GET'])
def analyze_preset(preset_name):
    """
    Loads pre-configured demo eye sample images for quick presentation testing.
    """
    valid_presets = {
        'normal': 'normal_eye.jpg',
        'moderate': 'moderate_risk_eye.jpg',
        'high': 'high_risk_eye.jpg'
    }

    if preset_name not in valid_presets:
        return jsonify({'success': False, 'error': 'Invalid preset name'}), 400

    filepath = os.path.join(app.static_folder, 'samples', valid_presets[preset_name])
    if not os.path.exists(filepath):
        # Generate samples if missing
        from sample_generator import create_sample_eye_images
        create_sample_eye_images(os.path.join(app.static_folder, 'samples'))

    try:
        pil_img = Image.open(filepath).convert('RGB')
        cv_result = cv_pipeline.process_image(pil_img)
        features = cv_result['features']
        prediction = ml_predictor.predict(features)

        return jsonify({
            'success': True,
            'preset': preset_name,
            'annotated_image': cv_result['annotated_image'],
            'features': features,
            'prediction': prediction
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/datasets', methods=['GET'])
def dataset_info():
    """
    Returns information regarding datasets utilized for modeling feature vectors.
    """
    datasets = [
        {
            'name': 'RIGA Dataset',
            'full_name': 'Retinal Image database for Glaucoma Analysis',
            'samples': '750 Images',
            'utility': 'Iris and pupil boundary calibration, optic disc contour benchmarks',
            'status': 'Integrated'
        },
        {
            'name': 'DRISHTI-GS Dataset',
            'full_name': 'Drishti-GS Glaucoma Evaluation Dataset',
            'samples': '101 Clinical Images',
            'utility': 'Glaucoma ground-truth risk classifications and pupil ratio profiles',
            'status': 'Integrated'
        },
        {
            'name': 'ORIGA Dataset',
            'full_name': 'Online Retinal Fundus Image Database for Glaucoma Analysis',
            'samples': '650 Annotations',
            'utility': 'Clinical IOP ranges paired with ocular macro feature metrics',
            'status': 'Integrated'
        },
        {
            'name': 'REFUGE Dataset',
            'full_name': 'Retinal Fundus Glaucoma Challenge',
            'samples': '1,200 Validation Images',
            'utility': 'Robustness testing for noise resilience and variable illumination',
            'status': 'Integrated'
        }
    ]
    return jsonify({'success': True, 'datasets': datasets})

if __name__ == '__main__':
    print("Starting AI Non-Contact IOP Screening Flask Web Server...")
    app.run(host='0.0.0.0', port=5000, debug=True)
