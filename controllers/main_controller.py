from flask import Blueprint, render_template, request, jsonify
import cv2
import base64
import re
import numpy as np
from models.sign_model import SignDetectionModel

main_bp = Blueprint('main', __name__)

# Initialize model once globally
global_model = None

def get_model():
    global global_model
    if global_model is None:
        global_model = SignDetectionModel()
    return global_model

@main_bp.route('/')
def index():
    return render_template('index.html')

@main_bp.route('/process_frame', methods=['POST'])
def process_frame():
    try:
        data = request.json['image']
        # Remove the data URL prefix
        image_data = re.sub('^data:image/.+;base64,', '', data)
        image_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        model = get_model()
        processed_frame, prediction = model.predict(frame)

        # Encode back to base64
        _, buffer = cv2.imencode('.jpg', processed_frame)
        encoded_image = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            'prediction': prediction,
            'image': 'data:image/jpeg;base64,' + encoded_image
        })
    except Exception as e:
        print(f"Error processing frame: {e}")
        return jsonify({'prediction': 'Error processing', 'image': ''}), 500

@main_bp.route('/privacy')
def privacy():
    return render_template('privacy.html')

@main_bp.route('/terms')
def terms():
    terms_html = """
    <div style="font-family: sans-serif; padding: 40px; background: #0f172a; color: white; min-height: 100vh;">
        <h3 style="color: #38bdf8;">Terms of Service</h3>
        <p>1. Legal: SignFlow AI is provided for educational purposes. We do not store user data.</p>
        <p>2. Age Restrictions: This service is intended for users 13 years of age or older in compliance with COPPA. We do not knowingly collect information from children under 13.</p>
        <p>3. AI Accuracy: Sign translation is currently in alpha and should not be relied upon for critical communication.</p>
        <a href='/' style="color: #38bdf8; text-decoration: none; padding-top: 20px; display: inline-block;">Back to App</a>
    </div>
    """
    return terms_html

@main_bp.route('/contact')
def contact():
    return "<h3>Contact Support</h3><p>Support: support@signflow-ai.com</p><a href='/'>Back</a>"
