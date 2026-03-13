from flask import Blueprint, render_template, Response
import cv2
from models.sign_model import SignDetectionModel

main_bp = Blueprint('main', __name__)
model = SignDetectionModel()

@main_bp.route('/')
def index():
    return render_template('index.html')

current_prediction = "Waiting for signs..."

def gen_frames():
    global current_prediction
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Process frame for sign detection
            processed_frame, prediction = model.predict(frame)
            current_prediction = prediction
            
            # Encode frame
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@main_bp.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@main_bp.route('/status')
def status():
    return {"prediction": current_prediction}
