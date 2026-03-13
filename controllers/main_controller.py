from flask import Blueprint, render_template, Response
import cv2
import threading
import time
from models.sign_model import SignDetectionModel

class Camera:
    def __init__(self):
        self.camera = cv2.VideoCapture(0)
        self.model = SignDetectionModel()
        self.output_frame = None
        self.prediction = "Waiting for signs..."
        self.lock = threading.Lock()
        self.stopped = False
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        while not self.stopped:
            success, frame = self.camera.read()
            if success:
                # Process frame
                processed_frame, prediction = self.model.predict(frame)
                with self.lock:
                    self.output_frame = processed_frame.copy()
                    self.prediction = prediction
            else:
                time.sleep(0.1)

    def get_frame(self):
        with self.lock:
            if self.output_frame is None:
                return None, "Camera starting..."
            return self.output_frame.copy(), self.prediction

    def stop(self):
        self.stopped = True
        self.camera.release()

main_bp = Blueprint('main', __name__)
global_camera = None

def get_camera():
    global global_camera
    if global_camera is None:
        global_camera = Camera()
    return global_camera

@main_bp.route('/')
def index():
    return render_template('index.html')

def gen_frames():
    cam = get_camera()
    while True:
        frame, pred = cam.get_frame()
        if frame is not None:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.03) # ~30 FPS

@main_bp.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@main_bp.route('/status')
def status():
    cam = get_camera()
    _, pred = cam.get_frame()
    return {"prediction": pred}

@main_bp.route('/privacy')
def privacy():
    return render_template('privacy.html')

@main_bp.route('/terms')
def terms():
    return "<h3>Terms of Service</h3><p>Legal: SignFlow AI is provided for educational purposes. We do not store user data.</p><a href='/'>Back</a>"

@main_bp.route('/contact')
def contact():
    return "<h3>Contact Support</h3><p>Support: support@signflow-ai.com</p><a href='/'>Back</a>"
