import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os

class SignDetectionModel:
    def __init__(self):
        # Path to the downloaded model
        model_path = os.path.join(os.path.dirname(__file__), 'hand_landmarker.task')
        
        # Configure Hand Landmarker
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        
        # Define connections for drawing
        self.connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),        # Index
            (0, 9), (9, 10), (10, 11), (11, 12),   # Middle
            (0, 13), (13, 14), (14, 15), (15, 16), # Ring
            (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
            (5, 9), (9, 13), (13, 17)              # Palm
        ]

    def classify_sign(self, landmarks):
        """Heuristic-based sign detection using landmark coordinates."""
        # Identification of finger tips and bases
        # Thumb: 4 (tip), 2 (base)
        # Index: 8 (tip), 6 (base)
        # Middle: 12 (tip), 10 (base)
        # Ring: 16 (tip), 14 (base)
        # Pinky: 20 (tip), 18 (base)
        
        fingers = []
        
        # Thumb (special case because it moves horizontally relative to palm)
        if landmarks[4].x < landmarks[3].x: # Assuming right hand mirror
            fingers.append(1)
        else:
            fingers.append(0)
            
        # 4 Fingers
        for tip, base in [(8, 6), (12, 10), (16, 14), (20, 18)]:
            if landmarks[tip].y < landmarks[base].y:
                fingers.append(1)
            else:
                fingers.append(0)
                
        # Classify based on finger count and pattern
        if fingers == [1, 1, 1, 1, 1]:
            return "HELLO / OPEN HAND"
        elif fingers == [1, 0, 0, 0, 0]:
            return "THUMBS UP"
        elif fingers == [0, 1, 1, 0, 0]:
            return "PEACE / VICTORY"
        elif fingers == [0, 1, 0, 0, 0]:
            return "POINTING"
        elif fingers == [0, 0, 0, 0, 0]:
            return "FIST"
        
        return "Detecting..."

    def predict(self, frame):
        # Flip frame for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Convert to RGB and Mediapipe Image format
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        
        # Detect landmarks
        detection_result = self.detector.detect(mp_image)
        
        prediction = "Waiting for signs..."
        
        if detection_result.hand_landmarks:
            h, w, c = frame.shape
            
            for landmarks in detection_result.hand_landmarks:
                # Classify the sign
                prediction = self.classify_sign(landmarks)
                
                # Convert normalized landmarks to pixel coordinates
                points = []
                for lm in landmarks:
                    px, py = int(lm.x * w), int(lm.y * h)
                    points.append((px, py))
                    
                # Draw connections
                for start_idx, end_idx in self.connections:
                    cv2.line(frame, points[start_idx], points[end_idx], 
                             (248, 189, 56), 2)
                
                # Draw landmark points
                for pt in points:
                    cv2.circle(frame, pt, 4, (248, 140, 129), -1)
        
        # Add visual feedback
        cv2.putText(frame, f"Sign: {prediction}", (20, 45), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (56, 189, 248), 3)
        
        return frame, prediction
