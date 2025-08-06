from ultralytics import YOLO
import cv2 as cv
import numpy as np
import keyboard  # For sending keystrokes
import torch
import time

# Check for CUDA availability and set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
if device == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Movement tracking variables
previous_nose_x = None
previous_knee_y = {}  # Track both knees
w_pressed = False
last_movement_time = 0
movement_threshold = 15  # Pixels for head movement detection
knee_movement_threshold = 10  # Pixels for knee movement detection

# Load model with GPU acceleration
pose_model = YOLO('yolo11n-pose.pt')
pose_model.to(device)  # Move model to GPU if available
print(f"Model loaded on: {pose_model.device}")

def detect_simple_gestures(results):
    """Detect simple head turns, running motion, and hands to right side"""
    global previous_nose_x, previous_knee_y, w_pressed, last_movement_time
    
    if not results or not results[0].keypoints:
        return False, False, False, False
    
    keypoints = results[0].keypoints.data[0]  # First person detected
    min_confidence = 0.5
    
    # Extract keypoints
    nose = keypoints[0]
    left_shoulder = keypoints[5]
    right_shoulder = keypoints[6]
    left_wrist = keypoints[9]
    right_wrist = keypoints[10]
    left_knee = keypoints[13]
    right_knee = keypoints[14]
    
    # Initialize return values
    head_left = False
    head_right = False
    running = False
    hands_right = False
    
    # 1. Head turning detection (left/right arrow keys)
    if nose[2] > min_confidence:
        if previous_nose_x is not None:
            nose_movement = nose[0] - previous_nose_x
            if abs(nose_movement) > movement_threshold:
                if nose_movement < -movement_threshold:  # Head turned left
                    head_left = True
                elif nose_movement > movement_threshold:  # Head turned right
                    head_right = True
        previous_nose_x = nose[0]
    
    # 2. Running detection (knees going up and down)
    current_time = time.time()
    if (left_knee[2] > min_confidence and right_knee[2] > min_confidence):
        current_knee_avg_y = (left_knee[1] + right_knee[1]) / 2
        
        if 'avg_y' in previous_knee_y:
            knee_movement = abs(current_knee_avg_y - previous_knee_y['avg_y'])
            if knee_movement > knee_movement_threshold:
                last_movement_time = current_time
                running = True
            elif current_time - last_movement_time > 1.0:  # Stop running after 1 second of no movement
                running = False
        
        previous_knee_y['avg_y'] = current_knee_avg_y
    
    # 3. Hands to right side detection (key 1)
    if (left_shoulder[2] > min_confidence and right_shoulder[2] > min_confidence and 
        left_wrist[2] > min_confidence and right_wrist[2] > min_confidence):
        
        body_center_x = (left_shoulder[0] + right_shoulder[0]) / 2
        shoulder_width = abs(right_shoulder[0] - left_shoulder[0])
        
        # Check if both hands are to the right side of the body
        right_side_threshold = body_center_x + (shoulder_width * 0.3)
        if left_wrist[0] > right_side_threshold and right_wrist[0] > right_side_threshold:
            hands_right = True
    
    return head_left, head_right, running, hands_right

# Try to connect camera
cap = None
try:
    cap = cv.VideoCapture(0)
    if not cap.isOpened() or cap.read()[0] == False:
        cap.release()
        cap = None
        print("Camera not available")
except:
    print("Camera not available")
    cap = None

# Exit if no camera found
if not cap:
    print("No camera available! Exiting...")
    exit()

print("Simple Gesture Control")
print("=" * 40)
print("� Gestures:")
print("� Turn head LEFT → Left Arrow")
print("👉 Turn head RIGHT → Right Arrow") 
print("🏃 Knees up/down (running) → Hold W")
print("� Both hands to RIGHT side → Key 1")
print("Press 'q' to quit")
print()

print("Starting camera feed...")
while True:
    if cap:
        ret, frame = cap.read()
        if ret:
            # Resize frame
            frame = cv.resize(frame, (960, 720))
            
            # Run pose detection with GPU acceleration
            results = pose_model(frame, device=device, verbose=False)
            annotated = results[0].plot() if results else frame
            
            # Detect simple gestures
            head_left, head_right, running, hands_right = detect_simple_gestures(results)
            
            # Handle gestures
            status_text = ""
            
            # Head turning
            if head_left:
                keyboard.press_and_release('left')
                status_text += "👈 LEFT ARROW "
            elif head_right:
                keyboard.press_and_release('right')
                status_text += "👉 RIGHT ARROW "
            
            # Running motion (hold W)
            if running:
                if not w_pressed:
                    keyboard.press('w')
                    w_pressed = True
                status_text += "� RUNNING (W) "
            else:
                if w_pressed:
                    keyboard.release('w')
                    w_pressed = False
            
            # Hands to right
            if hands_right:
                keyboard.press_and_release('1')
                status_text += "� HANDS RIGHT (1) "
            
            # Add title and status
            cv.putText(annotated, "Simple Gesture Control - Press 'q' to quit", 
                      (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if status_text:
                cv.putText(annotated, status_text, (10, 70), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            cv.imshow('Simple Gesture Control', annotated)
    
    key = cv.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# Cleanup
if w_pressed:
    keyboard.release('w')
if cap:
    cap.release()
cv.destroyAllWindows()
