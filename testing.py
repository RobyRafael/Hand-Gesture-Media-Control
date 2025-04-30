import mediapipe as mp
import cv2
import numpy as np
import math
import time
import pickle
import os
import csv
from datetime import datetime

# Function to calculate distance between two points
def calculate_distance(x1, y1, z1, x2, y2, z2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)

# Initialize mediapipe hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Initialize Camera
cap = cv2.VideoCapture(2)  # Change to appropriate camera index if needed

# Flag to track if we should exit the loop
running = True

# Gesture class mapping
gesture_classes = {
    0: "No gesture detected",
    1: "Play/Pause",
    2: "Stop",
    3: "Next Track",
    4: "Previous Track",
    5: "Volume Up",
    6: "Volume Down"
}

# Load the trained model
model_path = 'hand_gesture_model.pkl'

# Function to load the model
def load_model():
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            classifier = pickle.load(f)
        print(f"Model loaded from {model_path}")
        return classifier
    else:
        print("No pre-trained model found. Please train the model first.")
        return None

# Try to load the existing model
classifier = load_model()
if classifier is None:
    print("Cannot continue without a trained model.")
    cap.release()
    cv2.destroyAllWindows()
    exit()

# Function to extract features from hand landmarks
def extract_features(hand_landmarks):
    # Create a feature vector based on distances between fingers
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    middle_dip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    little_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    
    # Extract XYZ coordinates
    thumb_xyz = (thumb_tip.x, thumb_tip.y, thumb_tip.z)
    index_xyz = (index_tip.x, index_tip.y, index_tip.z)
    middle_xyz = (middle_tip.x, middle_tip.y, middle_tip.z)
    middle_dip_xyz = (middle_dip.x, middle_dip.y, middle_dip.z)
    ring_xyz = (ring_tip.x, ring_tip.y, ring_tip.z)
    little_xyz = (little_tip.x, little_tip.y, little_tip.z)
    
    # Calculate distances between finger points
    distance_thumb_index = calculate_distance(*thumb_xyz, *index_xyz)
    distance_thumb_middle = calculate_distance(*thumb_xyz, *middle_xyz)
    distance_thumb_ring = calculate_distance(*thumb_xyz, *ring_xyz)
    distance_thumb_little = calculate_distance(*thumb_xyz, *little_xyz)
    distance_index_middle = calculate_distance(*index_xyz, *middle_xyz)
    distance_index_middle_dip = calculate_distance(*index_xyz, *middle_dip_xyz)
    distance_ring_middle = calculate_distance(*ring_xyz, *middle_xyz)
    
    # Create feature vector
    features = [
        distance_thumb_index,
        distance_thumb_middle,
        distance_thumb_ring,
        distance_thumb_little,
        distance_index_middle,
        distance_index_middle_dip,
        distance_ring_middle
    ]
    
    return features

# Setup CSV file for data export
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_filename = f"gesture_data_{timestamp}.csv"

# Open the CSV file and write header
with open(csv_filename, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    
    # Write header row
    header = [
        "timestamp", 
        "gesture_id", 
        "gesture_name", 
        "confidence",
        "thumb_index_distance", 
        "thumb_middle_distance", 
        "thumb_ring_distance", 
        "thumb_little_distance", 
        "index_middle_distance", 
        "index_middle_dip_distance", 
        "ring_middle_distance"
    ]
    csv_writer.writerow(header)

    print(f"CSV file created: {csv_filename}")
    print("Recording gesture data. Press 'q' or ESC to stop.")
    
    # Data collection variables
    last_record_time = 0
    record_interval = 0.1  # seconds between recordings
    
    while running and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
    
        # Process the frame and get the hand landmarks
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        
        # Get frame dimensions for scaling
        h, w, c = frame.shape
    
        # Status text to display
        status_text = "No gesture detected"
        prediction = 0
        confidence = 0.0
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Extract features from hand landmarks
                features = extract_features(hand_landmarks)
                
                # Predict the gesture
                features_array = np.array([features])
                prediction = classifier.predict(features_array)[0]
                
                # Get the probabilities for each class
                probabilities = classifier.predict_proba(features_array)[0]
                confidence = probabilities[prediction]
                
                # Convert the prediction to a gesture name
                status_text = gesture_classes[prediction]
                
                # Display the prediction probabilities
                for i, prob in enumerate(probabilities):
                    if prob > 0.05:  # Only display significant probabilities
                        prob_text = f"{gesture_classes[i]}: {prob:.2f}"
                        cv2.putText(frame, prob_text, 
                                    (w - 300, 30 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)
                
                # Record data to CSV at regular intervals
                current_time = time.time()
                if current_time - last_record_time > record_interval:
                    # Create a row with timestamp, prediction, and features
                    current_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                    row = [
                        current_timestamp,
                        prediction,
                        gesture_classes[prediction],
                        confidence
                    ]
                    row.extend(features)  # Add all the feature values
                    
                    # Write to CSV
                    csv_writer.writerow(row)
                    csvfile.flush()  # Ensure data is written to disk
                    
                    last_record_time = current_time
                
                # Draw hand landmarks
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        # Display status text
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display recording status
        cv2.putText(frame, f"Recording to: {csv_filename}", 
                    (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Display frame
        cv2.imshow("Hand Gesture CSV Recorder", frame)
    
        # Check for key press
        key = cv2.waitKey(1) & 0xFF
        
        # ESC key or 'q' key to exit
        if key == 27 or key == ord('q'):
            running = False

print(f"Recording stopped. Data saved to {csv_filename}")

# Cleanup
cap.release()
cv2.destroyAllWindows()