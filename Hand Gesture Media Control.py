import mediapipe as mp
import cv2
import numpy as np
import math
import pyautogui
import time
import pickle
import os
from sklearn.naive_bayes import GaussianNB  # For continuous features
# from sklearn.naive_bayes import MultinomialNB  # Alternative for discrete features

# Function to calculate distance between two points
def calculate_distance(x1, y1, z1, x2, y2, z2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)

# Initialize mediapipe hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Initialize Camera
cap = cv2.VideoCapture(0)

# Flag to track if we should exit the loop
running = True

# Cooldown mechanism to prevent repeated key presses
last_action_time = 0
cooldown_time = 1.0  # seconds between actions

# Dictionary to track previous gesture state
previous_gesture = "No gesture detected"

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

# Initialize the Naive Bayes classifier
classifier = GaussianNB()

# Model file path
model_path = 'hand_gesture_model.pkl'

# Variables for collecting training data
collecting_data = False
training_data = []
training_labels = []
current_collection_class = 0

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

# Function to train the Naive Bayes model
def train_model():
    global classifier, training_data, training_labels
    
    if len(training_data) == 0 or len(training_labels) == 0:
        print("No training data available.")
        return False
    
    # Convert to numpy arrays
    X = np.array(training_data)
    y = np.array(training_labels)
    
    # Train the model
    classifier.fit(X, y)
    
    # Save the model
    with open(model_path, 'wb') as f:
        pickle.dump(classifier, f)
    
    print(f"Model trained with {len(training_data)} samples and saved to {model_path}")
    return True

# Function to load the model if it exists
def load_model():
    global classifier
    
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            classifier = pickle.load(f)
        print(f"Model loaded from {model_path}")
        return True
    else:
        print("No pre-trained model found.")
        return False

# Try to load the existing model
model_loaded = load_model()

# Print instructions before starting the program
print("Hand Gesture Media Controls using Naive Bayes:")
print("- Thumb + Index finger: Play/Pause")
print("- Thumb + Middle finger: Stop")
print("- Thumb + Ring finger: Next Track")
print("- Thumb + Little finger: Previous Track")
print("- Index + Middle finger: Volume Up")
print("- Ring + Middle finger: Volume Down")
print("\nTraining Mode Controls:")
print("- Press 't' to toggle training data collection")
print("- Press '0-6' to select the gesture class to collect")
print("- Press 'm' to train the model")
print("- ESC key: Exit program")

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
    
    # Display the training status
    if collecting_data:
        cv2.putText(frame, f"Collecting data for: {gesture_classes[current_collection_class]}", 
                    (10, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"Samples: {training_labels.count(current_collection_class)}", 
                    (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract features from hand landmarks
            features = extract_features(hand_landmarks)
            
            # If collecting data, add to training set
            if collecting_data:
                training_data.append(features)
                training_labels.append(current_collection_class)
                # Limit collection rate
                time.sleep(0.1)
            
            # If model is loaded, predict the gesture
            if model_loaded:
                # Reshape features for prediction
                features_array = np.array([features])
                
                # Predict the gesture
                prediction = classifier.predict(features_array)[0]
                
                # Get the probabilities for each class
                probabilities = classifier.predict_proba(features_array)[0]
                
                # Convert the prediction to a gesture name
                status_text = gesture_classes[prediction]
                
                # Display the prediction probabilities
                for i, prob in enumerate(probabilities):
                    if prob > 0.05:  # Only display significant probabilities
                        prob_text = f"{gesture_classes[i]}: {prob:.2f}"
                        cv2.putText(frame, prob_text, 
                                    (w - 300, 30 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)
                
                # Execute actions based on the predicted gesture
                current_time = time.time()
                
                # Volume control variables
                volume_adjustment_interval = 0.3  # Time between volume adjustments when holding gesture
                
                if prediction == 1 and previous_gesture != status_text:  # Play/Pause
                    if current_time - last_action_time > cooldown_time:
                        pyautogui.press('playpause')
                        last_action_time = current_time
                elif prediction == 2 and previous_gesture != status_text:  # Stop
                    if current_time - last_action_time > cooldown_time:
                        pyautogui.press('stop')
                        last_action_time = current_time
                elif prediction == 3 and previous_gesture != status_text:  # Next Track
                    if current_time - last_action_time > cooldown_time:
                        pyautogui.press('nexttrack')
                        last_action_time = current_time
                elif prediction == 4 and previous_gesture != status_text:  # Previous Track
                    if current_time - last_action_time > cooldown_time:
                        pyautogui.press('prevtrack')
                        last_action_time = current_time
                elif prediction == 5:  # Volume Up
                    if current_time - last_action_time > volume_adjustment_interval:
                        pyautogui.press('volumeup')
                        last_action_time = current_time
                elif prediction == 6:  # Volume Down
                    if current_time - last_action_time > volume_adjustment_interval:
                        pyautogui.press('volumedown')
                        last_action_time = current_time
                
                # Update previous gesture
                previous_gesture = status_text
            
            # Display extracted features (for debugging)
            for i, feature in enumerate(features):
                cv2.putText(frame, f"F{i}: {feature:.3f}", 
                            (10, 60 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Draw hand landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    # Display status text
    cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Display frame
    cv2.imshow("Hand Gesture Media Control (Naive Bayes)", frame)

    # Check for key press
    key = cv2.waitKey(1) & 0xFF
    
    # ESC key to exit
    if key == 27:
        running = False
    
    # 't' key to toggle training data collection
    elif key == ord('t'):
        collecting_data = not collecting_data
        print(f"Data collection {'started' if collecting_data else 'stopped'}")
    
    # 'm' key to train the model
    elif key == ord('m'):
        model_loaded = train_model()
    
    # Number keys to select the gesture class for training
    elif key >= ord('0') and key <= ord('6'):
        current_collection_class = key - ord('0')
        print(f"Selected class: {current_collection_class} - {gesture_classes[current_collection_class]}")
    
    # 'q' key as an alternative exit option
    elif key == ord('q'):
        running = False

# Cleanup
cap.release()
cv2.destroyAllWindows()
