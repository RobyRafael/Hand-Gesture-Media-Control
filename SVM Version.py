import mediapipe as mp
import cv2
import numpy as np
import math
import pyautogui
import time
import pickle
import os
import csv
from sklearn.svm import SVC
import traceback  # Added for error tracking
from datetime import datetime

# Create screenshots directory if it doesn't exist
screenshots_dir = 'gesture_screenshots'
if not os.path.exists(screenshots_dir):
    os.makedirs(screenshots_dir)
    print(f"Created directory: {screenshots_dir}")

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

# Initialize the SVM classifier with more specific parameters
classifier = SVC(kernel='rbf', probability=True, C=1.0, gamma='scale')

# File paths
model_path = 'hand_gesture_model_svm.pkl'
csv_path = 'hand_gesture_data_svm.csv'

# Variables for collecting training data
collecting_data = False
training_data = []
training_labels = []
current_collection_class = 0
sample_cooldown = 0  # To control screenshot frequency

# Selected key landmarks to track
LANDMARK_INDICES = [
    mp_hands.HandLandmark.THUMB_TIP,
    mp_hands.HandLandmark.INDEX_FINGER_TIP,
    mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
    mp_hands.HandLandmark.RING_FINGER_TIP,
    mp_hands.HandLandmark.PINKY_TIP,
    mp_hands.HandLandmark.WRIST
]

# Function to save a screenshot with gesture information
def save_screenshot(frame, class_id, sample_number):
    # Create class-specific directory if it doesn't exist
    class_dir = os.path.join(screenshots_dir, f"class_{class_id}_{gesture_classes[class_id]}")
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)
    
    # Generate a filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(class_dir, f"sample_{sample_number}_{timestamp}.jpg")
    
    # Add class label to the frame
    labeled_frame = frame.copy()
    cv2.putText(labeled_frame, f"Class: {class_id} - {gesture_classes[class_id]}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Save the screenshot
    cv2.imwrite(filename, labeled_frame)
    print(f"Saved screenshot: {filename}")

# Function to extract raw features from hand landmarks
def extract_features(hand_landmarks):
    # Create a feature vector based on raw x,y,z coordinates of key landmarks
    features = []
    
    for landmark_idx in LANDMARK_INDICES:
        landmark = hand_landmarks.landmark[landmark_idx]
        features.extend([landmark.x, landmark.y, landmark.z])
    
    # Validate feature values (check for NaN or infinity)
    for i, val in enumerate(features):
        if not np.isfinite(val):  # Replace NaN or Inf with 0
            features[i] = 0.0
            print(f"Warning: Non-finite value detected and replaced")
    
    return features

# Function to save training data to CSV
def save_data_to_csv():
    global training_data, training_labels
    
    if len(training_data) == 0 or len(training_labels) == 0:
        print("No training data available to save to CSV.")
        return False
    
    # Check for consistent data dimensions
    expected_length = len(training_data[0])
    valid_data = []
    valid_labels = []
    
    for features, label in zip(training_data, training_labels):
        if len(features) == expected_length:
            valid_data.append(features)
            valid_labels.append(label)
        else:
            print(f"Warning: Skipping inconsistent feature vector (length {len(features)} vs expected {expected_length})")
    
    # Prepare header (feature names and label)
    header = []
    for idx, landmark_name in enumerate([
        "THUMB_TIP", "INDEX_FINGER_TIP", "MIDDLE_FINGER_TIP", 
        "RING_FINGER_TIP", "PINKY_TIP", "WRIST"
    ]):
        for coord in ["x", "y", "z"]:
            header.append(f"{landmark_name}_{coord}")
    header.append('gesture_class')
    
    # Prepare rows (combine features and label for each sample)
    rows = []
    for features, label in zip(valid_data, valid_labels):
        row = features + [label]  # Combine features and label
        rows.append(row)
    
    try:
        # Write to CSV file
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)
        
        print(f"Training data saved to {csv_path} ({len(rows)} samples)")
        return True
    except Exception as e:
        print(f"Error saving data to CSV: {e}")
        return False

# Function to load training data from CSV
def load_data_from_csv():
    global training_data, training_labels
    
    if not os.path.exists(csv_path):
        print("No CSV training data file found.")
        return False
    
    training_data = []
    training_labels = []
    
    try:
        # Read from CSV file
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)  # Skip header
            
            for row in reader:
                if len(row) >= 19:  # 18 features (6 landmarks x 3 coordinates) + 1 label
                    try:
                        features = [float(x) for x in row[:18]]  # First 18 columns are features
                        label = int(row[18])  # Last column is the class label
                        
                        # Validate feature values
                        valid_features = True
                        for val in features:
                            if not np.isfinite(val):
                                valid_features = False
                                break
                        
                        if valid_features:
                            training_data.append(features)
                            training_labels.append(label)
                    except ValueError:
                        print(f"Warning: Skipping invalid row in CSV")
        
        print(f"Loaded {len(training_data)} samples from {csv_path}")
        return len(training_data) > 0
    except Exception as e:
        print(f"Error loading data from CSV: {e}")
        return False

# Function to train the SVM model
def train_model():
    global classifier, training_data, training_labels
    
    if len(training_data) == 0 or len(training_labels) == 0:
        print("No training data available.")
        return False
    
    try:
        # Check if we have enough samples of each class
        class_counts = {}
        for label in training_labels:
            class_counts[label] = class_counts.get(label, 0) + 1
        
        # Print class distribution
        print("Class distribution in training data:")
        for cls, count in class_counts.items():
            print(f"  Class {cls} ({gesture_classes.get(cls, 'Unknown')}): {count} samples")
        
        # Make sure we have at least 2 classes and each class has samples
        if len(class_counts) < 2:
            print("Error: Need at least 2 different classes to train the model.")
            return False
        
        # Check if there are any classes with too few samples (less than 3)
        low_sample_classes = [cls for cls, count in class_counts.items() if count < 3]
        if low_sample_classes:
            print(f"Warning: Classes {low_sample_classes} have fewer than 3 samples. Training may be unreliable.")
        
        # Convert to numpy arrays
        X = np.array(training_data)
        y = np.array(training_labels)
        
        print(f"Training SVM model with {len(training_data)} samples...")
        # Train the model
        classifier.fit(X, y)
        
        # Save the model to pkl file
        with open(model_path, 'wb') as f:
            pickle.dump(classifier, f)
        
        # Save training data to CSV
        save_data_to_csv()
        
        print(f"SVM model trained and saved to {model_path}")
        return True
    except Exception as e:
        print(f"Error training model: {str(e)}")
        print(traceback.format_exc())  # Print detailed error information
        return False

# Function to load the model if it exists
def load_model():
    global classifier, training_data, training_labels
    
    # Try to load training data from CSV first
    loaded_csv = load_data_from_csv()
    
    # Then try to load the model
    if os.path.exists(model_path):
        try:
            with open(model_path, 'rb') as f:
                classifier = pickle.load(f)
            print(f"SVM model loaded from {model_path}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            
            # If we have training data, attempt to retrain
            if loaded_csv and len(training_data) > 0:
                print("Attempting to retrain model with loaded data...")
                return train_model()
            return False
    else:
        print("No pre-trained SVM model found.")
        return False

# Try to load the existing model and training data
model_loaded = load_model()

# Print instructions before starting the program
print("Hand Gesture Media Controls using SVM:")
print("- Thumb + Index finger: Play/Pause")
print("- Thumb + Middle finger: Stop")
print("- Thumb + Ring finger: Next Track")
print("- Thumb + Little finger: Previous Track")
print("- Index + Middle finger: Volume Up")
print("- Ring + Middle finger: Volume Down")
print("\nTraining Mode Controls:")
print("- Press 't' to toggle training data collection")
print("- Press '0-6' to select the gesture class to collect")
print("- Press 'm' to train the model (saves to both PKL and CSV)")
print("- Press 's' to save training data to CSV without retraining")
print("- Press 'c' to clear all collected training data")
print("- ESC key: Exit program")
print("\nScreenshots of samples will be saved to the 'gesture_screenshots' folder")

try:
    while running and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to get frame from camera")
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
            cv2.putText(frame, f"Collecting class: {current_collection_class} - {gesture_classes[current_collection_class]}", 
                        (10, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Count samples for current class
            current_class_count = sum(1 for label in training_labels if label == current_collection_class)
            cv2.putText(frame, f"Class samples: {current_class_count}", 
                        (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Show total samples collected
        cv2.putText(frame, f"Total samples: {len(training_labels)}", 
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                try:
                    # Extract features from hand landmarks
                    features = extract_features(hand_landmarks)
                    
                    # If collecting data, add to training set
                    if collecting_data:
                        # Add sample to training data
                        training_data.append(features)
                        training_labels.append(current_collection_class)
                        
                        # Calculate current class count
                        current_class_count = sum(1 for label in training_labels if label == current_collection_class)
                        
                        # Take a screenshot if cooldown has passed
                        if sample_cooldown <= 0:
                            save_screenshot(frame, current_collection_class, current_class_count)
                            sample_cooldown = 10  # Set cooldown for 10 frames
                        else:
                            sample_cooldown -= 1
                        
                        # Limit collection rate
                        time.sleep(0.1)
                    
                    # If model is loaded, predict the gesture
                    if model_loaded:
                        try:
                            # Reshape features for prediction
                            features_array = np.array([features])
                            
                            # Predict the gesture
                            prediction = classifier.predict(features_array)[0]
                            
                            # Get the probabilities for each class
                            probabilities = classifier.predict_proba(features_array)[0]
                            
                            # Convert the prediction to a gesture name
                            status_text = gesture_classes.get(prediction, "Unknown gesture")
                            
                            # Display the prediction probabilities
                            for i, prob in enumerate(probabilities):
                                if prob > 0.05:  # Only display significant probabilities
                                    prob_text = f"{gesture_classes.get(i, 'Unknown')}: {prob:.2f}"
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
                        except Exception as e:
                            print(f"Error during prediction: {e}")
                    
                    # Display extracted features (for debugging) - just show first few
                    for i in range(min(6, len(features))):
                        cv2.putText(frame, f"F{i}: {features[i]:.3f}", 
                                    (10, 60 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    # Draw hand landmarks
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                except Exception as e:
                    print(f"Error processing hand landmarks: {e}")
        
        # Display status text
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display frame
        cv2.imshow("Hand Gesture Media Control (SVM)", frame)

        # Check for key press
        key = cv2.waitKey(1) & 0xFF
        
        # ESC key to exit
        if key == 27:
            running = False
        
        # 't' key to toggle training data collection
        elif key == ord('t'):
            collecting_data = not collecting_data
            print(f"Data collection {'started' if collecting_data else 'stopped'}")
            # Reset sample cooldown when toggling
            sample_cooldown = 0
        
        # 'm' key to train the model and save to both PKL and CSV
        elif key == ord('m'):
            print("Training model...")
            model_loaded = train_model()
        
        # 's' key to save data to CSV without retraining
        elif key == ord('s'):
            print("Saving data to CSV...")
            save_data_to_csv()
        
        # 'c' key to clear all collected training data
        elif key == ord('c'):
            print("Clearing all training data...")
            training_data = []
            training_labels = []
            print("Training data cleared")
        
        # Number keys to select the gesture class for training
        elif key >= ord('0') and key <= ord('6'):
            current_collection_class = key - ord('0')
            print(f"Selected class: {current_collection_class} - {gesture_classes[current_collection_class]}")
        
        # 'q' key as an alternative exit option
        elif key == ord('q'):
            running = False
except Exception as e:
    print(f"Unexpected error: {e}")
    print(traceback.format_exc())

# Before exiting, save the training data to CSV if it's not empty
if len(training_data) > 0:
    print("Saving data before exit...")
    save_data_to_csv()

# Cleanup
cap.release()
cv2.destroyAllWindows()