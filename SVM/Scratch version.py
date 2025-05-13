import cv2
import numpy as np
import pyautogui
import time
import os
import pickle
import math
from datetime import datetime

# Directory setup
script_dir = os.path.dirname(os.path.abspath(__file__))
screenshots_dir = os.path.join(script_dir, 'custom_gesture_screenshots')
if not os.path.exists(screenshots_dir):
    os.makedirs(screenshots_dir)
    print(f"Created directory: {screenshots_dir}")

# Model paths
model_path = os.path.join(script_dir, 'custom_hand_gesture_model.pkl')
training_data_path = os.path.join(script_dir, 'custom_training_data.pkl')

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

# Initialize webcam
cap = cv2.VideoCapture(1)  # Try 0 if this doesn't work
running = True

# Action cooldown and tracking
last_action_time = 0
cooldown_time = 1.0  # seconds between actions
previous_gesture = "No gesture detected"

# Training variables
collecting_data = False
training_data = []
training_labels = []
current_collection_class = 0
sample_cooldown = 0

# ============= CUSTOM HAND DETECTION AND FEATURE EXTRACTION =============
def detect_skin(frame):
    """Detect skin using color segmentation in YCrCb space"""
    # Convert to YCrCb
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    
    # Define skin color range for segmentation
    lower_skin = np.array([0, 135, 85], dtype=np.uint8)
    upper_skin = np.array([255, 180, 135], dtype=np.uint8)
    
    # Create mask for skin color
    mask = cv2.inRange(ycrcb, lower_skin, upper_skin)
    
    # Apply morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.erode(mask, kernel, iterations=2)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    
    return mask

def find_contours(mask):
    """Find contours in the mask"""
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the largest contour (assuming it's the hand)
    if contours:
        return max(contours, key=cv2.contourArea)
    return None

def find_convex_hull_and_defects(contour):
    """Find convex hull and convexity defects of a contour"""
    if contour is None or len(contour) < 5:
        return None, None, None
    
    # Find convex hull
    hull = cv2.convexHull(contour, returnPoints=False)
    
    # Find convexity defects
    try:
        defects = cv2.convexityDefects(contour, hull)
    except:
        return contour, hull, None
    
    return contour, hull, defects

def extract_features(contour, defects, frame_shape):
    """Extract hand features from contour and defects"""
    if contour is None or defects is None:
        return None
    
    # Initialize features
    features = []
    height, width, _ = frame_shape
    
    # 1. Extract contour area and perimeter
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    
    # Normalize by frame dimensions
    normalized_area = area / (height * width)
    features.append(normalized_area)
    
    # 2. Calculate convexity (ratio of contour area to convex hull area)
    hull = cv2.convexHull(contour, returnPoints=True)
    hull_area = cv2.contourArea(hull)
    if hull_area > 0:
        convexity = area / hull_area
        features.append(convexity)
    else:
        features.append(0)
    
    # 3. Find convexity defects (space between fingers)
    finger_count = 0
    defect_distances = []
    
    # Find the centroid of the contour
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx, cy = 0, 0
    
    # Process defects to find finger-like structures
    if defects is not None and len(defects) > 0:
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(contour[s][0])
            end = tuple(contour[e][0])
            far = tuple(contour[f][0])
            
            # Calculate distances between points
            dist_to_center = math.sqrt((far[0] - cx)**2 + (far[1] - cy)**2)
            normalized_dist = dist_to_center / math.sqrt(width**2 + height**2)
            defect_distances.append(normalized_dist)
            
            # Count potential fingers using angle
            a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
            b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
            c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
            
            # Apply cosine law to find angle
            if a*b > 0:
                angle = math.acos((b**2 + c**2 - a**2) / (2*b*c))
                
                # If angle is less than 90 degrees, it might be space between fingers
                if angle <= math.pi/2:
                    finger_count += 1
    
    # 4. Add finger count to features
    features.append(finger_count)
    
    # 5. Add defect distances (up to 5)
    defect_distances.sort(reverse=True)
    for i in range(min(5, len(defect_distances))):
        features.append(defect_distances[i])
    
    # Pad if less than 5 defects
    while len(features) < 8:  # 3 base features + 5 defect distances
        features.append(0)
    
    # 6. Add aspect ratio of bounding rect
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / h if h > 0 else 0
    features.append(aspect_ratio)
    
    # 7. Add hand orientation (angle of the ellipse)
    if len(contour) >= 5:
        (x, y), (MA, ma), angle = cv2.fitEllipse(contour)
        features.append(angle / 180.0)  # Normalize to 0-1
    else:
        features.append(0)
    
    return features

# ============= CUSTOM CLASSIFIER IMPLEMENTATION =============
class SimpleClassifier:
    """A simplified classifier to replace SVM"""
    def __init__(self):
        self.exemplars = {}  # Class -> list of feature vectors
        self.trained = False
    
    def fit(self, X, y):
        """Train the classifier by storing exemplars for each class"""
        # Reset exemplars
        self.exemplars = {}
        
        # Store feature vectors by class
        for features, label in zip(X, y):
            if label not in self.exemplars:
                self.exemplars[label] = []
            self.exemplars[label].append(features)
        
        self.trained = True
        return self
    
    def predict(self, X):
        """Predict class using nearest neighbor approach"""
        if not self.trained:
            return [0] * len(X)
        
        predictions = []
        for features in X:
            predictions.append(self._predict_single(features))
        return np.array(predictions)
    
    def _predict_single(self, features):
        """Predict class for a single feature vector"""
        if not self.trained or not self.exemplars:
            return 0
        
        best_dist = float('inf')
        best_class = 0
        
        # Check each class
        for class_label, examples in self.exemplars.items():
            for example in examples:
                # Calculate Euclidean distance
                dist = np.sqrt(np.sum((np.array(features) - np.array(example))**2))
                
                if dist < best_dist:
                    best_dist = dist
                    best_class = class_label
        
        return best_class
    
    def predict_proba(self, X):
        """Return confidence scores for each class"""
        if not self.trained:
            return [[1.0 if i == 0 else 0.0 for i in range(7)]] * len(X)
        
        probas = []
        for features in X:
            probas.append(self._predict_proba_single(features))
        return np.array(probas)
    
    def _predict_proba_single(self, features):
        """Calculate confidence for each class for a single feature vector"""
        if not self.trained or not self.exemplars:
            return [1.0 if i == 0 else 0.0 for i in range(7)]
        
        # Calculate distance to each example
        class_distances = {}
        for class_label, examples in self.exemplars.items():
            distances = []
            for example in examples:
                dist = np.sqrt(np.sum((np.array(features) - np.array(example))**2))
                distances.append(dist)
            
            # Use the minimum distance for this class
            if distances:
                class_distances[class_label] = min(distances)
        
        # Convert distances to probabilities (closer = higher probability)
        total_classes = len(gesture_classes)
        probas = [0.0] * total_classes
        
        if class_distances:
            # Normalize distances using softmax-like approach
            distances = np.array(list(class_distances.values()))
            distances = np.exp(-distances)  # Convert to similarity (higher is better)
            total = np.sum(distances)
            
            if total > 0:
                # Set probability for each class
                for i, (class_label, dist) in enumerate(class_distances.items()):
                    similarity = np.exp(-dist)
                    probas[class_label] = similarity / total
        else:
            probas[0] = 1.0  # Default to "No gesture" if no distances
        
        return probas

# Initialize classifier
classifier = SimpleClassifier()

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

# Functions to save and load training data
def save_training_data():
    global training_data, training_labels
    
    if len(training_data) == 0 or len(training_labels) == 0:
        print("No training data available to save.")
        return False
    
    try:
        # Save data using pickle
        with open(training_data_path, 'wb') as f:
            pickle.dump((training_data, training_labels), f)
        
        print(f"Training data saved to {training_data_path} ({len(training_data)} samples)")
        return True
    except Exception as e:
        print(f"Error saving training data: {e}")
        return False

def load_training_data():
    global training_data, training_labels
    
    if not os.path.exists(training_data_path):
        print("No training data file found.")
        return False
    
    try:
        # Load data using pickle
        with open(training_data_path, 'rb') as f:
            training_data, training_labels = pickle.load(f)
        
        print(f"Loaded {len(training_data)} samples from {training_data_path}")
        return len(training_data) > 0
    except Exception as e:
        print(f"Error loading training data: {e}")
        return False

# Function to train the model
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
        
        # Make sure we have at least 2 classes
        if len(class_counts) < 2:
            print("Error: Need at least 2 different classes to train the model.")
            return False
        
        # Convert to numpy arrays
        X = np.array(training_data)
        y = np.array(training_labels)
        
        print(f"Training custom classifier with {len(training_data)} samples...")
        # Train the model
        classifier.fit(X, y)
        
        # Save the model to pickle file
        with open(model_path, 'wb') as f:
            pickle.dump(classifier, f)
        
        # Save training data
        save_training_data()
        
        print(f"Custom classifier trained and saved to {model_path}")
        return True
    except Exception as e:
        print(f"Error training model: {str(e)}")
        return False

# Function to load the model if it exists
def load_model():
    global classifier, training_data, training_labels
    
    # Try to load training data first
    loaded_data = load_training_data()
    
    # Then try to load the model
    if os.path.exists(model_path):
        try:
            with open(model_path, 'rb') as f:
                classifier = pickle.load(f)
            print(f"Custom classifier model loaded from {model_path}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            
            # If we have training data, attempt to retrain
            if loaded_data and len(training_data) > 0:
                print("Attempting to retrain model with loaded data...")
                return train_model()
            return False
    else:
        print("No pre-trained model found.")
        return False

# Try to load the existing model and training data
model_loaded = load_model()

# Print instructions
print("Custom Hand Gesture Media Controls:")
print("- Different hand gestures will control media playback")
print("\nTraining Mode Controls:")
print("- Press 't' to toggle training data collection")
print("- Press '0-6' to select the gesture class to collect")
print("- Press 'm' to train the model")
print("- Press 's' to save training data without retraining")
print("- Press 'c' to clear all collected training data")
print("- ESC key: Exit program")

try:
    while running and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to get frame from camera")
            break
        
        # Mirror the frame horizontally for more intuitive interaction
        frame = cv2.flip(frame, 1)
        
        # Get frame dimensions
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
        
        # Detect hand in frame
        skin_mask = detect_skin(frame)
        hand_contour = find_contours(skin_mask)
        
        # Draw the skin mask in a smaller window
        skin_display = cv2.resize(skin_mask, (w//4, h//4))
        frame[0:h//4, 0:w//4] = cv2.cvtColor(skin_display, cv2.COLOR_GRAY2BGR)
        
        if hand_contour is not None:
            # Draw the contour
            cv2.drawContours(frame, [hand_contour], 0, (0, 255, 0), 2)
            
            # Find convex hull and defects
            contour, hull, defects = find_convex_hull_and_defects(hand_contour)
            
            if contour is not None:
                # Draw hull
                if hull is not None:
                    hull_points = [contour[h[0]] for h in hull]
                    cv2.drawContours(frame, [np.array(hull_points)], 0, (0, 0, 255), 3)
                
                # Draw defects
                if defects is not None:
                    for i in range(defects.shape[0]):
                        s, e, f, d = defects[i, 0]
                        start = tuple(contour[s][0])
                        end = tuple(contour[e][0])
                        far = tuple(contour[f][0])
                        
                        # Draw a line from start to end
                        cv2.line(frame, start, end, [0, 255, 0], 2)
                        
                        # Draw a circle at the farthest point
                        cv2.circle(frame, far, 5, [0, 0, 255], -1)
                
                # Extract features from hand
                features = extract_features(contour, defects, frame.shape)
                
                if features:
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
                            
                            # Only accept predictions with confidence above 0.4 (40%)
                            max_prob = np.max(probabilities)
                            if max_prob > 0.4:
                                # Get the class with the highest probability
                                prediction = np.argmax(probabilities)
                                # Convert the prediction to a gesture name
                                status_text = gesture_classes.get(prediction, "Unknown gesture")
                            else:
                                # If no class has confidence above 40%, don't recognize any gesture
                                prediction = 0
                                status_text = "No gesture detected (low confidence)"
                            
                            # Display the prediction probabilities
                            for i, prob in enumerate(probabilities):
                                if prob > 0.05:  # Only display significant probabilities
                                    prob_text = f"{gesture_classes.get(i, 'Unknown')}: {prob:.2f}"
                                    # Highlight the selected prediction with a different color
                                    text_color = (0, 0, 255) if prob > 0.4 else (255, 0, 0)
                                    cv2.putText(frame, prob_text, 
                                                (w - 300, 30 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)
                            
                            # Execute actions based on the predicted gesture only if confidence is high enough
                            if max_prob > 0.4:
                                current_time = time.time()
                                
                                # Volume control variables
                                volume_adjustment_interval = 0.3
                                
                                if prediction == 1 and previous_gesture != status_text:  # Play/Pause
                                    if current_time - last_action_time > cooldown_time:
                                        pyautogui.press('playpause')
                                        last_action_time = current_time
                                        print(f"Action: Play/Pause (confidence: {max_prob:.2f})")
                                elif prediction == 2 and previous_gesture != status_text:  # Stop
                                    if current_time - last_action_time > cooldown_time:
                                        pyautogui.press('stop')
                                        last_action_time = current_time
                                        print(f"Action: Stop (confidence: {max_prob:.2f})")
                                elif prediction == 3 and previous_gesture != status_text:  # Next Track
                                    if current_time - last_action_time > cooldown_time:
                                        pyautogui.press('nexttrack')
                                        last_action_time = current_time
                                        print(f"Action: Next Track (confidence: {max_prob:.2f})")
                                elif prediction == 4 and previous_gesture != status_text:  # Previous Track
                                    if current_time - last_action_time > cooldown_time:
                                        pyautogui.press('prevtrack')
                                        last_action_time = current_time
                                        print(f"Action: Previous Track (confidence: {max_prob:.2f})")
                                elif prediction == 5:  # Volume Up
                                    if current_time - last_action_time > volume_adjustment_interval:
                                        pyautogui.press('volumeup')
                                        last_action_time = current_time
                                        print(f"Action: Volume Up (confidence: {max_prob:.2f})")
                                elif prediction == 6:  # Volume Down
                                    if current_time - last_action_time > volume_adjustment_interval:
                                        pyautogui.press('volumedown')
                                        last_action_time = current_time
                                        print(f"Action: Volume Down (confidence: {max_prob:.2f})")
                                
                                # Update previous gesture only for high-confidence predictions
                                previous_gesture = status_text
                        except Exception as e:
                            print(f"Error during prediction: {e}")
                    
                    # Display extracted features
                    for i in range(min(6, len(features))):
                        cv2.putText(frame, f"F{i}: {features[i]:.3f}", 
                                    (10, 60 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Display status text
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display frame
        cv2.imshow("Custom Hand Gesture Media Control", frame)
        
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
        
        # 'm' key to train the model
        elif key == ord('m'):
            print("Training model...")
            model_loaded = train_model()
        
        # 's' key to save data without retraining
        elif key == ord('s'):
            print("Saving training data...")
            save_training_data()
        
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
    print(f"Unexpected error: {str(e)}")

# Before exiting, save the training data if it's not empty
if len(training_data) > 0:
    print("Saving data before exit...")
    save_training_data()

# Cleanup
cap.release()
cv2.destroyAllWindows()