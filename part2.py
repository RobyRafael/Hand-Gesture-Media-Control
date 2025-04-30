import pickle
import numpy as np
import pandas as pd
import os
import argparse

def convert_raw_data_to_csv(model_path, output_dir="model_data"):
    """
    Convert the raw training data and model labels to CSV format for inspection.
    
    Args:
        model_path (str): Path to the pickle file containing the trained Naive Bayes model.
        output_dir (str): Directory to save the CSV files.
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load the model
    print(f"Loading model from {model_path}...")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    try:
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
        
        # Get the classes (labels) used by the model
        classes = model.classes_

        # Define feature names (you can adjust these names if needed)
        feature_names = [
            'distance_thumb_index',
            'distance_thumb_middle',
            'distance_thumb_ring',
            'distance_thumb_little',
            'distance_index_middle',
            'distance_index_middle_dip',
            'distance_ring_middle'
        ]
        
        # Extract training data and labels (these are the raw training data used for fitting the model)
        # The model itself doesn't contain X_train, so it should be passed from the code where training occurs
        # In this case, you would need to load the data from where it was originally stored
        # Let's assume we can access the raw training data

        # Manually add this data for demonstration (in real scenario, you'd use the original X_train and y_train)
        # For example:
        # X_train = np.array(training_data)  # (replace with actual training data)
        # y_train = np.array(training_labels)  # (replace with actual training labels)
        
        # Here's an example of how to create a raw dataset (using dummy data for demonstration):
        # Replace this with the actual training data if available
        X_train = np.random.rand(1000, 7)  # Random data for example purposes
        y_train = np.random.choice(classes, 1000)  # Random class labels for example purposes

        # Convert to DataFrame
        raw_data_df = pd.DataFrame(X_train, columns=feature_names)

        # Add class number and class label columns
        raw_data_df['classno'] = y_train
        raw_data_df['Class'] = [gesture_classes[class_no] for class_no in y_train]

        # Save to CSV
        output_csv_path = os.path.join(output_dir, 'raw_data.csv')
        raw_data_df.to_csv(output_csv_path, index=False)
        print(f"Saved raw data to {output_csv_path}")
        
    except Exception as e:
        print(f"Error during conversion: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert Naive Bayes model raw training data to CSV')
    parser.add_argument('--model', type=str, default='hand_gesture_model.pkl',
                        help='Path to the pickle model file')
    parser.add_argument('--output', type=str, default='model_data',
                        help='Output directory for CSV files')
    
    args = parser.parse_args()

    # Convert the model raw data
    convert_raw_data_to_csv(args.model, args.output)
