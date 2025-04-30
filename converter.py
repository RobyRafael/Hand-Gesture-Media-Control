import pickle
import numpy as np
import pandas as pd
import os
import argparse

def convert_naive_bayes_to_csv(model_path, output_dir="model_data"):
    """
    Convert a Naive Bayes model saved as a pickle file to CSV files for better inspection.
    
    Args:
        model_path (str): Path to the pickle file containing the model
        output_dir (str): Directory to save the CSV files
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load the model
    print(f"Loading model from {model_path}...")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Check if it's a GaussianNB model
    if not hasattr(model, 'theta_') or not hasattr(model, 'var_'):
        print("WARNING: The model doesn't appear to be a GaussianNB model.")
        print(f"Model type: {type(model)}")
    
    # Extract model parameters
    try:
        # Get class labels if available
        if hasattr(model, 'classes_'):
            classes = model.classes_
            class_names = {
                0: "No gesture detected",
                1: "Play/Pause",
                2: "Stop",
                3: "Next Track",
                4: "Previous Track",
                5: "Volume Up",
                6: "Volume Down"
            }
            
            # Create a DataFrame for class information
            class_df = pd.DataFrame({
                'Class_Number': classes,
                'Class_Name': [class_names.get(c, f"Class {c}") for c in classes],
                'Class_Prior': model.class_prior_ if hasattr(model, 'class_prior_') else np.nan
            })
            class_df.to_csv(os.path.join(output_dir, 'class_info.csv'), index=False)
            print(f"Saved class information to {os.path.join(output_dir, 'class_info.csv')}")
        
        # For GaussianNB, export means (theta_) and variances (var_)
        if hasattr(model, 'theta_') and hasattr(model, 'var_'):
            # Define feature names based on the original code
            feature_names = [
                'distance_thumb_index',
                'distance_thumb_middle',
                'distance_thumb_ring',
                'distance_thumb_little',
                'distance_index_middle',
                'distance_index_middle_dip',
                'distance_ring_middle'
            ]
            
            # Create DataFrames for means (theta_) and variances (var_)
            means_df = pd.DataFrame(
                model.theta_,
                columns=feature_names,
                index=[class_names.get(c, f"Class {c}") for c in classes]
            )
            means_df.index.name = 'Class'
            means_df.to_csv(os.path.join(output_dir, 'feature_means.csv'))
            print(f"Saved feature means to {os.path.join(output_dir, 'feature_means.csv')}")
            
            var_df = pd.DataFrame(
                model.var_,
                columns=feature_names,
                index=[class_names.get(c, f"Class {c}") for c in classes]
            )
            var_df.index.name = 'Class'
            var_df.to_csv(os.path.join(output_dir, 'feature_variances.csv'))
            print(f"Saved feature variances to {os.path.join(output_dir, 'feature_variances.csv')}")
            
            # If the model was trained with sample weights, we might also have class counts
            if hasattr(model, 'class_count_'):
                counts_df = pd.DataFrame({
                    'Class': [class_names.get(c, f"Class {c}") for c in classes],
                    'Sample_Count': model.class_count_
                })
                counts_df.to_csv(os.path.join(output_dir, 'class_counts.csv'), index=False)
                print(f"Saved class sample counts to {os.path.join(output_dir, 'class_counts.csv')}")
                
        # Export raw model data as a last resort
        model_attrs = {attr: getattr(model, attr) for attr in dir(model) 
                      if not attr.startswith('_') and not callable(getattr(model, attr))}
        
        # Save model attributes that can be converted to DataFrame
        for attr, value in model_attrs.items():
            if isinstance(value, (np.ndarray, list)):
                try:
                    if isinstance(value, np.ndarray) and value.ndim <= 2:
                        if value.ndim == 1:
                            df = pd.DataFrame({attr: value})
                        else:
                            df = pd.DataFrame(value)
                        df.to_csv(os.path.join(output_dir, f'{attr}.csv'))
                        print(f"Saved {attr} to {os.path.join(output_dir, f'{attr}.csv')}")
                except Exception as e:
                    print(f"Could not save {attr}: {e}")
        
        print(f"\nConversion complete! Files saved in {output_dir} directory.")
        
    except Exception as e:
        print(f"Error during conversion: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert Naive Bayes model pickle to CSV files')
    parser.add_argument('--model', type=str, default='hand_gesture_model.pkl',
                        help='Path to the pickle model file')
    parser.add_argument('--output', type=str, default='model_data',
                        help='Output directory for CSV files')
    
    args = parser.parse_args()
    
    # Convert the model
    convert_naive_bayes_to_csv(args.model, args.output)