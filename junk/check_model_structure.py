#!/usr/bin/env python3
"""
Script to check the structure of trained LSTM models.
"""

import sys
import os
import json
import tensorflow as tf

# Add src to Python path
sys.path.append('src')

def main():
    """Check the structure of trained LSTM models."""
    # Get list of model directories
    model_dirs = [d for d in os.listdir('models') if os.path.isdir(os.path.join('models', d)) and d != 'checkpoints']
    
    print(f"Found {len(model_dirs)} model directories")
    
    # Check a sample model
    if model_dirs:
        sample_dir = model_dirs[0]
        model_path = os.path.join('models', sample_dir, 'lstm_model.keras')
        
        if os.path.exists(model_path):
            # Load model
            model = tf.keras.models.load_model(model_path)
            
            # Print model summary
            print(f"\nModel structure for {sample_dir}:")
            model.summary()
            
            # Load metadata
            metadata_path = f"{model_path}_metadata.json"
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                print("\nModel metadata:")
                for key, value in metadata.items():
                    print(f"  {key}: {value}")
        else:
            print(f"No model file found in {sample_dir}")
    
    # Count models with different feature counts
    feature_counts = {}
    for model_dir in model_dirs:
        metadata_path = os.path.join('models', model_dir, 'lstm_model.keras_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            features = metadata.get('features', 0)
            if features in feature_counts:
                feature_counts[features] += 1
            else:
                feature_counts[features] = 1
    
    print("\nFeature count distribution:")
    for features, count in sorted(feature_counts.items()):
        print(f"  {features} features: {count} models")

if __name__ == "__main__":
    main()