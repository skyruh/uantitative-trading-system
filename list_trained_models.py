#!/usr/bin/env python3
"""
Script to list all trained models and their key metrics.
"""

import sys
import os
import json
import pandas as pd

# Add src to Python path
sys.path.append('src')

def main():
    """List all trained models and their key metrics."""
    # Get list of model directories
    model_dirs = [d for d in os.listdir('models') if os.path.isdir(os.path.join('models', d)) and d != 'checkpoints']
    
    print(f"Found {len(model_dirs)} trained models")
    
    # Create a list to store model information
    model_info = []
    
    for model_dir in model_dirs:
        symbol = f"{model_dir}.NS"
        model_path = os.path.join('models', model_dir, 'lstm_model.keras')
        metadata_path = f"{model_path}_metadata.json"
        
        if os.path.exists(model_path) and os.path.exists(metadata_path):
            # Load metadata
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Get model file size
            model_size = os.path.getsize(model_path) / 1024  # KB
            
            # Check if processed data exists
            processed_file = f"data/processed/{model_dir}_processed.csv"
            data_size = 0
            num_samples = 0
            
            if os.path.exists(processed_file):
                data_size = os.path.getsize(processed_file) / 1024  # KB
                
                # Get number of samples
                try:
                    df = pd.read_csv(processed_file)
                    num_samples = len(df)
                except:
                    pass
            
            # Add to model info
            model_info.append({
                'symbol': symbol,
                'features': metadata.get('features', 0),
                'sequence_length': metadata.get('sequence_length', 0),
                'is_trained': metadata.get('is_trained', False),
                'model_size_kb': model_size,
                'data_size_kb': data_size,
                'num_samples': num_samples
            })
    
    # Convert to DataFrame for better display
    df = pd.DataFrame(model_info)
    
    # Sort by symbol
    df = df.sort_values('symbol')
    
    # Print summary
    print("\nModel Summary:")
    print(df.to_string(index=False))
    
    # Print statistics
    print("\nStatistics:")
    print(f"Total models: {len(df)}")
    print(f"Average features: {df['features'].mean():.2f}")
    print(f"Average model size: {df['model_size_kb'].mean():.2f} KB")
    print(f"Average data size: {df['data_size_kb'].mean():.2f} KB")
    print(f"Average samples: {df['num_samples'].mean():.2f}")

if __name__ == "__main__":
    main()