#!/usr/bin/env python3
"""
Test script to verify the data processing functionality with test data.
"""

import sys
import pandas as pd
import os

# Add src to Python path
sys.path.append('src')

from src.data.data_processor import DataProcessor
from src.data.data_transformer import transform_yfinance_test_data

def test_data_processing():
    """Test the processing of test data."""
    print("Testing data processing with test data...")
    
    # Define test file path
    test_file = "RELIANCE_NS_test_data.csv"
    
    if not os.path.exists(test_file):
        print(f"Error: Test file {test_file} not found")
        return False
    
    # Load the raw data
    raw_data = pd.read_csv(test_file)
    print(f"Raw data shape: {raw_data.shape}")
    print(f"Raw data columns: {raw_data.columns.tolist()}")
    
    # Initialize the data processor
    processor = DataProcessor()
    
    # Process the data with file path for transformation
    symbol = "RELIANCE.NS"
    processed_data, stats = processor.process_stock_data(raw_data, symbol, file_path=test_file)
    
    if processed_data is None or processed_data.empty:
        print("Error: Data processing failed")
        return False
    
    # Print information about the processed data
    print("\nProcessed data shape:", processed_data.shape)
    print("\nProcessed data columns:", processed_data.columns.tolist())
    print("\nProcessed data types:")
    print(processed_data.dtypes)
    
    # Print first few rows
    print("\nFirst 5 rows of processed data:")
    print(processed_data.head())
    
    # Print processing statistics
    print("\nProcessing statistics:")
    for key, value in stats.items():
        if key != "cleaning_stats" and key != "validation_stats" and key != "normalization_stats":
            print(f"  {key}: {value}")
    
    # Save processed data to CSV for inspection
    processed_file = "processed_test_data.csv"
    processed_data.to_csv(processed_file)
    print(f"\nProcessed data saved to {processed_file}")
    
    return True

if __name__ == "__main__":
    success = test_data_processing()
    print("\nTest result:", "Success" if success else "Failed")