#!/usr/bin/env python3
"""
Test script to verify the data transformer functionality.
"""

import sys
import pandas as pd
import os

# Add src to Python path
sys.path.append('src')

from src.data.data_transformer import transform_yfinance_test_data

def test_data_transformation():
    """Test the transformation of test data."""
    print("Testing data transformation...")
    
    # Define test file path
    test_file = "RELIANCE_NS_test_data.csv"
    
    if not os.path.exists(test_file):
        print(f"Error: Test file {test_file} not found")
        return False
    
    # Transform the data
    transformed_data = transform_yfinance_test_data(test_file)
    
    if transformed_data is None:
        print("Error: Data transformation failed")
        return False
    
    # Print information about the transformed data
    print("\nTransformed data shape:", transformed_data.shape)
    print("\nTransformed data columns:", transformed_data.columns.tolist())
    print("\nTransformed data types:")
    print(transformed_data.dtypes)
    
    # Print first few rows
    print("\nFirst 5 rows of transformed data:")
    print(transformed_data.head())
    
    # Save transformed data to CSV for inspection
    transformed_file = "transformed_test_data.csv"
    transformed_data.to_csv(transformed_file)
    print(f"\nTransformed data saved to {transformed_file}")
    
    return True

if __name__ == "__main__":
    success = test_data_transformation()
    print("\nTest result:", "Success" if success else "Failed")