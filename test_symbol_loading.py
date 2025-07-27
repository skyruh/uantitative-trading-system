#!/usr/bin/env python3
"""
Test script to check what symbols are being loaded for backtesting
"""

import os
import pandas as pd

def test_symbol_loading():
    """Test what symbols have processed data available"""
    
    processed_dir = "data/processed"
    
    if not os.path.exists(processed_dir):
        print(f"Processed directory {processed_dir} does not exist")
        return
    
    # List all processed files
    processed_files = [f for f in os.listdir(processed_dir) if f.endswith('_processed.csv')]
    print(f"Found {len(processed_files)} processed files:")
    
    for file in processed_files:
        symbol = file.replace('_processed.csv', '')
        file_path = os.path.join(processed_dir, file)
        
        try:
            data = pd.read_csv(file_path)
            print(f"  {symbol}: {len(data)} rows, dates from {data['date'].iloc[0]} to {data['date'].iloc[-1]}")
        except Exception as e:
            print(f"  {symbol}: Error reading file - {e}")
    
    # Check what symbols are expected
    symbols_file = "config/indian_stocks.txt"
    if os.path.exists(symbols_file):
        with open(symbols_file, 'r') as f:
            expected_symbols = [line.strip() for line in f if line.strip()]
        print(f"\nExpected {len(expected_symbols)} symbols from {symbols_file}")
        print(f"First 10 expected symbols: {expected_symbols[:10]}")
        
        # Check which symbols have processed data
        symbols_with_data = [f.replace('_processed.csv', '') for f in processed_files]
        missing_symbols = [s for s in expected_symbols if s not in symbols_with_data]
        print(f"\nSymbols with processed data: {len(symbols_with_data)}")
        print(f"Symbols missing processed data: {len(missing_symbols)}")
        if missing_symbols:
            print(f"First 10 missing symbols: {missing_symbols[:10]}")

if __name__ == "__main__":
    test_symbol_loading()