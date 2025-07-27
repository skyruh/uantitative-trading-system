#!/usr/bin/env python3
"""
Test script to debug market data loading
"""

import pandas as pd
from datetime import datetime

def test_market_data_loading():
    """Test how market data is loaded and indexed"""
    
    # Load the processed data
    data = pd.read_csv("data/processed/RELIANCE_processed.csv")
    print(f"Original data shape: {data.shape}")
    print(f"Original columns: {data.columns.tolist()}")
    print(f"First few dates: {data['date'].head().tolist()}")
    
    # Process the data like the orchestrator does
    if 'date' in data.columns:
        data['Date'] = pd.to_datetime(data['date'])
        # Convert timezone-aware datetime to timezone-naive for backtesting compatibility
        if data['Date'].dt.tz is not None:
            data['Date'] = data['Date'].dt.tz_localize(None)
        data = data.set_index('Date')
    
    print(f"\nProcessed data shape: {data.shape}")
    print(f"Index type: {type(data.index)}")
    print(f"Index name: {data.index.name}")
    print(f"First few index values: {data.index[:5].tolist()}")
    print(f"Last few index values: {data.index[-5:].tolist()}")
    
    # Test date filtering like the backtesting engine does
    test_date = datetime(2025, 7, 20)
    print(f"\nTesting date filtering for {test_date}")
    
    # Get data up to and including current date (no look-ahead)
    available_data = data[data.index <= test_date]
    print(f"Available data shape: {available_data.shape}")
    
    if len(available_data) > 0:
        latest_data = available_data.iloc[-1]
        print(f"Latest data date: {available_data.index[-1]}")
        print(f"Latest data: {latest_data[['open', 'high', 'low', 'close', 'volume']].to_dict()}")
    else:
        print("No data found for the test date!")

if __name__ == "__main__":
    test_market_data_loading()