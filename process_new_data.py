#!/usr/bin/env python3
"""
Script to process the new data from data/stocks folder and prepare it for model training.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import json
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add src to Python path
sys.path.append('src')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("DataProcessing")

def add_technical_indicators(df):
    """Add technical indicators to the dataframe."""
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Ensure the dataframe is sorted by date
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
    
    # Rename columns to lowercase for consistency
    df.columns = [col.lower() for col in df.columns]
    
    # Calculate RSI (14-period)
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    
    rs = avg_gain / avg_loss
    df['rsi_14'] = 100 - (100 / (1 + rs))
    
    # Calculate Simple Moving Averages
    df['sma_50'] = df['close'].rolling(window=50).mean()
    df['sma_200'] = df['close'].rolling(window=200).mean()
    
    # Calculate Bollinger Bands (20-period, 2 standard deviations)
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    std_dev = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + (std_dev * 2)
    df['bb_lower'] = df['bb_middle'] - (std_dev * 2)
    
    # Calculate MACD
    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # Calculate Average True Range (ATR)
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr_14'] = true_range.rolling(window=14).mean()
    
    # Calculate price momentum
    df['momentum_14'] = df['close'] / df['close'].shift(14) - 1
    
    # Calculate rate of change
    df['roc_10'] = df['close'].pct_change(periods=10) * 100
    
    # Add target column (1 if next day's close is higher, 0 otherwise)
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    
    # Drop rows with NaN values
    df = df.dropna()
    
    return df

def process_stock_data(file_path, output_dir):
    """Process a single stock data file."""
    try:
        # Extract symbol from file name
        file_name = os.path.basename(file_path)
        symbol = file_name.replace('_data.csv', '')
        
        logger.info(f"Processing {symbol}...")
        
        # Read data
        df = pd.read_csv(file_path)
        
        # Add technical indicators
        processed_df = add_technical_indicators(df)
        
        # Save processed data
        output_path = os.path.join(output_dir, f"{symbol}_processed.csv")
        processed_df.to_csv(output_path, index=False)
        
        return symbol, True, len(processed_df)
    except Exception as e:
        logger.error(f"Error processing {file_path}: {str(e)}")
        return os.path.basename(file_path).replace('_data.csv', ''), False, 0

def main():
    """Process all stock data files in the data/stocks folder."""
    # Create output directory if it doesn't exist
    output_dir = "data/processed"
    os.makedirs(output_dir, exist_ok=True)
    
    # Backup existing processed data
    backup_dir = "data/backups/processed_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    if os.path.exists(output_dir) and os.listdir(output_dir):
        logger.info(f"Backing up existing processed data to {backup_dir}")
        os.makedirs(backup_dir, exist_ok=True)
        for file in os.listdir(output_dir):
            if file.endswith("_processed.csv"):
                shutil.copy2(os.path.join(output_dir, file), os.path.join(backup_dir, file))
    
    # Get list of data files
    data_dir = "data/stocks"
    data_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('_data.csv')]
    
    logger.info(f"Found {len(data_files)} data files to process")
    
    # Process files in parallel
    results = []
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {executor.submit(process_stock_data, file, output_dir): file for file in data_files}
        for future in as_completed(futures):
            symbol, success, row_count = future.result()
            results.append({
                'symbol': symbol,
                'success': success,
                'row_count': row_count
            })
    
    # Summarize results
    success_count = sum(1 for r in results if r['success'])
    logger.info(f"Processing completed: {success_count}/{len(data_files)} files processed successfully")
    
    # Save processing summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_files': len(data_files),
        'success_count': success_count,
        'failed_count': len(data_files) - success_count,
        'results': results
    }
    
    with open(os.path.join(output_dir, "processing_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Summary saved to {os.path.join(output_dir, 'processing_summary.json')}")
    
    # Update system state
    system_state_path = "data/system_state.json"
    if os.path.exists(system_state_path):
        with open(system_state_path, 'r') as f:
            system_state = json.load(f)
        
        system_state['system_state']['data_collected'] = True
        system_state['timestamp'] = datetime.now().isoformat()
        
        with open(system_state_path, 'w') as f:
            json.dump(system_state, f, indent=2)
        
        logger.info("Updated system state: data_collected = True")

if __name__ == "__main__":
    main()