#!/usr/bin/env python3
"""
Script to update all stock data to the latest date using yfinance.
This script extracts symbols from the FINAL_COMBINED_SYMBOLS.csv file
and updates the data for all 517 stocks.
"""

import os
import sys
import pandas as pd
import yfinance as yf
from datetime import datetime
import time
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("StockDataUpdater")

def load_symbols(symbols_file="quant_bot/FINAL_COMBINED_SYMBOLS.csv"):
    """Load stock symbols from the CSV file"""
    try:
        df = pd.read_csv(symbols_file)
        symbols = df['symbol'].tolist()
        logger.info(f"Loaded {len(symbols)} symbols from {symbols_file}")
        return symbols
    except Exception as e:
        logger.error(f"Error loading symbols from {symbols_file}: {e}")
        return []

def fetch_stock_data(symbol, period="max"):
    """Fetch stock data for a symbol using yfinance"""
    try:
        logger.info(f"Fetching data for {symbol} with period={period}")
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period)
        
        if data.empty:
            logger.warning(f"No data found for {symbol}")
            return None
        
        logger.info(f"Fetched {len(data)} records for {symbol}")
        
        # Add some metadata
        if not data.empty:
            start_date = data.index.min().strftime('%Y-%m-%d')
            end_date = data.index.max().strftime('%Y-%m-%d')
            years = (data.index.max() - data.index.min()).days / 365.25
            logger.info(f"{symbol}: {len(data)} records ({start_date} to {end_date}, {years:.1f} years)")
        
        return data
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {e}")
        return None

def save_stock_data(symbol, data, output_dir="data/processed"):
    """Save stock data to CSV file"""
    if data is None or data.empty:
        logger.warning(f"No data to save for {symbol}")
        return False
    
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create filename (remove .NS suffix if present)
        filename = f"{symbol.replace('.NS', '')}_processed.csv"
        filepath = os.path.join(output_dir, filename)
        
        # Save to CSV
        data.to_csv(filepath)
        logger.info(f"Saved {len(data)} records for {symbol} to {filepath}")
        return True
    except Exception as e:
        logger.error(f"Error saving data for {symbol}: {e}")
        return False

def update_all_stocks(symbols, output_dir="data/processed", delay=0.5):
    """Update data for all stocks"""
    logger.info(f"Starting data update for {len(symbols)} symbols")
    
    results = {
        "success": 0,
        "failed": 0,
        "symbols": []
    }
    
    for i, symbol in enumerate(symbols):
        try:
            logger.info(f"Processing {i+1}/{len(symbols)}: {symbol}")
            
            # Fetch data
            data = fetch_stock_data(symbol, period="max")
            
            if data is not None and not data.empty:
                # Save data
                if save_stock_data(symbol, data, output_dir):
                    results["success"] += 1
                    results["symbols"].append({
                        "symbol": symbol,
                        "records": len(data),
                        "start_date": data.index.min().strftime('%Y-%m-%d'),
                        "end_date": data.index.max().strftime('%Y-%m-%d'),
                        "years": (data.index.max() - data.index.min()).days / 365.25
                    })
                else:
                    results["failed"] += 1
            else:
                results["failed"] += 1
                
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
            results["failed"] += 1
        
        # Add delay to avoid rate limiting
        time.sleep(delay)
    
    # Save summary
    try:
        summary_df = pd.DataFrame(results["symbols"])
        summary_df.to_csv(f"{output_dir}/update_summary_{datetime.now().strftime('%Y%m%d')}.csv", index=False)
    except Exception as e:
        logger.error(f"Error saving summary: {e}")
    
    logger.info(f"Data update completed: {results['success']} successful, {results['failed']} failed")
    return results

def main():
    """Main function"""
    print("=== STOCK DATA UPDATE ===")
    print("This script will update data for all stocks to the latest date.")
    
    # Load symbols
    symbols = load_symbols()
    
    if not symbols:
        print("No symbols found. Exiting.")
        return
    
    print(f"Found {len(symbols)} symbols to update.")
    
    # Update all stocks
    results = update_all_stocks(symbols)
    
    print("\n=== UPDATE SUMMARY ===")
    print(f"Total symbols: {len(symbols)}")
    print(f"Successfully updated: {results['success']}")
    print(f"Failed: {results['failed']}")
    print(f"Success rate: {results['success']/len(symbols)*100:.1f}%")
    
    print("\n=== UPDATE COMPLETE ===")

if __name__ == "__main__":
    main()