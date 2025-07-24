#!/usr/bin/env python3
"""
Script to process sentiment data and integrate it with stock price data.
"""

import sys
import os
import logging
import json
import pandas as pd
from datetime import datetime, timedelta
import concurrent.futures

# Add src to Python path
sys.path.append('src')

from src.data.sentiment_processor import SentimentProcessor
from src.config.settings import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SentimentProcessing")

def load_stock_data(symbol: str) -> pd.DataFrame:
    """Load processed stock data for a symbol."""
    # Try different naming conventions for the processed file
    base_symbol = symbol.split('.')[0] if '.' in symbol else symbol
    
    # Check for file with exact symbol name
    data_path = f"data/processed/{symbol.replace('.', '_')}_processed.csv"
    
    # If not found, try with base symbol (without .NS)
    if not os.path.exists(data_path):
        data_path = f"data/processed/{base_symbol}_processed.csv"
    
    # If still not found, return empty DataFrame
    if not os.path.exists(data_path):
        logger.warning(f"No processed data found for {symbol}")
        return pd.DataFrame()
    
    # Load data
    try:
        data = pd.read_csv(data_path)
        logger.info(f"Loaded {len(data)} rows for {symbol} from {data_path}")
        return data
    except Exception as e:
        logger.error(f"Error loading data for {symbol}: {str(e)}")
        return pd.DataFrame()

def save_stock_data(symbol: str, data: pd.DataFrame) -> bool:
    """Save processed stock data with sentiment features."""
    try:
        # Create directory if it doesn't exist
        os.makedirs("data/processed", exist_ok=True)
        
        # Determine file path
        base_symbol = symbol.split('.')[0] if '.' in symbol else symbol
        data_path = f"data/processed/{base_symbol}_processed.csv"
        
        # Save data
        data.to_csv(data_path, index=False)
        logger.info(f"Saved {len(data)} rows for {symbol} to {data_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving data for {symbol}: {str(e)}")
        return False

def process_stock(symbol: str, processor: SentimentProcessor) -> bool:
    """Process sentiment for a single stock."""
    try:
        # Load stock data
        data = load_stock_data(symbol)
        if data.empty:
            return False
        
        # Process sentiment
        data_with_sentiment = processor.process_sentiment_for_stock(symbol, data)
        
        # Save processed data
        return save_stock_data(symbol, data_with_sentiment)
    except Exception as e:
        logger.error(f"Error processing {symbol}: {str(e)}")
        return False

def main():
    """Process sentiment for all stocks."""
    try:
        logger.info("Starting sentiment processing...")
        
        # Get stock symbols
        symbols = config.get_stock_symbols()
        logger.info(f"Found {len(symbols)} symbols to process")
        
        # Create sentiment processor
        processor = SentimentProcessor()
        
        # Process stocks in parallel
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            future_to_symbol = {
                executor.submit(process_stock, symbol, processor): symbol
                for symbol in symbols
            }
            for future in concurrent.futures.as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    success = future.result()
                    results.append({
                        'symbol': symbol,
                        'success': success
                    })
                    if success:
                        logger.info(f"Successfully processed sentiment for {symbol}")
                    else:
                        logger.warning(f"Failed to process sentiment for {symbol}")
                except Exception as e:
                    logger.error(f"Error processing sentiment for {symbol}: {str(e)}")
                    results.append({
                        'symbol': symbol,
                        'success': False,
                        'error': str(e)
                    })
        
        # Summarize results
        success_count = sum(1 for r in results if r['success'])
        logger.info(f"Sentiment processing completed: {success_count}/{len(symbols)} successful")
        
        # Save processing summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_symbols': len(symbols),
            'success_count': success_count,
            'failed_count': len(symbols) - success_count,
            'results': results
        }
        
        with open("data/sentiment_processing_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Summary saved to data/sentiment_processing_summary.json")
        
        return 0 if success_count > 0 else 1
        
    except Exception as e:
        logger.error(f"Error during sentiment processing: {str(e)}")
        logger.exception("Full traceback:")
        return 1

if __name__ == "__main__":
    sys.exit(main())