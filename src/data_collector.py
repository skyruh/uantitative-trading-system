"""
Enhanced Data Collection Module based on quant_bot approach
"""
import yfinance as yf
import pandas as pd
import os
import logging
from datetime import datetime, timedelta
import sys

class DataCollector:
    def __init__(self, data_dir="data/stocks"):
        self.logger = logging.getLogger('trading_system.data_collector')
        self.data_dir = data_dir
        
        # Ensure data directory exists
        os.makedirs(self.data_dir, exist_ok=True)
        
    def fetch_historical_data(self, symbol, period="max", interval="1d"):
        """Fetch historical data for a symbol"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                self.logger.warning(f"No data found for {symbol}")
                return None
                
            self.logger.info(f"Fetched {len(data)} records for {symbol} ({period})")
            return data
            
        except Exception as e:
            self.logger.error(f"Error fetching data for {symbol}: {e}")
            return None
    
    def save_data(self, symbol, data):
        """Save data to CSV file"""
        if data is None or data.empty:
            return False
            
        # Make sure the index is a DatetimeIndex
        if not isinstance(data.index, pd.DatetimeIndex):
            try:
                data.index = pd.to_datetime(data.index)
            except Exception as e:
                self.logger.error(f"Error converting index to DatetimeIndex for {symbol}: {e}")
                return False
                
        filename = f"{symbol.replace('.NS', '')}_data.csv"
        filepath = os.path.join(self.data_dir, filename)
        
        try:
            data.to_csv(filepath)
            self.logger.info(f"Saved data for {symbol} to {filepath}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving data for {symbol}: {e}")
            return False
    
    def load_data(self, symbol):
        """Load data from CSV file"""
        filename = f"{symbol.replace('.NS', '')}_data.csv"
        filepath = os.path.join(self.data_dir, filename)
        
        try:
            if os.path.exists(filepath):
                data = pd.read_csv(filepath, index_col=0, parse_dates=True)
                self.logger.info(f"Loaded {len(data)} records for {symbol}")
                return data
            else:
                self.logger.warning(f"No saved data found for {symbol}")
                return None
        except Exception as e:
            self.logger.error(f"Error loading data for {symbol}: {e}")
            return None
    
    def collect_historical_data(self, symbols):
        """Collect historical data for all symbols"""
        self.logger.info("Starting historical data collection...")
        
        periods = ["2y", "3y", "5y", "max"]  # Try different periods
        
        for symbol in symbols:
            print(f"\nCollecting data for {symbol}...")
            
            best_data = None
            best_period = None
            
            # Try different periods to get maximum data
            for period in periods:
                try:
                    data = self.fetch_historical_data(symbol, period=period, interval="1d")
                    
                    if data is not None and len(data) > 0:
                        print(f"  {period}: {len(data)} records")
                        
                        if best_data is None or len(data) > len(best_data):
                            best_data = data
                            best_period = period
                            
                except Exception as e:
                    print(f"  {period}: Failed - {e}")
            
            # Save the best dataset
            if best_data is not None:
                saved = self.save_data(symbol, best_data)
                if saved:
                    print(f"  ✓ Saved {len(best_data)} records from {best_period}")
                    self.logger.info(f"Historical data saved for {symbol}: {len(best_data)} records")
                else:
                    print(f"  ✗ Failed to save data for {symbol}")
            else:
                print(f"  ✗ No data available for {symbol}")
        
        print("\n=== COLLECTION COMPLETE ===")
        self.logger.info("Historical data collection completed")

def load_symbols(file_path):
    """Load symbols from a file"""
    symbols = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    symbols.append(line)
        return symbols
    except Exception as e:
        print(f"Error loading symbols from {file_path}: {e}")
        return []

if __name__ == "__main__":
    # Setup basic logging
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Get symbols file path from command line or use default
    symbols_file = sys.argv[1] if len(sys.argv) > 1 else "config/indian_stocks.txt"
    
    # Load symbols
    symbols = load_symbols(symbols_file)
    print(f"Loaded {len(symbols)} symbols from {symbols_file}")
    
    # Create collector and collect data
    collector = DataCollector()
    collector.collect_historical_data(symbols)