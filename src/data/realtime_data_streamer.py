#!/usr/bin/env python3
"""
Real-time Data Streamer using yfinance for live market data.
Provides streaming market data for paper trading.
"""

import yfinance as yf
import pandas as pd
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Callable, Optional
import logging
from queue import Queue
import asyncio


class RealtimeDataStreamer:
    """
    Real-time market data streamer using yfinance.
    Fetches live market data at specified intervals.
    """
    
    def __init__(self, symbols: List[str], update_interval: int = 5):
        """
        Initialize the real-time data streamer.
        
        Args:
            symbols: List of stock symbols to stream
            update_interval: Update interval in seconds (minimum 5 for yfinance)
        """
        self.symbols = symbols
        self.update_interval = max(update_interval, 5)  # yfinance rate limit
        self.logger = logging.getLogger(__name__)
        
        # Data storage
        self.current_data: Dict[str, Dict] = {}
        self.data_queue = Queue()
        self.subscribers: List[Callable] = []
        
        # Control flags
        self.is_streaming = False
        self.stream_thread = None
        
        # Market hours (IST for Indian stocks)
        self.market_open_time = "09:15"
        self.market_close_time = "15:30"
        
        self.logger.info(f"Initialized real-time streamer for {len(symbols)} symbols")
    
    def add_subscriber(self, callback: Callable):
        """Add a callback function to receive real-time data updates."""
        self.subscribers.append(callback)
        self.logger.info(f"Added subscriber: {callback.__name__}")
    
    def is_market_open(self) -> bool:
        """Check if the market is currently open."""
        now = datetime.now()
        current_time = now.strftime("%H:%M")
        
        # Check if it's a weekday (Monday=0, Sunday=6)
        if now.weekday() >= 5:  # Saturday or Sunday
            return False
        
        return self.market_open_time <= current_time <= self.market_close_time
    
    def fetch_current_data(self) -> Dict[str, Dict]:
        """Fetch current market data for all symbols."""
        current_data = {}
        
        try:
            for symbol in self.symbols:
                try:
                    # Fetch 1-minute data for the last 2 periods
                    ticker = yf.Ticker(symbol)
                    data = ticker.history(period="1d", interval="1m")
                    
                    if not data.empty:
                        latest = data.iloc[-1]
                        
                        current_data[symbol] = {
                            'symbol': symbol,
                            'timestamp': datetime.now(),
                            'open': float(latest['Open']),
                            'high': float(latest['High']),
                            'low': float(latest['Low']),
                            'close': float(latest['Close']),
                            'volume': int(latest['Volume']),
                            'price': float(latest['Close']),  # Current price
                        }
                        
                        # Calculate additional metrics
                        if len(data) >= 2:
                            prev_close = float(data.iloc[-2]['Close'])
                            current_price = float(latest['Close'])
                            
                            current_data[symbol].update({
                                'change': current_price - prev_close,
                                'change_percent': ((current_price - prev_close) / prev_close) * 100,
                                'prev_close': prev_close
                            })
                        
                except Exception as e:
                    self.logger.error(f"Error fetching data for {symbol}: {e}")
                    continue
            
            self.logger.debug(f"Fetched data for {len(current_data)} symbols")
            
        except Exception as e:
            self.logger.error(f"Error in fetch_current_data: {e}")
        
        return current_data
    
    def _stream_worker(self):
        """Background worker thread for streaming data."""
        self.logger.info("Starting real-time data streaming...")
        
        while self.is_streaming:
            try:
                # Only fetch data during market hours or for testing
                if self.is_market_open() or True:  # Set to True for testing
                    # Fetch current data
                    new_data = self.fetch_current_data()
                    
                    if new_data:
                        # Update current data
                        self.current_data.update(new_data)
                        
                        # Add to queue
                        self.data_queue.put({
                            'timestamp': datetime.now(),
                            'data': new_data.copy()
                        })
                        
                        # Notify subscribers
                        for callback in self.subscribers:
                            try:
                                callback(new_data.copy())
                            except Exception as e:
                                self.logger.error(f"Error in subscriber callback: {e}")
                        
                        self.logger.info(f"Streamed data for {len(new_data)} symbols")
                    
                else:
                    self.logger.info("Market is closed, pausing data streaming...")
                
                # Wait for next update
                time.sleep(self.update_interval)
                
            except Exception as e:
                self.logger.error(f"Error in streaming worker: {e}")
                time.sleep(self.update_interval)
    
    def start_streaming(self):
        """Start real-time data streaming."""
        if self.is_streaming:
            self.logger.warning("Streaming is already active")
            return
        
        self.is_streaming = True
        self.stream_thread = threading.Thread(target=self._stream_worker, daemon=True)
        self.stream_thread.start()
        
        self.logger.info("Real-time data streaming started")
    
    def stop_streaming(self):
        """Stop real-time data streaming."""
        if not self.is_streaming:
            self.logger.warning("Streaming is not active")
            return
        
        self.is_streaming = False
        
        if self.stream_thread and self.stream_thread.is_alive():
            self.stream_thread.join(timeout=10)
        
        self.logger.info("Real-time data streaming stopped")
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get the current price for a symbol."""
        if symbol in self.current_data:
            return self.current_data[symbol].get('price')
        return None
    
    def get_current_data(self, symbol: str) -> Optional[Dict]:
        """Get the current market data for a symbol."""
        return self.current_data.get(symbol)
    
    def get_all_current_data(self) -> Dict[str, Dict]:
        """Get current market data for all symbols."""
        return self.current_data.copy()
    
    def get_market_status(self) -> Dict:
        """Get current market status information."""
        return {
            'is_open': self.is_market_open(),
            'current_time': datetime.now().strftime("%H:%M:%S"),
            'market_open': self.market_open_time,
            'market_close': self.market_close_time,
            'symbols_streaming': len(self.current_data),
            'last_update': max([data.get('timestamp', datetime.min) 
                              for data in self.current_data.values()], 
                              default=datetime.min)
        }


# Example usage and testing
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Test symbols
    symbols = ["RELIANCE.NS", "TCS.NS", "INFY.NS"]
    
    # Create streamer
    streamer = RealtimeDataStreamer(symbols, update_interval=10)
    
    # Add a test subscriber
    def data_callback(data):
        print(f"Received data update: {len(data)} symbols")
        for symbol, info in data.items():
            print(f"  {symbol}: ₹{info['price']:.2f} ({info.get('change_percent', 0):.2f}%)")
    
    streamer.add_subscriber(data_callback)
    
    # Start streaming
    streamer.start_streaming()
    
    try:
        # Run for 60 seconds
        time.sleep(60)
    except KeyboardInterrupt:
        print("Stopping streamer...")
    finally:
        streamer.stop_streaming()