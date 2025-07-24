"""
Stock data fetcher for collecting OHLCV data for Indian stocks.
Implements batch processing with progress tracking and error recovery.
"""

import logging
import time
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from src.data.yfinance_client import YFinanceClient
from src.data.data_storage import DataStorage
from src.data.data_transformer import transform_yfinance_test_data


class StockDataFetcher:
    """
    Fetches stock data for multiple symbols with batch processing and error recovery.
    
    Features:
    - Batch processing with progress tracking
    - Error recovery and retry mechanisms
    - Support for different stock categories (NIFTY 50, mid-cap, small-cap)
    - Comprehensive logging and reporting
    """
    
    def __init__(self, 
                 data_client: Optional[YFinanceClient] = None,
                 data_storage: Optional[DataStorage] = None,
                 batch_size: int = 10,
                 delay_between_batches: float = 1.0):
        """
        Initialize stock data fetcher.
        
        Args:
            data_client: YFinance client for data fetching
            data_storage: Data storage for saving fetched data
            batch_size: Number of stocks to process in each batch
            delay_between_batches: Delay between batches in seconds
        """
        self.data_client = data_client or YFinanceClient()
        self.data_storage = data_storage or DataStorage()
        self.batch_size = batch_size
        self.delay_between_batches = delay_between_batches
        self.logger = logging.getLogger(__name__)
        
        # Statistics tracking
        self.stats = {
            'total_symbols': 0,
            'successful_fetches': 0,
            'failed_fetches': 0,
            'total_records': 0,
            'start_time': None,
            'end_time': None,
            'errors': []
        }
    
    def load_stock_symbols(self, config_file: str = "config/indian_stocks.txt") -> List[str]:
        """
        Load stock symbols from configuration file.
        
        Args:
            config_file: Path to configuration file with stock symbols
            
        Returns:
            List of stock symbols
        """
        try:
            config_path = Path(config_file)
            if not config_path.exists():
                self.logger.error(f"Stock symbols file not found: {config_file}")
                return []
            
            symbols = []
            with open(config_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    # Skip comments and empty lines
                    if line and not line.startswith('#'):
                        symbols.append(line)
            
            self.logger.info(f"Loaded {len(symbols)} stock symbols from {config_file}")
            return symbols
            
        except Exception as e:
            self.logger.error(f"Failed to load stock symbols: {str(e)}")
            return []
    
    def get_nifty50_symbols(self) -> List[str]:
        """Get NIFTY 50 stock symbols."""
        return self.load_stock_symbols("config/indian_stocks.txt")
    
    def get_extended_stock_list(self) -> List[str]:
        """
        Get extended list of Indian stocks including mid-cap and small-cap.
        For now, returns NIFTY 50. Can be extended with additional symbol files.
        """
        symbols = self.get_nifty50_symbols()
        
        # TODO: Add mid-cap and small-cap symbols from additional config files
        # symbols.extend(self.load_stock_symbols("config/midcap_stocks.txt"))
        # symbols.extend(self.load_stock_symbols("config/smallcap_stocks.txt"))
        
        return symbols
    
    def fetch_single_stock_data(self, 
                               symbol: str, 
                               start_date: str, 
                               end_date: str,
                               include_news: bool = True) -> Dict:
        """
        Fetch data for a single stock symbol.
        
        Args:
            symbol: Stock symbol to fetch
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            include_news: Whether to fetch news data
            
        Returns:
            Dictionary with fetch results and statistics
        """
        result = {
            'symbol': symbol,
            'success': False,
            'stock_records': 0,
            'news_records': 0,
            'error': None
        }
        
        try:
            self.logger.info(f"Fetching data for {symbol}")
            
            # Fetch stock data
            stock_data = self.data_client.fetch_stock_data(symbol, start_date, end_date)
            
            if not stock_data.empty:
                # Process multi-level columns if present
                if isinstance(stock_data.columns, pd.MultiIndex):
                    # Extract the first level (OHLCV) and convert to lowercase
                    stock_data.columns = [col[0].lower() for col in stock_data.columns]
                else:
                    # Just convert to lowercase if not multi-level
                    stock_data.columns = [col.lower() for col in stock_data.columns]
                
                # Reset index to make date a column
                if isinstance(stock_data.index, pd.DatetimeIndex):
                    stock_data = stock_data.reset_index()
                    stock_data.rename(columns={'index': 'date'}, inplace=True)
                
                # Add symbol column if not present
                if 'symbol' not in stock_data.columns:
                    clean_symbol = symbol.replace('.NS', '').replace('.BO', '')
                    stock_data['symbol'] = clean_symbol
                
                # Save stock data
                if self.data_storage.save_stock_data(symbol, stock_data):
                    result['stock_records'] = len(stock_data)
                    self.logger.info(f"Saved {len(stock_data)} stock records for {symbol}")
                else:
                    raise Exception("Failed to save stock data")
            else:
                self.logger.warning(f"No stock data retrieved for {symbol}")
                raise Exception("No stock data available")
            
            # Fetch news data if requested
            if include_news:
                try:
                    news_data = self.data_client.fetch_news_data(symbol, start_date, end_date)
                    
                    if news_data:
                        if self.data_storage.save_news_data(symbol, news_data):
                            result['news_records'] = len(news_data)
                            self.logger.info(f"Saved {len(news_data)} news records for {symbol}")
                        else:
                            self.logger.warning(f"Failed to save news data for {symbol}")
                    else:
                        self.logger.info(f"No news data available for {symbol}")
                        
                except Exception as e:
                    self.logger.warning(f"Failed to fetch news for {symbol}: {str(e)}")
                    # Don't fail the entire fetch for news errors
            
            result['success'] = True
            
        except Exception as e:
            error_msg = f"Failed to fetch data for {symbol}: {str(e)}"
            self.logger.error(error_msg)
            result['error'] = str(e)
            self.stats['errors'].append({'symbol': symbol, 'error': str(e)})
        
        return result
    
    def fetch_batch_data(self, 
                        symbols: List[str], 
                        start_date: str, 
                        end_date: str,
                        include_news: bool = True) -> List[Dict]:
        """
        Fetch data for a batch of symbols.
        
        Args:
            symbols: List of stock symbols
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            include_news: Whether to fetch news data
            
        Returns:
            List of fetch results for each symbol
        """
        results = []
        
        for symbol in symbols:
            result = self.fetch_single_stock_data(symbol, start_date, end_date, include_news)
            results.append(result)
            
            # Update statistics
            if result['success']:
                self.stats['successful_fetches'] += 1
                self.stats['total_records'] += result['stock_records']
            else:
                self.stats['failed_fetches'] += 1
        
        return results
    
    def fetch_all_stocks_data(self, 
                             start_date: str, 
                             end_date: str,
                             symbols: Optional[List[str]] = None,
                             include_news: bool = True,
                             stock_category: str = "nifty50") -> Dict:
        """
        Fetch data for all stocks with batch processing and progress tracking.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            symbols: Custom list of symbols (optional)
            include_news: Whether to fetch news data
            stock_category: Category of stocks ("nifty50" or "extended")
            
        Returns:
            Dictionary with overall fetch statistics and results
        """
        # Initialize statistics
        self.stats = {
            'total_symbols': 0,
            'successful_fetches': 0,
            'failed_fetches': 0,
            'total_records': 0,
            'start_time': datetime.now(),
            'end_time': None,
            'errors': []
        }
        
        try:
            # Get symbols to fetch
            if symbols is None:
                if stock_category == "extended":
                    symbols = self.get_extended_stock_list()
                else:
                    symbols = self.get_nifty50_symbols()
            
            if not symbols:
                raise Exception("No symbols to fetch")
            
            self.stats['total_symbols'] = len(symbols)
            self.logger.info(f"Starting data fetch for {len(symbols)} symbols from {start_date} to {end_date}")
            
            # Process symbols in batches
            all_results = []
            
            with tqdm(total=len(symbols), desc="Fetching stock data") as pbar:
                for i in range(0, len(symbols), self.batch_size):
                    batch_symbols = symbols[i:i + self.batch_size]
                    batch_num = (i // self.batch_size) + 1
                    total_batches = (len(symbols) + self.batch_size - 1) // self.batch_size
                    
                    self.logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch_symbols)} symbols)")
                    
                    # Fetch batch data
                    batch_results = self.fetch_batch_data(batch_symbols, start_date, end_date, include_news)
                    all_results.extend(batch_results)
                    
                    # Update progress bar
                    pbar.update(len(batch_symbols))
                    
                    # Add delay between batches (except for the last batch)
                    if i + self.batch_size < len(symbols):
                        time.sleep(self.delay_between_batches)
            
            self.stats['end_time'] = datetime.now()
            duration = self.stats['end_time'] - self.stats['start_time']
            
            # Generate summary report
            summary = self._generate_summary_report(duration)
            
            return {
                'success': True,
                'summary': summary,
                'detailed_results': all_results,
                'statistics': self.stats
            }
            
        except Exception as e:
            self.logger.error(f"Failed to fetch all stocks data: {str(e)}")
            self.stats['end_time'] = datetime.now()
            
            return {
                'success': False,
                'error': str(e),
                'statistics': self.stats
            }
    
    def _generate_summary_report(self, duration: timedelta) -> Dict:
        """Generate summary report of the fetch operation."""
        success_rate = (self.stats['successful_fetches'] / self.stats['total_symbols'] * 100) if self.stats['total_symbols'] > 0 else 0
        
        summary = {
            'total_symbols_processed': self.stats['total_symbols'],
            'successful_fetches': self.stats['successful_fetches'],
            'failed_fetches': self.stats['failed_fetches'],
            'success_rate_percent': round(success_rate, 2),
            'total_records_fetched': self.stats['total_records'],
            'duration_seconds': duration.total_seconds(),
            'duration_formatted': str(duration).split('.')[0],  # Remove microseconds
            'average_time_per_symbol': round(duration.total_seconds() / self.stats['total_symbols'], 2) if self.stats['total_symbols'] > 0 else 0,
            'errors_count': len(self.stats['errors'])
        }
        
        # Log summary
        self.logger.info("=== FETCH SUMMARY ===")
        self.logger.info(f"Total symbols: {summary['total_symbols_processed']}")
        self.logger.info(f"Successful: {summary['successful_fetches']}")
        self.logger.info(f"Failed: {summary['failed_fetches']}")
        self.logger.info(f"Success rate: {summary['success_rate_percent']}%")
        self.logger.info(f"Total records: {summary['total_records_fetched']}")
        self.logger.info(f"Duration: {summary['duration_formatted']}")
        self.logger.info(f"Avg time per symbol: {summary['average_time_per_symbol']}s")
        
        if self.stats['errors']:
            self.logger.warning(f"Errors encountered: {len(self.stats['errors'])}")
            for error in self.stats['errors'][:5]:  # Show first 5 errors
                self.logger.warning(f"  {error['symbol']}: {error['error']}")
        
        return summary
    
    def fetch_incremental_data(self, 
                              symbols: Optional[List[str]] = None,
                              days_back: int = 7,
                              include_news: bool = True) -> Dict:
        """
        Fetch incremental data for the last N days.
        
        Args:
            symbols: List of symbols to update (None for all available)
            days_back: Number of days to fetch from current date
            include_news: Whether to fetch news data
            
        Returns:
            Dictionary with fetch results
        """
        try:
            # Calculate date range
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
            
            # Get symbols to update
            if symbols is None:
                symbols = self.data_storage.get_available_symbols()
                if not symbols:
                    symbols = self.get_nifty50_symbols()
            
            self.logger.info(f"Fetching incremental data for {len(symbols)} symbols ({days_back} days)")
            
            return self.fetch_all_stocks_data(
                start_date=start_date,
                end_date=end_date,
                symbols=symbols,
                include_news=include_news
            )
            
        except Exception as e:
            self.logger.error(f"Failed to fetch incremental data: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def get_fetch_statistics(self) -> Dict:
        """Get current fetch statistics."""
        return self.stats.copy()
    
    def reset_statistics(self):
        """Reset fetch statistics."""
        self.stats = {
            'total_symbols': 0,
            'successful_fetches': 0,
            'failed_fetches': 0,
            'total_records': 0,
            'start_time': None,
            'end_time': None,
            'errors': []
        }