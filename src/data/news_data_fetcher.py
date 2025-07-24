"""
News data fetcher for retrieving headlines from yfinance for sentiment analysis.
Implements date-based filtering and association with stock data.
"""

import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
from pathlib import Path

from src.data.yfinance_client import YFinanceClient
from src.data.data_storage import DataStorage


class NewsDataFetcher:
    """
    Fetches news headlines for sentiment analysis with date-based filtering.
    
    Features:
    - Date-based news filtering and association with stock data
    - Error handling for missing news data
    - Integration with existing data storage system
    - Batch processing for multiple symbols
    """
    
    def __init__(self, 
                 data_client: Optional[YFinanceClient] = None,
                 data_storage: Optional[DataStorage] = None):
        """
        Initialize news data fetcher.
        
        Args:
            data_client: YFinance client for data fetching
            data_storage: Data storage for saving fetched news
        """
        self.data_client = data_client or YFinanceClient()
        self.data_storage = data_storage or DataStorage()
        self.logger = logging.getLogger(__name__)
    
    def fetch_news_for_symbol(self, 
                             symbol: str, 
                             start_date: str, 
                             end_date: str) -> List[Dict]:
        """
        Fetch news headlines for a single stock symbol.
        
        Args:
            symbol: Stock symbol (e.g., 'RELIANCE', 'TCS')
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            List of news dictionaries with filtered and cleaned data
            
        Raises:
            ValueError: For invalid input parameters
            Exception: For API or data processing errors
        """
        try:
            self.logger.info(f"Fetching news for {symbol} from {start_date} to {end_date}")
            
            # Validate date range
            self._validate_date_range(start_date, end_date)
            
            # Fetch news data using YFinance client
            news_data = self.data_client.fetch_news_data(symbol, start_date, end_date)
            
            if not news_data:
                self.logger.warning(f"No news data available for {symbol} in date range {start_date} to {end_date}")
                return []
            
            # Additional filtering and processing
            processed_news = self._process_news_data(news_data, symbol, start_date, end_date)
            
            self.logger.info(f"Successfully processed {len(processed_news)} news items for {symbol}")
            return processed_news
            
        except Exception as e:
            self.logger.error(f"Failed to fetch news for {symbol}: {str(e)}")
            # Return empty list instead of raising exception to handle missing news gracefully
            return []
    
    def fetch_and_save_news(self, 
                           symbol: str, 
                           start_date: str, 
                           end_date: str) -> Dict:
        """
        Fetch news data and save to storage.
        
        Args:
            symbol: Stock symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            Dictionary with fetch results and statistics
        """
        result = {
            'symbol': symbol,
            'success': False,
            'news_count': 0,
            'date_range': f"{start_date} to {end_date}",
            'error': None
        }
        
        try:
            # Fetch news data
            news_data = self.fetch_news_for_symbol(symbol, start_date, end_date)
            
            # Save to storage
            if self.data_storage.save_news_data(symbol, news_data):
                result['success'] = True
                result['news_count'] = len(news_data)
                self.logger.info(f"Successfully saved {len(news_data)} news items for {symbol}")
            else:
                raise Exception("Failed to save news data to storage")
                
        except Exception as e:
            error_msg = f"Failed to fetch and save news for {symbol}: {str(e)}"
            self.logger.error(error_msg)
            result['error'] = str(e)
        
        return result
    
    def fetch_news_for_multiple_symbols(self, 
                                       symbols: List[str], 
                                       start_date: str, 
                                       end_date: str,
                                       save_to_storage: bool = True) -> Dict:
        """
        Fetch news data for multiple symbols.
        
        Args:
            symbols: List of stock symbols
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            save_to_storage: Whether to save data to storage
            
        Returns:
            Dictionary with overall results and statistics
        """
        results = {
            'total_symbols': len(symbols),
            'successful_fetches': 0,
            'failed_fetches': 0,
            'total_news_items': 0,
            'detailed_results': [],
            'errors': []
        }
        
        self.logger.info(f"Fetching news for {len(symbols)} symbols from {start_date} to {end_date}")
        
        for symbol in symbols:
            try:
                if save_to_storage:
                    result = self.fetch_and_save_news(symbol, start_date, end_date)
                else:
                    news_data = self.fetch_news_for_symbol(symbol, start_date, end_date)
                    result = {
                        'symbol': symbol,
                        'success': True,
                        'news_count': len(news_data),
                        'date_range': f"{start_date} to {end_date}",
                        'news_data': news_data
                    }
                
                results['detailed_results'].append(result)
                
                if result['success']:
                    results['successful_fetches'] += 1
                    results['total_news_items'] += result['news_count']
                else:
                    results['failed_fetches'] += 1
                    if result.get('error'):
                        results['errors'].append({
                            'symbol': symbol,
                            'error': result['error']
                        })
                        
            except Exception as e:
                error_msg = f"Unexpected error processing {symbol}: {str(e)}"
                self.logger.error(error_msg)
                results['failed_fetches'] += 1
                results['errors'].append({
                    'symbol': symbol,
                    'error': str(e)
                })
        
        # Log summary
        success_rate = (results['successful_fetches'] / results['total_symbols'] * 100) if results['total_symbols'] > 0 else 0
        self.logger.info(f"News fetch completed: {results['successful_fetches']}/{results['total_symbols']} successful ({success_rate:.1f}%)")
        self.logger.info(f"Total news items fetched: {results['total_news_items']}")
        
        if results['errors']:
            self.logger.warning(f"Errors encountered for {len(results['errors'])} symbols")
        
        return results
    
    def associate_news_with_stock_data(self, 
                                      symbol: str, 
                                      stock_data: pd.DataFrame,
                                      news_data: Optional[List[Dict]] = None) -> pd.DataFrame:
        """
        Associate news data with stock trading dates.
        
        Args:
            symbol: Stock symbol
            stock_data: DataFrame with stock price data
            news_data: List of news dictionaries (if None, loads from storage)
            
        Returns:
            DataFrame with stock data and associated news counts/sentiment placeholders
        """
        try:
            # Load news data if not provided
            if news_data is None:
                news_data = self.data_storage.load_news_data(symbol)
            
            if not news_data or stock_data.empty:
                self.logger.warning(f"No news or stock data available for association for {symbol}")
                # Return stock data with neutral sentiment columns
                stock_data_copy = stock_data.copy()
                stock_data_copy['news_count'] = 0
                stock_data_copy['sentiment_score'] = 0.0  # Neutral sentiment
                return stock_data_copy
            
            # Convert news data to DataFrame for easier processing
            news_df = pd.DataFrame(news_data)
            
            # Convert publish_time to date for association
            if 'publish_time' in news_df.columns:
                news_df['news_date'] = pd.to_datetime(news_df['publish_time'], unit='s').dt.date
            elif 'publish_date' in news_df.columns:
                news_df['news_date'] = pd.to_datetime(news_df['publish_date']).dt.date
            else:
                self.logger.warning(f"No valid date column found in news data for {symbol}")
                stock_data_copy = stock_data.copy()
                stock_data_copy['news_count'] = 0
                stock_data_copy['sentiment_score'] = 0.0
                return stock_data_copy
            
            # Prepare stock data
            stock_data_copy = stock_data.copy()
            
            # Ensure date column is datetime and extract date part
            if 'date' in stock_data_copy.columns:
                stock_data_copy['stock_date'] = pd.to_datetime(stock_data_copy['date']).dt.date
            else:
                self.logger.error(f"No date column found in stock data for {symbol}")
                stock_data_copy['news_count'] = 0
                stock_data_copy['sentiment_score'] = 0.0
                return stock_data_copy
            
            # Count news items per date
            news_counts = news_df.groupby('news_date').size().reset_index(name='news_count')
            news_counts['stock_date'] = news_counts['news_date']
            
            # Merge with stock data
            stock_data_copy = stock_data_copy.merge(
                news_counts[['stock_date', 'news_count']], 
                on='stock_date', 
                how='left'
            )
            
            # Fill missing news counts with 0
            stock_data_copy['news_count'] = stock_data_copy['news_count'].fillna(0).astype(int)
            
            # Add placeholder sentiment score (will be calculated by sentiment analyzer)
            stock_data_copy['sentiment_score'] = 0.0  # Neutral sentiment as default
            
            # Clean up temporary columns
            stock_data_copy = stock_data_copy.drop('stock_date', axis=1)
            
            self.logger.info(f"Associated news data with stock data for {symbol}: {len(stock_data_copy)} records")
            return stock_data_copy
            
        except Exception as e:
            self.logger.error(f"Failed to associate news with stock data for {symbol}: {str(e)}")
            # Return original stock data with neutral sentiment
            stock_data_copy = stock_data.copy()
            stock_data_copy['news_count'] = 0
            stock_data_copy['sentiment_score'] = 0.0
            return stock_data_copy
    
    def get_news_summary_for_date_range(self, 
                                       symbol: str, 
                                       start_date: str, 
                                       end_date: str) -> Dict:
        """
        Get summary statistics for news data in a date range.
        
        Args:
            symbol: Stock symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            Dictionary with news summary statistics
        """
        try:
            # Load existing news data
            news_data = self.data_storage.load_news_data(symbol)
            
            if not news_data:
                return {
                    'symbol': symbol,
                    'date_range': f"{start_date} to {end_date}",
                    'total_news_items': 0,
                    'unique_publishers': 0,
                    'date_coverage': 0,
                    'avg_news_per_day': 0.0
                }
            
            # Filter by date range
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            
            filtered_news = []
            for news_item in news_data:
                if 'publish_time' in news_item:
                    news_dt = datetime.fromtimestamp(news_item['publish_time'])
                    if start_dt <= news_dt <= end_dt:
                        filtered_news.append(news_item)
            
            if not filtered_news:
                return {
                    'symbol': symbol,
                    'date_range': f"{start_date} to {end_date}",
                    'total_news_items': 0,
                    'unique_publishers': 0,
                    'date_coverage': 0,
                    'avg_news_per_day': 0.0
                }
            
            # Calculate statistics
            publishers = set()
            dates = set()
            
            for news_item in filtered_news:
                if 'publisher' in news_item and news_item['publisher']:
                    publishers.add(news_item['publisher'])
                if 'publish_time' in news_item:
                    news_date = datetime.fromtimestamp(news_item['publish_time']).date()
                    dates.add(news_date)
            
            total_days = (end_dt - start_dt).days + 1
            avg_news_per_day = len(filtered_news) / total_days if total_days > 0 else 0
            
            summary = {
                'symbol': symbol,
                'date_range': f"{start_date} to {end_date}",
                'total_news_items': len(filtered_news),
                'unique_publishers': len(publishers),
                'date_coverage': len(dates),
                'total_days_in_range': total_days,
                'avg_news_per_day': round(avg_news_per_day, 2)
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Failed to get news summary for {symbol}: {str(e)}")
            return {
                'symbol': symbol,
                'date_range': f"{start_date} to {end_date}",
                'error': str(e)
            }
    
    def _validate_date_range(self, start_date: str, end_date: str):
        """
        Validate date range parameters.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Raises:
            ValueError: For invalid date formats or ranges
        """
        try:
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            
            if start_dt >= end_dt:
                raise ValueError("Start date must be before end date")
                
            if end_dt > datetime.now():
                self.logger.warning("End date is in the future, will be adjusted to current date")
                
        except ValueError as e:
            if "time data" in str(e):
                raise ValueError("Invalid date format. Use YYYY-MM-DD format")
            else:
                raise
    
    def _process_news_data(self, 
                          news_data: List[Dict], 
                          symbol: str, 
                          start_date: str, 
                          end_date: str) -> List[Dict]:
        """
        Additional processing and validation of news data.
        
        Args:
            news_data: Raw news data from YFinance client
            symbol: Stock symbol
            start_date: Start date string
            end_date: End date string
            
        Returns:
            Processed and validated news data
        """
        processed_news = []
        
        for news_item in news_data:
            try:
                # Validate required fields
                if not news_item.get('title'):
                    continue
                
                # Ensure all required fields are present
                processed_item = {
                    'symbol': news_item.get('symbol', symbol.replace('.NS', '').replace('.BO', '')),
                    'title': news_item.get('title', '').strip(),
                    'summary': news_item.get('summary', '').strip(),
                    'publisher': news_item.get('publisher', 'Unknown'),
                    'publish_time': news_item.get('publish_time', 0),
                    'url': news_item.get('url', '')
                }
                
                # Additional validation
                if processed_item['publish_time'] > 0 and processed_item['title']:
                    processed_news.append(processed_item)
                    
            except Exception as e:
                self.logger.warning(f"Error processing news item for {symbol}: {str(e)}")
                continue
        
        return processed_news