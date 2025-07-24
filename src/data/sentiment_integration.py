"""
Sentiment integration system for associating sentiment analysis with trading data.
Combines news data fetching, sentiment analysis, and data storage.
"""

import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from src.data.news_data_fetcher import NewsDataFetcher
from src.data.sentiment_analyzer import SentimentAnalyzer
from src.data.data_storage import DataStorage
from src.data.yfinance_client import YFinanceClient


class SentimentIntegration:
    """
    Integrates sentiment analysis with trading data.
    
    Features:
    - Combines news fetching and sentiment analysis
    - Associates sentiment scores with stock price data
    - Handles missing news data with neutral sentiment
    - Stores integrated data alongside price and indicator data
    - Provides aggregated sentiment metrics for trading decisions
    """
    
    def __init__(self,
                 news_fetcher: Optional[NewsDataFetcher] = None,
                 sentiment_analyzer: Optional[SentimentAnalyzer] = None,
                 data_storage: Optional[DataStorage] = None):
        """
        Initialize sentiment integration system.
        
        Args:
            news_fetcher: NewsDataFetcher instance for retrieving news
            sentiment_analyzer: SentimentAnalyzer instance for sentiment analysis
            data_storage: DataStorage instance for saving integrated data
        """
        self.news_fetcher = news_fetcher or NewsDataFetcher()
        self.sentiment_analyzer = sentiment_analyzer or SentimentAnalyzer()
        self.data_storage = data_storage or DataStorage()
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("SentimentIntegration initialized successfully")
    
    def process_sentiment_for_symbol(self, 
                                   symbol: str, 
                                   start_date: str, 
                                   end_date: str,
                                   stock_data: Optional[pd.DataFrame] = None) -> Dict:
        """
        Process sentiment analysis for a single symbol and integrate with stock data.
        
        Args:
            symbol: Stock symbol to process
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            stock_data: Optional stock data DataFrame (loads from storage if None)
            
        Returns:
            Dictionary with processing results and integrated data
        """
        result = {
            'symbol': symbol,
            'success': False,
            'news_count': 0,
            'sentiment_processed': False,
            'data_integrated': False,
            'error': None,
            'sentiment_summary': {}
        }
        
        try:
            self.logger.info(f"Processing sentiment for {symbol} from {start_date} to {end_date}")
            
            # Step 1: Fetch news data
            news_data = self.news_fetcher.fetch_news_for_symbol(symbol, start_date, end_date)
            result['news_count'] = len(news_data)
            
            if not news_data:
                self.logger.warning(f"No news data found for {symbol}, using neutral sentiment")
                news_data = []  # Will be handled as neutral sentiment
            
            # Step 2: Analyze sentiment for news headlines
            if news_data:
                news_with_sentiment = self.sentiment_analyzer.analyze_news_headlines(news_data)
                result['sentiment_processed'] = True
                
                # Calculate sentiment summary
                result['sentiment_summary'] = self._calculate_sentiment_summary(news_with_sentiment)
                
                # Save news data with sentiment scores
                if self.data_storage.save_news_data(symbol, news_with_sentiment):
                    self.logger.info(f"Saved {len(news_with_sentiment)} news items with sentiment for {symbol}")
                else:
                    self.logger.warning(f"Failed to save news data for {symbol}")
            else:
                news_with_sentiment = []
                result['sentiment_summary'] = {
                    'avg_sentiment': 0.0,
                    'sentiment_count': 0,
                    'positive_count': 0,
                    'negative_count': 0,
                    'neutral_count': 0
                }
            
            # Step 3: Load stock data if not provided
            if stock_data is None:
                stock_data = self.data_storage.load_stock_data(symbol)
                
            if stock_data.empty:
                self.logger.warning(f"No stock data available for {symbol}")
                result['error'] = "No stock data available"
                return result
            
            # Step 4: Integrate sentiment with stock data
            integrated_data = self._integrate_sentiment_with_stock_data(
                symbol, stock_data, news_with_sentiment, start_date, end_date
            )
            
            if not integrated_data.empty:
                # Save integrated data (this will include sentiment scores)
                if self.data_storage.save_stock_data(symbol, integrated_data):
                    result['data_integrated'] = True
                    result['success'] = True
                    self.logger.info(f"Successfully integrated sentiment data for {symbol}")
                else:
                    result['error'] = "Failed to save integrated data"
            else:
                result['error'] = "Failed to integrate sentiment with stock data"
            
        except Exception as e:
            error_msg = f"Error processing sentiment for {symbol}: {str(e)}"
            self.logger.error(error_msg)
            result['error'] = str(e)
        
        return result
    
    def process_sentiment_for_multiple_symbols(self, 
                                             symbols: List[str], 
                                             start_date: str, 
                                             end_date: str) -> Dict:
        """
        Process sentiment analysis for multiple symbols.
        
        Args:
            symbols: List of stock symbols to process
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            Dictionary with overall processing results
        """
        results = {
            'total_symbols': len(symbols),
            'successful_processing': 0,
            'failed_processing': 0,
            'total_news_items': 0,
            'detailed_results': [],
            'errors': []
        }
        
        self.logger.info(f"Processing sentiment for {len(symbols)} symbols from {start_date} to {end_date}")
        
        for symbol in symbols:
            try:
                result = self.process_sentiment_for_symbol(symbol, start_date, end_date)
                results['detailed_results'].append(result)
                
                if result['success']:
                    results['successful_processing'] += 1
                    results['total_news_items'] += result['news_count']
                else:
                    results['failed_processing'] += 1
                    if result.get('error'):
                        results['errors'].append({
                            'symbol': symbol,
                            'error': result['error']
                        })
                        
            except Exception as e:
                error_msg = f"Unexpected error processing {symbol}: {str(e)}"
                self.logger.error(error_msg)
                results['failed_processing'] += 1
                results['errors'].append({
                    'symbol': symbol,
                    'error': str(e)
                })
        
        # Log summary
        success_rate = (results['successful_processing'] / results['total_symbols'] * 100) if results['total_symbols'] > 0 else 0
        self.logger.info(f"Sentiment processing completed: {results['successful_processing']}/{results['total_symbols']} successful ({success_rate:.1f}%)")
        self.logger.info(f"Total news items processed: {results['total_news_items']}")
        
        if results['errors']:
            self.logger.warning(f"Errors encountered for {len(results['errors'])} symbols")
        
        return results
    
    def get_sentiment_for_trading_date(self, 
                                     symbol: str, 
                                     trading_date: str,
                                     lookback_days: int = 1) -> Dict:
        """
        Get sentiment score for a specific trading date.
        
        Args:
            symbol: Stock symbol
            trading_date: Trading date in YYYY-MM-DD format
            lookback_days: Number of days to look back for news (default: 1)
            
        Returns:
            Dictionary with sentiment information for the trading date
        """
        try:
            # Calculate date range for news lookup
            trade_dt = datetime.strptime(trading_date, '%Y-%m-%d')
            start_dt = trade_dt - timedelta(days=lookback_days)
            start_date = start_dt.strftime('%Y-%m-%d')
            
            # Load news data
            news_data = self.data_storage.load_news_data(symbol)
            
            if not news_data:
                return {
                    'symbol': symbol,
                    'trading_date': trading_date,
                    'sentiment_score': 0.0,
                    'news_count': 0,
                    'sentiment_label': 'Neutral'
                }
            
            # Filter news for the date range
            relevant_news = []
            for news_item in news_data:
                if 'publish_time' in news_item:
                    news_dt = datetime.fromtimestamp(news_item['publish_time'])
                    if start_dt <= news_dt <= trade_dt:
                        relevant_news.append(news_item)
            
            if not relevant_news:
                return {
                    'symbol': symbol,
                    'trading_date': trading_date,
                    'sentiment_score': 0.0,
                    'news_count': 0,
                    'sentiment_label': 'Neutral'
                }
            
            # Calculate aggregated sentiment
            sentiment_score = self.sentiment_analyzer.get_aggregated_sentiment(
                relevant_news, aggregation_method="weighted_mean"
            )
            
            return {
                'symbol': symbol,
                'trading_date': trading_date,
                'sentiment_score': sentiment_score,
                'news_count': len(relevant_news),
                'sentiment_label': self._get_sentiment_label(sentiment_score),
                'news_items': relevant_news
            }
            
        except Exception as e:
            self.logger.error(f"Error getting sentiment for {symbol} on {trading_date}: {str(e)}")
            return {
                'symbol': symbol,
                'trading_date': trading_date,
                'sentiment_score': 0.0,
                'news_count': 0,
                'sentiment_label': 'Neutral',
                'error': str(e)
            }
    
    def update_stock_data_with_sentiment(self, 
                                       symbol: str, 
                                       stock_data: pd.DataFrame) -> pd.DataFrame:
        """
        Update stock data DataFrame with sentiment scores.
        
        Args:
            symbol: Stock symbol
            stock_data: Stock data DataFrame
            
        Returns:
            Updated DataFrame with sentiment scores
        """
        try:
            if stock_data.empty:
                return stock_data
            
            # Load news data
            news_data = self.data_storage.load_news_data(symbol)
            
            # Create a copy of stock data
            updated_data = stock_data.copy()
            
            # Initialize sentiment columns
            updated_data['sentiment_score'] = 0.0
            updated_data['news_count'] = 0
            
            if not news_data:
                self.logger.warning(f"No news data available for {symbol}, using neutral sentiment")
                return updated_data
            
            # Process each trading date
            for idx, row in updated_data.iterrows():
                if 'date' in row:
                    trading_date = pd.to_datetime(row['date']).strftime('%Y-%m-%d')
                    sentiment_info = self.get_sentiment_for_trading_date(symbol, trading_date)
                    
                    updated_data.at[idx, 'sentiment_score'] = sentiment_info['sentiment_score']
                    updated_data.at[idx, 'news_count'] = sentiment_info['news_count']
            
            self.logger.info(f"Updated {len(updated_data)} stock records with sentiment for {symbol}")
            return updated_data
            
        except Exception as e:
            self.logger.error(f"Error updating stock data with sentiment for {symbol}: {str(e)}")
            # Return original data with neutral sentiment
            stock_data_copy = stock_data.copy()
            stock_data_copy['sentiment_score'] = 0.0
            stock_data_copy['news_count'] = 0
            return stock_data_copy
    
    def _integrate_sentiment_with_stock_data(self, 
                                           symbol: str, 
                                           stock_data: pd.DataFrame, 
                                           news_with_sentiment: List[Dict],
                                           start_date: str, 
                                           end_date: str) -> pd.DataFrame:
        """
        Integrate sentiment data with stock price data.
        
        Args:
            symbol: Stock symbol
            stock_data: Stock price data DataFrame
            news_with_sentiment: News data with sentiment scores
            start_date: Start date for filtering
            end_date: End date for filtering
            
        Returns:
            Integrated DataFrame with sentiment scores
        """
        try:
            if stock_data.empty:
                return pd.DataFrame()
            
            # Filter stock data by date range
            stock_data_copy = stock_data.copy()
            
            if 'date' in stock_data_copy.columns:
                # Convert dates to naive datetime objects to avoid timezone issues
                stock_data_copy['date'] = pd.to_datetime(stock_data_copy['date']).dt.tz_localize(None)
                start_dt = pd.to_datetime(start_date).tz_localize(None)
                end_dt = pd.to_datetime(end_date).tz_localize(None)
                
                # Filter by date range
                mask = (stock_data_copy['date'] >= start_dt) & (stock_data_copy['date'] <= end_dt)
                stock_data_filtered = stock_data_copy[mask].copy()
            else:
                stock_data_filtered = stock_data_copy.copy()
            
            if stock_data_filtered.empty:
                self.logger.warning(f"No stock data in date range for {symbol}")
                return pd.DataFrame()
            
            # Initialize sentiment columns
            stock_data_filtered['sentiment_score'] = 0.0
            stock_data_filtered['news_count'] = 0
            
            if not news_with_sentiment:
                self.logger.info(f"No news data available for {symbol}, using neutral sentiment")
                return stock_data_filtered
            
            # Create news DataFrame for easier processing
            news_df = pd.DataFrame(news_with_sentiment)
            
            if 'publish_time' in news_df.columns:
                # Convert to naive datetime to avoid timezone issues
                news_df['news_date'] = pd.to_datetime(news_df['publish_time'], unit='s').dt.tz_localize(None).dt.date
            else:
                self.logger.warning(f"No publish_time in news data for {symbol}")
                return stock_data_filtered
            
            # Associate sentiment with each trading date
            for idx, row in stock_data_filtered.iterrows():
                if 'date' in row:
                    # Ensure date is naive datetime to avoid timezone comparison issues
                    trading_date = pd.to_datetime(row['date']).tz_localize(None).date()
                    
                    # Find news for this date (and previous day for after-hours news)
                    prev_date = trading_date - timedelta(days=1)
                    
                    # Use string comparison if needed to avoid timezone issues
                    trading_date_str = trading_date.isoformat()
                    prev_date_str = prev_date.isoformat()
                    
                    # Convert news dates to strings for comparison
                    if isinstance(news_df['news_date'].iloc[0], str):
                        relevant_news = news_df[
                            (news_df['news_date'] == trading_date_str) | 
                            (news_df['news_date'] == prev_date_str)
                        ]
                    else:
                        relevant_news = news_df[
                            (news_df['news_date'] == trading_date) | 
                            (news_df['news_date'] == prev_date)
                        ]
                    
                    if not relevant_news.empty:
                        # Calculate aggregated sentiment for this date
                        sentiment_scores = relevant_news['sentiment_score'].tolist()
                        if sentiment_scores:
                            # Use weighted average (more recent news has higher weight)
                            weights = np.linspace(0.5, 1.0, len(sentiment_scores))
                            avg_sentiment = np.average(sentiment_scores, weights=weights)
                            
                            stock_data_filtered.at[idx, 'sentiment_score'] = avg_sentiment
                            stock_data_filtered.at[idx, 'news_count'] = len(sentiment_scores)
            
            self.logger.info(f"Integrated sentiment data with {len(stock_data_filtered)} stock records for {symbol}")
            return stock_data_filtered
            
        except Exception as e:
            self.logger.error(f"Error integrating sentiment with stock data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def _calculate_sentiment_summary(self, news_with_sentiment: List[Dict]) -> Dict:
        """
        Calculate summary statistics for sentiment data.
        
        Args:
            news_with_sentiment: List of news items with sentiment scores
            
        Returns:
            Dictionary with sentiment summary statistics
        """
        if not news_with_sentiment:
            return {
                'avg_sentiment': 0.0,
                'sentiment_count': 0,
                'positive_count': 0,
                'negative_count': 0,
                'neutral_count': 0
            }
        
        sentiment_scores = [item.get('sentiment_score', 0.0) for item in news_with_sentiment]
        sentiment_scores = [score for score in sentiment_scores if isinstance(score, (int, float))]
        
        if not sentiment_scores:
            return {
                'avg_sentiment': 0.0,
                'sentiment_count': 0,
                'positive_count': 0,
                'negative_count': 0,
                'neutral_count': 0
            }
        
        # Calculate counts
        positive_count = sum(1 for score in sentiment_scores if score > 0.1)
        negative_count = sum(1 for score in sentiment_scores if score < -0.1)
        neutral_count = len(sentiment_scores) - positive_count - negative_count
        
        return {
            'avg_sentiment': float(np.mean(sentiment_scores)),
            'sentiment_count': len(sentiment_scores),
            'positive_count': positive_count,
            'negative_count': negative_count,
            'neutral_count': neutral_count,
            'sentiment_std': float(np.std(sentiment_scores)) if len(sentiment_scores) > 1 else 0.0
        }
    
    def _get_sentiment_label(self, score: float) -> str:
        """
        Convert sentiment score to human-readable label.
        
        Args:
            score: Sentiment score between -1 and +1
            
        Returns:
            Sentiment label
        """
        if score > 0.1:
            return "Positive"
        elif score < -0.1:
            return "Negative"
        else:
            return "Neutral"
    
    def integrate_sentiment_for_all_stocks(self, symbols: List[str]) -> bool:
        """
        Integrate sentiment analysis for all stocks in the provided list.
        
        Args:
            symbols: List of stock symbols
            
        Returns:
            True if successful for at least one symbol, False otherwise
        """
        try:
            self.logger.info(f"Integrating sentiment for {len(symbols)} stocks")
            
            # Calculate date range (last 30 days by default)
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            
            results = self.process_sentiment_for_multiple_symbols(symbols, start_date, end_date)
            
            success_count = results['successful_processing']
            if success_count > 0:
                self.logger.info(f"Successfully integrated sentiment for {success_count}/{len(symbols)} stocks")
                return True
            else:
                self.logger.error(f"Failed to integrate sentiment for any of the {len(symbols)} stocks")
                return False
                
        except Exception as e:
            self.logger.error(f"Error integrating sentiment for stocks: {str(e)}")
            return False
    
    def get_integration_stats(self) -> Dict:
        """
        Get statistics about the sentiment integration system.
        
        Returns:
            Dictionary with system statistics
        """
        try:
            storage_stats = self.data_storage.get_storage_stats()
            model_info = self.sentiment_analyzer.get_model_info()
            
            return {
                'storage_stats': storage_stats,
                'sentiment_model_info': model_info,
                'components_loaded': {
                    'news_fetcher': self.news_fetcher is not None,
                    'sentiment_analyzer': self.sentiment_analyzer is not None,
                    'data_storage': self.data_storage is not None
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error getting integration stats: {str(e)}")
            return {'error': str(e)}