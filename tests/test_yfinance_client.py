"""
Unit tests for YFinanceClient class.
Tests API client functionality, error handling, and data validation.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
from datetime import datetime, timedelta
import time
import logging

from src.data.yfinance_client import YFinanceClient


class TestYFinanceClient(unittest.TestCase):
    """Test cases for YFinanceClient class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.client = YFinanceClient(rate_limit_delay=0.01, max_retries=2)
        
        # Sample stock data for testing
        self.sample_stock_data = pd.DataFrame({
            'Open': [100.0, 101.0, 102.0],
            'High': [105.0, 106.0, 107.0],
            'Low': [99.0, 100.0, 101.0],
            'Close': [104.0, 105.0, 106.0],
            'Volume': [1000000, 1100000, 1200000]
        }, index=pd.date_range('2023-01-01', periods=3))
        
        # Sample news data for testing
        self.sample_news_data = [
            {
                'title': 'Company reports strong earnings',
                'summary': 'Quarterly results exceed expectations',
                'publisher': 'Financial Times',
                'providerPublishTime': int(datetime(2023, 1, 1).timestamp()),
                'link': 'https://example.com/news1'
            },
            {
                'title': 'Market outlook positive',
                'summary': 'Analysts upgrade rating',
                'publisher': 'Reuters',
                'providerPublishTime': int(datetime(2023, 1, 2).timestamp()),
                'link': 'https://example.com/news2'
            }
        ]
    
    def test_initialization(self):
        """Test YFinanceClient initialization."""
        client = YFinanceClient(rate_limit_delay=0.5, max_retries=5)
        self.assertEqual(client.rate_limit_delay, 0.5)
        self.assertEqual(client.max_retries, 5)
        self.assertEqual(client.last_request_time, 0)
        self.assertIsInstance(client.logger, logging.Logger)
    
    def test_rate_limiting(self):
        """Test rate limiting functionality."""
        start_time = time.time()
        
        # First call should not delay
        self.client._rate_limit()
        first_call_time = time.time()
        
        # Second call should delay
        self.client._rate_limit()
        second_call_time = time.time()
        
        # Check that delay was applied
        time_diff = second_call_time - first_call_time
        self.assertGreaterEqual(time_diff, self.client.rate_limit_delay * 0.9)  # Allow small tolerance
    
    def test_validate_symbol(self):
        """Test symbol validation and formatting."""
        # Test basic symbol
        self.assertEqual(self.client._validate_symbol('RELIANCE'), 'RELIANCE.NS')
        
        # Test symbol with .NS suffix
        self.assertEqual(self.client._validate_symbol('TCS.NS'), 'TCS.NS')
        
        # Test symbol with .BO suffix
        self.assertEqual(self.client._validate_symbol('INFY.BO'), 'INFY.BO')
        
        # Test lowercase symbol
        self.assertEqual(self.client._validate_symbol('hdfc'), 'HDFC.NS')
        
        # Test symbol with whitespace
        self.assertEqual(self.client._validate_symbol(' WIPRO '), 'WIPRO.NS')
        
        # Test invalid symbols
        with self.assertRaises(ValueError):
            self.client._validate_symbol('')
        
        with self.assertRaises(ValueError):
            self.client._validate_symbol(None)
    
    def test_validate_dates(self):
        """Test date validation."""
        # Test valid dates
        start, end = self.client._validate_dates('2023-01-01', '2023-12-31')
        self.assertEqual(start, '2023-01-01')
        self.assertEqual(end, '2023-12-31')
        
        # Test invalid date format
        with self.assertRaises(ValueError):
            self.client._validate_dates('01-01-2023', '31-12-2023')
        
        # Test start date after end date
        with self.assertRaises(ValueError):
            self.client._validate_dates('2023-12-31', '2023-01-01')
        
        # Test future end date (should be adjusted)
        future_date = (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d')
        start, end = self.client._validate_dates('2023-01-01', future_date)
        self.assertEqual(start, '2023-01-01')
        self.assertEqual(end, datetime.now().strftime('%Y-%m-%d'))
    
    @patch('src.data.yfinance_client.yf.Ticker')
    def test_fetch_stock_data_success(self, mock_ticker):
        """Test successful stock data fetching."""
        # Mock yfinance ticker
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = self.sample_stock_data
        mock_ticker.return_value = mock_ticker_instance
        
        # Fetch data
        result = self.client.fetch_stock_data('RELIANCE', '2023-01-01', '2023-01-03')
        
        # Verify results
        self.assertIsInstance(result, pd.DataFrame)
        self.assertFalse(result.empty)
        self.assertIn('symbol', result.columns)
        self.assertEqual(result['symbol'].iloc[0], 'RELIANCE')
        
        # Verify API was called correctly
        mock_ticker.assert_called_once_with('RELIANCE.NS')
        mock_ticker_instance.history.assert_called_once()
    
    @patch('src.data.yfinance_client.yf.Ticker')
    def test_fetch_stock_data_empty_response(self, mock_ticker):
        """Test handling of empty stock data response."""
        # Mock empty response
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = pd.DataFrame()
        mock_ticker.return_value = mock_ticker_instance
        
        # Fetch data
        result = self.client.fetch_stock_data('INVALID', '2023-01-01', '2023-01-03')
        
        # Verify empty DataFrame is returned
        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue(result.empty)
    
    @patch('src.data.yfinance_client.yf.Ticker')
    def test_fetch_news_data_success(self, mock_ticker):
        """Test successful news data fetching."""
        # Mock yfinance ticker
        mock_ticker_instance = Mock()
        mock_ticker_instance.news = self.sample_news_data
        mock_ticker.return_value = mock_ticker_instance
        
        # Fetch data
        result = self.client.fetch_news_data('RELIANCE', '2023-01-01', '2023-01-03')
        
        # Verify results
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
        
        # Check structure of first news item
        news_item = result[0]
        self.assertIn('symbol', news_item)
        self.assertIn('title', news_item)
        self.assertIn('summary', news_item)
        self.assertIn('publisher', news_item)
        self.assertIn('publish_time', news_item)
        self.assertIn('url', news_item)
        
        # Verify API was called correctly
        mock_ticker.assert_called_once_with('RELIANCE.NS')
    
    @patch('src.data.yfinance_client.yf.Ticker')
    def test_fetch_news_data_empty_response(self, mock_ticker):
        """Test handling of empty news data response."""
        # Mock empty response
        mock_ticker_instance = Mock()
        mock_ticker_instance.news = []
        mock_ticker.return_value = mock_ticker_instance
        
        # Fetch data
        result = self.client.fetch_news_data('INVALID', '2023-01-01', '2023-01-03')
        
        # Verify empty list is returned
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 0)
    
    def test_clean_stock_data(self):
        """Test stock data cleaning functionality."""
        # Test with valid data
        cleaned_data = self.client._clean_stock_data(self.sample_stock_data, 'RELIANCE.NS')
        
        # Check column names are standardized
        expected_columns = ['open', 'high', 'low', 'close', 'volume', 'symbol', 'date']
        for col in expected_columns:
            self.assertIn(col, cleaned_data.columns)
        
        # Check symbol is added correctly
        self.assertEqual(cleaned_data['symbol'].iloc[0], 'RELIANCE')
    
    def test_clean_stock_data_missing_columns(self):
        """Test handling of missing columns in stock data."""
        # Create data with missing columns
        incomplete_data = pd.DataFrame({
            'Open': [100.0, 101.0],
            'High': [105.0, 106.0]
            # Missing Low, Close, Volume
        })
        
        result = self.client._clean_stock_data(incomplete_data, 'TEST.NS')
        self.assertTrue(result.empty)
    
    def test_validate_ohlcv_data(self):
        """Test OHLCV data validation."""
        # Test valid data
        valid_data = pd.DataFrame({
            'open': [100.0, 101.0],
            'high': [105.0, 106.0],
            'low': [99.0, 100.0],
            'close': [104.0, 105.0],
            'volume': [1000000, 1100000]
        })
        self.assertTrue(self.client._validate_ohlcv_data(valid_data))
        
        # Test invalid data - negative values
        invalid_data = valid_data.copy()
        invalid_data.loc[0, 'close'] = -10.0
        self.assertFalse(self.client._validate_ohlcv_data(invalid_data))
        
        # Test invalid data - high < low
        invalid_data = valid_data.copy()
        invalid_data.loc[0, 'high'] = 90.0  # Less than low
        self.assertFalse(self.client._validate_ohlcv_data(invalid_data))
        
        # Test invalid data - open outside range
        invalid_data = valid_data.copy()
        invalid_data.loc[0, 'open'] = 110.0  # Greater than high
        self.assertFalse(self.client._validate_ohlcv_data(invalid_data))
    
    def test_filter_news_by_date(self):
        """Test news filtering by date range."""
        filtered_news = self.client._filter_news_by_date(
            self.sample_news_data, '2023-01-01', '2023-01-01'
        )
        
        # Should only include news from 2023-01-01
        self.assertEqual(len(filtered_news), 1)
        self.assertEqual(filtered_news[0]['title'], 'Company reports strong earnings')
    
    def test_clean_news_data(self):
        """Test news data cleaning."""
        cleaned_news = self.client._clean_news_data(self.sample_news_data, 'RELIANCE.NS')
        
        # Check structure
        self.assertEqual(len(cleaned_news), 2)
        
        news_item = cleaned_news[0]
        self.assertEqual(news_item['symbol'], 'RELIANCE')
        self.assertEqual(news_item['title'], 'Company reports strong earnings')
        self.assertEqual(news_item['publisher'], 'Financial Times')
    
    @patch('src.data.yfinance_client.yf.Ticker')
    def test_retry_mechanism(self, mock_ticker):
        """Test retry mechanism with exponential backoff."""
        # Mock ticker to raise exception first time, succeed second time
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.side_effect = [
            Exception("Network error"),
            self.sample_stock_data
        ]
        mock_ticker.return_value = mock_ticker_instance
        
        # Should succeed after retry
        result = self.client.fetch_stock_data('RELIANCE', '2023-01-01', '2023-01-03')
        self.assertFalse(result.empty)
        
        # Verify retry was attempted
        self.assertEqual(mock_ticker_instance.history.call_count, 2)
    
    @patch('src.data.yfinance_client.yf.Ticker')
    def test_max_retries_exceeded(self, mock_ticker):
        """Test behavior when max retries are exceeded."""
        # Mock ticker to always raise exception
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.side_effect = Exception("Persistent error")
        mock_ticker.return_value = mock_ticker_instance
        
        # Should raise exception after max retries
        with self.assertRaises(Exception):
            self.client.fetch_stock_data('RELIANCE', '2023-01-01', '2023-01-03')
        
        # Verify max retries were attempted
        self.assertEqual(mock_ticker_instance.history.call_count, self.client.max_retries)
    
    @patch('src.data.yfinance_client.yf.Ticker')
    def test_get_ticker_info(self, mock_ticker):
        """Test ticker info retrieval."""
        # Mock ticker info
        mock_info = {
            'longName': 'Reliance Industries Limited',
            'sector': 'Energy',
            'marketCap': 1000000000
        }
        mock_ticker_instance = Mock()
        mock_ticker_instance.info = mock_info
        mock_ticker.return_value = mock_ticker_instance
        
        # Get ticker info
        result = self.client.get_ticker_info('RELIANCE')
        
        # Verify results
        self.assertEqual(result, mock_info)
        mock_ticker.assert_called_once_with('RELIANCE.NS')


if __name__ == '__main__':
    # Configure logging for tests
    logging.basicConfig(level=logging.WARNING)
    unittest.main()