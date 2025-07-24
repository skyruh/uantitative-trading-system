"""
Unit tests for NewsDataFetcher class.
Tests news data retrieval, date filtering, and error handling.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
from datetime import datetime, timedelta
import tempfile
import shutil
from pathlib import Path

from src.data.news_data_fetcher import NewsDataFetcher
from src.data.yfinance_client import YFinanceClient
from src.data.data_storage import DataStorage


class TestNewsDataFetcher(unittest.TestCase):
    """Test cases for NewsDataFetcher class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for test data
        self.test_dir = tempfile.mkdtemp()
        
        # Mock dependencies
        self.mock_client = Mock(spec=YFinanceClient)
        self.mock_storage = Mock(spec=DataStorage)
        
        # Create fetcher instance
        self.fetcher = NewsDataFetcher(
            data_client=self.mock_client,
            data_storage=self.mock_storage
        )
        
        # Sample news data for testing
        self.sample_news_data = [
            {
                'symbol': 'RELIANCE',
                'title': 'Reliance Industries reports strong Q4 results',
                'summary': 'Company shows growth in all segments',
                'publisher': 'Economic Times',
                'publish_time': int(datetime(2024, 1, 15).timestamp()),
                'url': 'https://example.com/news1'
            },
            {
                'symbol': 'RELIANCE',
                'title': 'Reliance announces new investment in green energy',
                'summary': 'Major investment in renewable energy sector',
                'publisher': 'Business Standard',
                'publish_time': int(datetime(2024, 1, 16).timestamp()),
                'url': 'https://example.com/news2'
            },
            {
                'symbol': 'RELIANCE',
                'title': 'Market outlook positive for Reliance',
                'summary': 'Analysts upgrade rating',
                'publisher': 'Mint',
                'publish_time': int(datetime(2024, 1, 17).timestamp()),
                'url': 'https://example.com/news3'
            }
        ]
        
        # Sample stock data for association tests
        self.sample_stock_data = pd.DataFrame({
            'date': pd.to_datetime(['2024-01-15', '2024-01-16', '2024-01-17', '2024-01-18']),
            'symbol': ['RELIANCE'] * 4,
            'open': [2500.0, 2510.0, 2520.0, 2530.0],
            'high': [2550.0, 2560.0, 2570.0, 2580.0],
            'low': [2480.0, 2490.0, 2500.0, 2510.0],
            'close': [2540.0, 2550.0, 2560.0, 2570.0],
            'volume': [1000000, 1100000, 1200000, 1300000]
        })
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_init_with_default_dependencies(self):
        """Test initialization with default dependencies."""
        fetcher = NewsDataFetcher()
        
        self.assertIsInstance(fetcher.data_client, YFinanceClient)
        self.assertIsInstance(fetcher.data_storage, DataStorage)
        self.assertIsNotNone(fetcher.logger)
    
    def test_init_with_custom_dependencies(self):
        """Test initialization with custom dependencies."""
        self.assertEqual(self.fetcher.data_client, self.mock_client)
        self.assertEqual(self.fetcher.data_storage, self.mock_storage)
    
    def test_fetch_news_for_symbol_success(self):
        """Test successful news fetching for a symbol."""
        # Setup mock
        self.mock_client.fetch_news_data.return_value = self.sample_news_data
        
        # Test
        result = self.fetcher.fetch_news_for_symbol('RELIANCE', '2024-01-15', '2024-01-17')
        
        # Assertions
        self.assertEqual(len(result), 3)
        self.mock_client.fetch_news_data.assert_called_once_with('RELIANCE', '2024-01-15', '2024-01-17')
        
        # Check data structure
        for news_item in result:
            self.assertIn('symbol', news_item)
            self.assertIn('title', news_item)
            self.assertIn('publish_time', news_item)
    
    def test_fetch_news_for_symbol_no_data(self):
        """Test news fetching when no data is available."""
        # Setup mock to return empty list
        self.mock_client.fetch_news_data.return_value = []
        
        # Test
        result = self.fetcher.fetch_news_for_symbol('RELIANCE', '2024-01-15', '2024-01-17')
        
        # Assertions
        self.assertEqual(len(result), 0)
        self.mock_client.fetch_news_data.assert_called_once()
    
    def test_fetch_news_for_symbol_api_error(self):
        """Test news fetching when API error occurs."""
        # Setup mock to raise exception
        self.mock_client.fetch_news_data.side_effect = Exception("API Error")
        
        # Test - should return empty list instead of raising exception
        result = self.fetcher.fetch_news_for_symbol('RELIANCE', '2024-01-15', '2024-01-17')
        
        # Assertions
        self.assertEqual(len(result), 0)
        self.mock_client.fetch_news_data.assert_called_once()
    
    def test_fetch_and_save_news_success(self):
        """Test successful news fetching and saving."""
        # Setup mocks
        self.mock_client.fetch_news_data.return_value = self.sample_news_data
        self.mock_storage.save_news_data.return_value = True
        
        # Test
        result = self.fetcher.fetch_and_save_news('RELIANCE', '2024-01-15', '2024-01-17')
        
        # Assertions
        self.assertTrue(result['success'])
        self.assertEqual(result['symbol'], 'RELIANCE')
        self.assertEqual(result['news_count'], 3)
        self.assertIsNone(result['error'])
        
        self.mock_client.fetch_news_data.assert_called_once()
        self.mock_storage.save_news_data.assert_called_once()
    
    def test_fetch_and_save_news_storage_failure(self):
        """Test news fetching when storage fails."""
        # Setup mocks
        self.mock_client.fetch_news_data.return_value = self.sample_news_data
        self.mock_storage.save_news_data.return_value = False
        
        # Test
        result = self.fetcher.fetch_and_save_news('RELIANCE', '2024-01-15', '2024-01-17')
        
        # Assertions
        self.assertFalse(result['success'])
        self.assertIsNotNone(result['error'])
        self.assertEqual(result['news_count'], 0)
    
    def test_fetch_news_for_multiple_symbols_success(self):
        """Test fetching news for multiple symbols."""
        symbols = ['RELIANCE', 'TCS', 'INFY']
        
        # Setup mock to return different data for each symbol
        def mock_fetch_news(symbol, start_date, end_date):
            if symbol == 'RELIANCE':
                return self.sample_news_data
            elif symbol == 'TCS':
                return self.sample_news_data[:2]  # 2 items
            else:
                return []  # No news for INFY
        
        self.mock_client.fetch_news_data.side_effect = mock_fetch_news
        self.mock_storage.save_news_data.return_value = True
        
        # Test
        result = self.fetcher.fetch_news_for_multiple_symbols(
            symbols, '2024-01-15', '2024-01-17', save_to_storage=True
        )
        
        # Assertions
        self.assertEqual(result['total_symbols'], 3)
        self.assertEqual(result['successful_fetches'], 3)  # All should succeed even with 0 news
        self.assertEqual(result['failed_fetches'], 0)
        self.assertEqual(result['total_news_items'], 5)  # 3 + 2 + 0
        self.assertEqual(len(result['detailed_results']), 3)
    
    def test_fetch_news_for_multiple_symbols_with_errors(self):
        """Test fetching news for multiple symbols with some errors."""
        symbols = ['RELIANCE', 'TCS', 'INVALID']
        
        # Setup mock to raise exception for invalid symbol
        def mock_fetch_news(symbol, start_date, end_date):
            if symbol == 'INVALID':
                raise Exception("Invalid symbol")
            elif symbol == 'RELIANCE':
                return self.sample_news_data
            else:
                return []
        
        # Setup storage mock to fail for INVALID symbol
        def mock_save_news(symbol, news_data):
            if symbol == 'INVALID':
                return False  # Simulate storage failure
            return True
        
        self.mock_client.fetch_news_data.side_effect = mock_fetch_news
        self.mock_storage.save_news_data.side_effect = mock_save_news
        
        # Test
        result = self.fetcher.fetch_news_for_multiple_symbols(
            symbols, '2024-01-15', '2024-01-17', save_to_storage=True
        )
        
        # Assertions
        self.assertEqual(result['total_symbols'], 3)
        self.assertEqual(result['successful_fetches'], 2)  # RELIANCE and TCS succeed
        self.assertEqual(result['failed_fetches'], 1)  # INVALID fails due to empty news from error
        # Note: Since fetch_news_for_symbol returns [] on error, it's treated as successful fetch with 0 news
        # The actual error handling happens at the client level
    
    def test_associate_news_with_stock_data_success(self):
        """Test successful association of news with stock data."""
        # Setup mock storage to return news data
        self.mock_storage.load_news_data.return_value = self.sample_news_data
        
        # Test
        result = self.fetcher.associate_news_with_stock_data('RELIANCE', self.sample_stock_data)
        
        # Assertions
        self.assertFalse(result.empty)
        self.assertIn('news_count', result.columns)
        self.assertIn('sentiment_score', result.columns)
        
        # Check that news counts are properly associated
        # Dates 2024-01-15, 2024-01-16, 2024-01-17 should have news
        # Date 2024-01-18 should have 0 news
        news_counts = result['news_count'].tolist()
        self.assertTrue(any(count > 0 for count in news_counts[:3]))  # First 3 dates have news
        self.assertEqual(news_counts[3], 0)  # Last date has no news
        
        # Check sentiment scores are initialized to neutral
        self.assertTrue(all(score == 0.0 for score in result['sentiment_score']))
    
    def test_associate_news_with_stock_data_no_news(self):
        """Test association when no news data is available."""
        # Setup mock storage to return empty news data
        self.mock_storage.load_news_data.return_value = []
        
        # Test
        result = self.fetcher.associate_news_with_stock_data('RELIANCE', self.sample_stock_data)
        
        # Assertions
        self.assertFalse(result.empty)
        self.assertIn('news_count', result.columns)
        self.assertIn('sentiment_score', result.columns)
        
        # All news counts should be 0
        self.assertTrue(all(count == 0 for count in result['news_count']))
        # All sentiment scores should be neutral (0.0)
        self.assertTrue(all(score == 0.0 for score in result['sentiment_score']))
    
    def test_associate_news_with_stock_data_provided_news(self):
        """Test association with provided news data instead of loading from storage."""
        # Test with provided news data (should not call storage)
        result = self.fetcher.associate_news_with_stock_data(
            'RELIANCE', self.sample_stock_data, news_data=self.sample_news_data
        )
        
        # Assertions
        self.assertFalse(result.empty)
        self.assertIn('news_count', result.columns)
        self.assertIn('sentiment_score', result.columns)
        
        # Storage should not be called
        self.mock_storage.load_news_data.assert_not_called()
    
    def test_associate_news_with_stock_data_empty_stock_data(self):
        """Test association with empty stock data."""
        empty_stock_data = pd.DataFrame()
        
        # Test
        result = self.fetcher.associate_news_with_stock_data('RELIANCE', empty_stock_data)
        
        # Assertions
        self.assertTrue(result.empty)
    
    def test_get_news_summary_for_date_range_success(self):
        """Test getting news summary for date range."""
        # Setup mock storage
        self.mock_storage.load_news_data.return_value = self.sample_news_data
        
        # Test
        summary = self.fetcher.get_news_summary_for_date_range('RELIANCE', '2024-01-15', '2024-01-17')
        
        # Assertions
        self.assertEqual(summary['symbol'], 'RELIANCE')
        self.assertEqual(summary['total_news_items'], 3)
        self.assertEqual(summary['unique_publishers'], 3)  # ET, BS, Mint
        self.assertEqual(summary['date_coverage'], 3)  # 3 different dates
        self.assertEqual(summary['total_days_in_range'], 3)
        self.assertEqual(summary['avg_news_per_day'], 1.0)  # 3 news / 3 days
    
    def test_get_news_summary_for_date_range_no_news(self):
        """Test getting news summary when no news data exists."""
        # Setup mock storage to return empty list
        self.mock_storage.load_news_data.return_value = []
        
        # Test
        summary = self.fetcher.get_news_summary_for_date_range('RELIANCE', '2024-01-15', '2024-01-17')
        
        # Assertions
        self.assertEqual(summary['symbol'], 'RELIANCE')
        self.assertEqual(summary['total_news_items'], 0)
        self.assertEqual(summary['unique_publishers'], 0)
        self.assertEqual(summary['date_coverage'], 0)
        self.assertEqual(summary['avg_news_per_day'], 0.0)
    
    def test_validate_date_range_valid(self):
        """Test date range validation with valid dates."""
        # Should not raise exception
        try:
            self.fetcher._validate_date_range('2024-01-15', '2024-01-17')
        except Exception:
            self.fail("Valid date range should not raise exception")
    
    def test_validate_date_range_invalid_format(self):
        """Test date range validation with invalid format."""
        with self.assertRaises(ValueError) as context:
            self.fetcher._validate_date_range('2024/01/15', '2024-01-17')
        
        self.assertIn("Invalid date format", str(context.exception))
    
    def test_validate_date_range_start_after_end(self):
        """Test date range validation when start date is after end date."""
        with self.assertRaises(ValueError) as context:
            self.fetcher._validate_date_range('2024-01-17', '2024-01-15')
        
        self.assertIn("Start date must be before end date", str(context.exception))
    
    def test_process_news_data_filtering(self):
        """Test news data processing and filtering."""
        # Create test data with some invalid items
        raw_news_data = [
            {
                'symbol': 'RELIANCE',
                'title': 'Valid news item',
                'summary': 'Valid summary',
                'publisher': 'Test Publisher',
                'publish_time': int(datetime(2024, 1, 15).timestamp()),
                'url': 'https://example.com/news1'
            },
            {
                'symbol': 'RELIANCE',
                'title': '',  # Empty title - should be filtered out
                'summary': 'Summary without title',
                'publisher': 'Test Publisher',
                'publish_time': int(datetime(2024, 1, 16).timestamp()),
                'url': 'https://example.com/news2'
            },
            {
                'symbol': 'RELIANCE',
                'title': 'Another valid item',
                'summary': '',  # Empty summary is OK
                'publisher': 'Test Publisher',
                'publish_time': int(datetime(2024, 1, 17).timestamp()),
                'url': 'https://example.com/news3'
            },
            {
                'symbol': 'RELIANCE',
                'title': 'Item with zero timestamp',
                'summary': 'Summary',
                'publisher': 'Test Publisher',
                'publish_time': 0,  # Zero timestamp - should be filtered out
                'url': 'https://example.com/news4'
            }
        ]
        
        # Test
        processed = self.fetcher._process_news_data(raw_news_data, 'RELIANCE', '2024-01-15', '2024-01-17')
        
        # Assertions
        self.assertEqual(len(processed), 2)  # Only 2 valid items should remain
        
        # Check that all processed items have required fields
        for item in processed:
            self.assertIn('symbol', item)
            self.assertIn('title', item)
            self.assertIn('publish_time', item)
            self.assertTrue(item['title'])  # Title should not be empty
            self.assertGreater(item['publish_time'], 0)  # Timestamp should be valid
    
    def test_fetch_news_for_multiple_symbols_without_saving(self):
        """Test fetching news for multiple symbols without saving to storage."""
        symbols = ['RELIANCE', 'TCS']
        
        # Setup mock
        self.mock_client.fetch_news_data.return_value = self.sample_news_data
        
        # Test
        result = self.fetcher.fetch_news_for_multiple_symbols(
            symbols, '2024-01-15', '2024-01-17', save_to_storage=False
        )
        
        # Assertions
        self.assertEqual(result['total_symbols'], 2)
        self.assertEqual(result['successful_fetches'], 2)
        self.assertEqual(result['failed_fetches'], 0)
        
        # Check that detailed results contain news data
        for detail in result['detailed_results']:
            self.assertIn('news_data', detail)
            self.assertEqual(len(detail['news_data']), 3)
        
        # Storage should not be called
        self.mock_storage.save_news_data.assert_not_called()


if __name__ == '__main__':
    unittest.main()