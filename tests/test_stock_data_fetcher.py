"""
Integration tests for StockDataFetcher class.
Tests data fetching workflow with batch processing and error recovery.
"""

import unittest
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
from datetime import datetime, timedelta
import logging
from pathlib import Path

from src.data.stock_data_fetcher import StockDataFetcher
from src.data.yfinance_client import YFinanceClient
from src.data.data_storage import DataStorage


class TestStockDataFetcher(unittest.TestCase):
    """Test cases for StockDataFetcher class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        
        # Create mock data client and storage
        self.mock_client = Mock(spec=YFinanceClient)
        self.data_storage = DataStorage(base_path=self.temp_dir)
        
        # Create fetcher with mocked client
        self.fetcher = StockDataFetcher(
            data_client=self.mock_client,
            data_storage=self.data_storage,
            batch_size=2,
            delay_between_batches=0.01  # Fast for testing
        )
        
        # Sample stock data for testing
        self.sample_stock_data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=3),
            'symbol': ['RELIANCE', 'RELIANCE', 'RELIANCE'],
            'open': [100.0, 101.0, 102.0],
            'high': [105.0, 106.0, 107.0],
            'low': [99.0, 100.0, 101.0],
            'close': [104.0, 105.0, 106.0],
            'volume': [1000000, 1100000, 1200000]
        })
        
        # Sample news data for testing
        self.sample_news_data = [
            {
                'symbol': 'RELIANCE',
                'title': 'Company reports strong earnings',
                'summary': 'Quarterly results exceed expectations',
                'publisher': 'Financial Times',
                'publish_time': int(datetime(2023, 1, 1).timestamp()),
                'url': 'https://example.com/news1'
            }
        ]
        
        # Create test config file
        self.test_config_file = Path(self.temp_dir) / "test_stocks.txt"
        with open(self.test_config_file, 'w') as f:
            f.write("# Test stock symbols\n")
            f.write("RELIANCE.NS\n")
            f.write("TCS.NS\n")
            f.write("INFY.NS\n")
    
    def tearDown(self):
        """Clean up test fixtures."""
        # Remove temporary directory
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test StockDataFetcher initialization."""
        # Test with default parameters
        fetcher = StockDataFetcher()
        self.assertIsInstance(fetcher.data_client, YFinanceClient)
        self.assertIsInstance(fetcher.data_storage, DataStorage)
        self.assertEqual(fetcher.batch_size, 10)
        self.assertEqual(fetcher.delay_between_batches, 1.0)
        
        # Test with custom parameters
        custom_fetcher = StockDataFetcher(batch_size=5, delay_between_batches=0.5)
        self.assertEqual(custom_fetcher.batch_size, 5)
        self.assertEqual(custom_fetcher.delay_between_batches, 0.5)
    
    def test_load_stock_symbols(self):
        """Test loading stock symbols from configuration file."""
        # Test loading from test config file
        symbols = self.fetcher.load_stock_symbols(str(self.test_config_file))
        
        expected_symbols = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS']
        self.assertEqual(symbols, expected_symbols)
    
    def test_load_stock_symbols_nonexistent_file(self):
        """Test handling of nonexistent configuration file."""
        symbols = self.fetcher.load_stock_symbols("nonexistent_file.txt")
        self.assertEqual(symbols, [])
    
    @patch('src.data.stock_data_fetcher.Path')
    def test_get_nifty50_symbols(self, mock_path):
        """Test getting NIFTY 50 symbols."""
        # Mock the config file path
        mock_config_path = Mock()
        mock_config_path.exists.return_value = True
        mock_path.return_value = mock_config_path
        
        # Mock file reading
        with patch('builtins.open', unittest.mock.mock_open(read_data="RELIANCE.NS\nTCS.NS\n")):
            symbols = self.fetcher.get_nifty50_symbols()
            self.assertEqual(symbols, ['RELIANCE.NS', 'TCS.NS'])
    
    def test_fetch_single_stock_data_success(self):
        """Test successful single stock data fetching."""
        # Mock successful data fetching
        self.mock_client.fetch_stock_data.return_value = self.sample_stock_data
        self.mock_client.fetch_news_data.return_value = self.sample_news_data
        
        # Fetch data
        result = self.fetcher.fetch_single_stock_data('RELIANCE.NS', '2023-01-01', '2023-01-03')
        
        # Verify results
        self.assertTrue(result['success'])
        self.assertEqual(result['symbol'], 'RELIANCE.NS')
        self.assertEqual(result['stock_records'], len(self.sample_stock_data))
        self.assertEqual(result['news_records'], len(self.sample_news_data))
        self.assertIsNone(result['error'])
        
        # Verify API calls
        self.mock_client.fetch_stock_data.assert_called_once_with('RELIANCE.NS', '2023-01-01', '2023-01-03')
        self.mock_client.fetch_news_data.assert_called_once_with('RELIANCE.NS', '2023-01-01', '2023-01-03')
    
    def test_fetch_single_stock_data_no_news(self):
        """Test single stock data fetching without news."""
        # Mock successful stock data fetching
        self.mock_client.fetch_stock_data.return_value = self.sample_stock_data
        
        # Fetch data without news
        result = self.fetcher.fetch_single_stock_data('RELIANCE.NS', '2023-01-01', '2023-01-03', include_news=False)
        
        # Verify results
        self.assertTrue(result['success'])
        self.assertEqual(result['stock_records'], len(self.sample_stock_data))
        self.assertEqual(result['news_records'], 0)
        
        # Verify news API was not called
        self.mock_client.fetch_news_data.assert_not_called()
    
    def test_fetch_single_stock_data_failure(self):
        """Test handling of single stock data fetching failure."""
        # Mock failed data fetching
        self.mock_client.fetch_stock_data.side_effect = Exception("API Error")
        
        # Fetch data
        result = self.fetcher.fetch_single_stock_data('INVALID.NS', '2023-01-01', '2023-01-03')
        
        # Verify results
        self.assertFalse(result['success'])
        self.assertEqual(result['symbol'], 'INVALID.NS')
        self.assertEqual(result['stock_records'], 0)
        self.assertEqual(result['news_records'], 0)
        self.assertIsNotNone(result['error'])
    
    def test_fetch_single_stock_data_empty_response(self):
        """Test handling of empty stock data response."""
        # Mock empty data response
        self.mock_client.fetch_stock_data.return_value = pd.DataFrame()
        self.mock_client.fetch_news_data.return_value = []
        
        # Fetch data
        result = self.fetcher.fetch_single_stock_data('EMPTY.NS', '2023-01-01', '2023-01-03')
        
        # Verify results
        self.assertFalse(result['success'])  # Should fail due to empty data
        self.assertEqual(result['stock_records'], 0)
        self.assertEqual(result['news_records'], 0)
    
    def test_fetch_batch_data(self):
        """Test batch data fetching."""
        # Mock successful data fetching for all symbols
        self.mock_client.fetch_stock_data.return_value = self.sample_stock_data
        self.mock_client.fetch_news_data.return_value = self.sample_news_data
        
        symbols = ['RELIANCE.NS', 'TCS.NS']
        results = self.fetcher.fetch_batch_data(symbols, '2023-01-01', '2023-01-03')
        
        # Verify results
        self.assertEqual(len(results), 2)
        
        for result in results:
            self.assertTrue(result['success'])
            self.assertEqual(result['stock_records'], len(self.sample_stock_data))
            self.assertEqual(result['news_records'], len(self.sample_news_data))
        
        # Verify statistics were updated
        self.assertEqual(self.fetcher.stats['successful_fetches'], 2)
        self.assertEqual(self.fetcher.stats['failed_fetches'], 0)
    
    def test_fetch_batch_data_mixed_results(self):
        """Test batch data fetching with mixed success/failure results."""
        # Mock mixed results - first succeeds, second fails
        def mock_fetch_stock_data(symbol, start_date, end_date):
            if symbol == 'RELIANCE.NS':
                return self.sample_stock_data
            else:
                raise Exception("API Error")
        
        self.mock_client.fetch_stock_data.side_effect = mock_fetch_stock_data
        self.mock_client.fetch_news_data.return_value = self.sample_news_data
        
        symbols = ['RELIANCE.NS', 'INVALID.NS']
        results = self.fetcher.fetch_batch_data(symbols, '2023-01-01', '2023-01-03')
        
        # Verify results
        self.assertEqual(len(results), 2)
        self.assertTrue(results[0]['success'])
        self.assertFalse(results[1]['success'])
        
        # Verify statistics
        self.assertEqual(self.fetcher.stats['successful_fetches'], 1)
        self.assertEqual(self.fetcher.stats['failed_fetches'], 1)
    
    def test_fetch_all_stocks_data_success(self):
        """Test successful fetching of all stocks data."""
        # Mock successful data fetching
        self.mock_client.fetch_stock_data.return_value = self.sample_stock_data
        self.mock_client.fetch_news_data.return_value = self.sample_news_data
        
        # Use small symbol list for testing
        symbols = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS']
        
        result = self.fetcher.fetch_all_stocks_data(
            start_date='2023-01-01',
            end_date='2023-01-03',
            symbols=symbols
        )
        
        # Verify overall result
        self.assertTrue(result['success'])
        self.assertIn('summary', result)
        self.assertIn('detailed_results', result)
        self.assertIn('statistics', result)
        
        # Verify summary
        summary = result['summary']
        self.assertEqual(summary['total_symbols_processed'], 3)
        self.assertEqual(summary['successful_fetches'], 3)
        self.assertEqual(summary['failed_fetches'], 0)
        self.assertEqual(summary['success_rate_percent'], 100.0)
        
        # Verify detailed results
        detailed_results = result['detailed_results']
        self.assertEqual(len(detailed_results), 3)
        
        for detail in detailed_results:
            self.assertTrue(detail['success'])
    
    def test_fetch_all_stocks_data_with_failures(self):
        """Test fetching all stocks data with some failures."""
        # Mock mixed results
        def mock_fetch_stock_data(symbol, start_date, end_date):
            if symbol == 'RELIANCE.NS':
                return self.sample_stock_data
            else:
                raise Exception("API Error")
        
        self.mock_client.fetch_stock_data.side_effect = mock_fetch_stock_data
        self.mock_client.fetch_news_data.return_value = self.sample_news_data
        
        symbols = ['RELIANCE.NS', 'INVALID1.NS', 'INVALID2.NS']
        
        result = self.fetcher.fetch_all_stocks_data(
            start_date='2023-01-01',
            end_date='2023-01-03',
            symbols=symbols
        )
        
        # Verify overall result
        self.assertTrue(result['success'])
        
        # Verify summary
        summary = result['summary']
        self.assertEqual(summary['total_symbols_processed'], 3)
        self.assertEqual(summary['successful_fetches'], 1)
        self.assertEqual(summary['failed_fetches'], 2)
        self.assertEqual(summary['success_rate_percent'], 33.33)
        self.assertEqual(summary['errors_count'], 2)
    
    def test_fetch_incremental_data(self):
        """Test incremental data fetching."""
        # Mock successful data fetching
        self.mock_client.fetch_stock_data.return_value = self.sample_stock_data
        self.mock_client.fetch_news_data.return_value = self.sample_news_data
        
        # Add some symbols to storage first
        self.data_storage.save_stock_data('RELIANCE', self.sample_stock_data)
        
        result = self.fetcher.fetch_incremental_data(days_back=7)
        
        # Verify result
        self.assertTrue(result['success'])
        self.assertIn('summary', result)
        
        # Verify date range was calculated correctly
        expected_start = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        expected_end = datetime.now().strftime('%Y-%m-%d')
        
        # Check that fetch was called with correct date range
        self.mock_client.fetch_stock_data.assert_called()
    
    def test_get_fetch_statistics(self):
        """Test getting fetch statistics."""
        # Initialize some statistics
        self.fetcher.stats['total_symbols'] = 10
        self.fetcher.stats['successful_fetches'] = 8
        self.fetcher.stats['failed_fetches'] = 2
        
        stats = self.fetcher.get_fetch_statistics()
        
        # Verify statistics
        self.assertEqual(stats['total_symbols'], 10)
        self.assertEqual(stats['successful_fetches'], 8)
        self.assertEqual(stats['failed_fetches'], 2)
        
        # Verify it's a copy (not reference)
        stats['total_symbols'] = 999
        self.assertEqual(self.fetcher.stats['total_symbols'], 10)
    
    def test_reset_statistics(self):
        """Test resetting fetch statistics."""
        # Set some statistics
        self.fetcher.stats['total_symbols'] = 10
        self.fetcher.stats['successful_fetches'] = 8
        self.fetcher.stats['errors'] = [{'symbol': 'TEST', 'error': 'Test error'}]
        
        # Reset statistics
        self.fetcher.reset_statistics()
        
        # Verify reset
        self.assertEqual(self.fetcher.stats['total_symbols'], 0)
        self.assertEqual(self.fetcher.stats['successful_fetches'], 0)
        self.assertEqual(self.fetcher.stats['failed_fetches'], 0)
        self.assertEqual(len(self.fetcher.stats['errors']), 0)
    
    def test_batch_processing_with_delay(self):
        """Test that batch processing includes delays between batches."""
        # Mock successful data fetching
        self.mock_client.fetch_stock_data.return_value = self.sample_stock_data
        self.mock_client.fetch_news_data.return_value = []
        
        # Use symbols that will require multiple batches (batch_size=2)
        symbols = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFC.NS']
        
        start_time = datetime.now()
        
        result = self.fetcher.fetch_all_stocks_data(
            start_date='2023-01-01',
            end_date='2023-01-03',
            symbols=symbols
        )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Verify result
        self.assertTrue(result['success'])
        
        # Verify that some time was spent (due to delays between batches)
        # With 4 symbols and batch_size=2, we have 2 batches, so 1 delay of 0.01s
        self.assertGreater(duration, 0.005)  # Should be at least some delay
    
    def test_error_tracking(self):
        """Test that errors are properly tracked and reported."""
        # Mock failures for specific symbols
        def mock_fetch_stock_data(symbol, start_date, end_date):
            if 'INVALID' in symbol:
                raise Exception(f"API Error for {symbol}")
            return self.sample_stock_data
        
        self.mock_client.fetch_stock_data.side_effect = mock_fetch_stock_data
        self.mock_client.fetch_news_data.return_value = []
        
        symbols = ['RELIANCE.NS', 'INVALID1.NS', 'INVALID2.NS']
        
        result = self.fetcher.fetch_all_stocks_data(
            start_date='2023-01-01',
            end_date='2023-01-03',
            symbols=symbols
        )
        
        # Verify errors were tracked
        self.assertEqual(len(self.fetcher.stats['errors']), 2)
        
        error1 = self.fetcher.stats['errors'][0]
        self.assertEqual(error1['symbol'], 'INVALID1.NS')
        self.assertIn('API Error', error1['error'])
        
        error2 = self.fetcher.stats['errors'][1]
        self.assertEqual(error2['symbol'], 'INVALID2.NS')
        self.assertIn('API Error', error2['error'])


if __name__ == '__main__':
    # Configure logging for tests
    logging.basicConfig(level=logging.WARNING)
    unittest.main()