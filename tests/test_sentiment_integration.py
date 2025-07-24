"""
Integration tests for SentimentIntegration class.
Tests the complete sentiment analysis pipeline integration with trading data.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
from datetime import datetime, timedelta
import tempfile
import shutil

from src.data.sentiment_integration import SentimentIntegration
from src.data.news_data_fetcher import NewsDataFetcher
from src.data.sentiment_analyzer import SentimentAnalyzer
from src.data.data_storage import DataStorage


class TestSentimentIntegration(unittest.TestCase):
    """Test cases for SentimentIntegration class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for test data
        self.test_dir = tempfile.mkdtemp()
        
        # Mock dependencies
        self.mock_news_fetcher = Mock(spec=NewsDataFetcher)
        self.mock_sentiment_analyzer = Mock(spec=SentimentAnalyzer)
        self.mock_data_storage = Mock(spec=DataStorage)
        
        # Create integration instance
        self.integration = SentimentIntegration(
            news_fetcher=self.mock_news_fetcher,
            sentiment_analyzer=self.mock_sentiment_analyzer,
            data_storage=self.mock_data_storage
        )
        
        # Sample news data with sentiment scores
        self.sample_news_with_sentiment = [
            {
                'symbol': 'RELIANCE',
                'title': 'Reliance Industries reports strong Q4 results',
                'summary': 'Company shows growth in all segments',
                'publisher': 'Economic Times',
                'publish_time': int(datetime(2024, 1, 15).timestamp()),
                'url': 'https://example.com/news1',
                'sentiment_score': 0.8
            },
            {
                'symbol': 'RELIANCE',
                'title': 'Reliance stock crashes amid market volatility',
                'summary': 'Major losses reported',
                'publisher': 'Business Standard',
                'publish_time': int(datetime(2024, 1, 16).timestamp()),
                'url': 'https://example.com/news2',
                'sentiment_score': -0.7
            },
            {
                'symbol': 'RELIANCE',
                'title': 'Neutral outlook for Reliance shares',
                'summary': 'Analysts maintain hold rating',
                'publisher': 'Mint',
                'publish_time': int(datetime(2024, 1, 17).timestamp()),
                'url': 'https://example.com/news3',
                'sentiment_score': 0.1
            }
        ]
        
        # Sample stock data
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
        integration = SentimentIntegration()
        
        self.assertIsInstance(integration.news_fetcher, NewsDataFetcher)
        self.assertIsInstance(integration.sentiment_analyzer, SentimentAnalyzer)
        self.assertIsInstance(integration.data_storage, DataStorage)
        self.assertIsNotNone(integration.logger)
    
    def test_init_with_custom_dependencies(self):
        """Test initialization with custom dependencies."""
        self.assertEqual(self.integration.news_fetcher, self.mock_news_fetcher)
        self.assertEqual(self.integration.sentiment_analyzer, self.mock_sentiment_analyzer)
        self.assertEqual(self.integration.data_storage, self.mock_data_storage)
    
    def test_process_sentiment_for_symbol_success(self):
        """Test successful sentiment processing for a single symbol."""
        # Setup mocks
        raw_news_data = [item.copy() for item in self.sample_news_with_sentiment]
        for item in raw_news_data:
            del item['sentiment_score']  # Remove sentiment score to simulate raw news
        
        self.mock_news_fetcher.fetch_news_for_symbol.return_value = raw_news_data
        self.mock_sentiment_analyzer.analyze_news_headlines.return_value = self.sample_news_with_sentiment
        self.mock_data_storage.save_news_data.return_value = True
        self.mock_data_storage.load_stock_data.return_value = self.sample_stock_data
        self.mock_data_storage.save_stock_data.return_value = True
        
        # Test
        result = self.integration.process_sentiment_for_symbol('RELIANCE', '2024-01-15', '2024-01-17')
        
        # Assertions
        self.assertTrue(result['success'])
        self.assertEqual(result['symbol'], 'RELIANCE')
        self.assertEqual(result['news_count'], 3)
        self.assertTrue(result['sentiment_processed'])
        self.assertTrue(result['data_integrated'])
        self.assertIsNone(result['error'])
        self.assertIn('sentiment_summary', result)
        
        # Verify method calls
        self.mock_news_fetcher.fetch_news_for_symbol.assert_called_once_with('RELIANCE', '2024-01-15', '2024-01-17')
        self.mock_sentiment_analyzer.analyze_news_headlines.assert_called_once()
        self.mock_data_storage.save_news_data.assert_called_once()
        self.mock_data_storage.load_stock_data.assert_called_once_with('RELIANCE')
        self.mock_data_storage.save_stock_data.assert_called_once()
    
    def test_process_sentiment_for_symbol_no_news(self):
        """Test sentiment processing when no news data is available."""
        # Setup mocks
        self.mock_news_fetcher.fetch_news_for_symbol.return_value = []
        self.mock_data_storage.load_stock_data.return_value = self.sample_stock_data
        self.mock_data_storage.save_stock_data.return_value = True
        
        # Test
        result = self.integration.process_sentiment_for_symbol('RELIANCE', '2024-01-15', '2024-01-17')
        
        # Assertions
        self.assertTrue(result['success'])
        self.assertEqual(result['news_count'], 0)
        self.assertFalse(result['sentiment_processed'])
        self.assertTrue(result['data_integrated'])
        
        # Check sentiment summary for no news
        self.assertEqual(result['sentiment_summary']['avg_sentiment'], 0.0)
        self.assertEqual(result['sentiment_summary']['sentiment_count'], 0)
    
    def test_process_sentiment_for_symbol_no_stock_data(self):
        """Test sentiment processing when no stock data is available."""
        # Setup mocks - need to mock the raw news data without sentiment scores first
        raw_news_data = [item.copy() for item in self.sample_news_with_sentiment]
        for item in raw_news_data:
            del item['sentiment_score']  # Remove sentiment score to simulate raw news
        
        self.mock_news_fetcher.fetch_news_for_symbol.return_value = raw_news_data
        self.mock_sentiment_analyzer.analyze_news_headlines.return_value = self.sample_news_with_sentiment
        self.mock_data_storage.save_news_data.return_value = True
        self.mock_data_storage.load_stock_data.return_value = pd.DataFrame()  # Empty DataFrame
        
        # Test
        result = self.integration.process_sentiment_for_symbol('RELIANCE', '2024-01-15', '2024-01-17')
        
        # Assertions
        self.assertFalse(result['success'])
        self.assertEqual(result['error'], "No stock data available")
        self.assertFalse(result['data_integrated'])
    
    def test_process_sentiment_for_symbol_with_provided_stock_data(self):
        """Test sentiment processing with provided stock data."""
        # Setup mocks
        raw_news_data = [item.copy() for item in self.sample_news_with_sentiment]
        for item in raw_news_data:
            del item['sentiment_score']
        
        self.mock_news_fetcher.fetch_news_for_symbol.return_value = raw_news_data
        self.mock_sentiment_analyzer.analyze_news_headlines.return_value = self.sample_news_with_sentiment
        self.mock_data_storage.save_news_data.return_value = True
        self.mock_data_storage.save_stock_data.return_value = True
        
        # Test with provided stock data
        result = self.integration.process_sentiment_for_symbol(
            'RELIANCE', '2024-01-15', '2024-01-17', stock_data=self.sample_stock_data
        )
        
        # Assertions
        self.assertTrue(result['success'])
        
        # Should not call load_stock_data since data was provided
        self.mock_data_storage.load_stock_data.assert_not_called()
    
    def test_process_sentiment_for_multiple_symbols_success(self):
        """Test sentiment processing for multiple symbols."""
        symbols = ['RELIANCE', 'TCS', 'INFY']
        
        # Setup mocks to return different results for each symbol
        def mock_process_single(symbol, start_date, end_date, stock_data=None):
            if symbol == 'RELIANCE':
                return {'success': True, 'symbol': symbol, 'news_count': 3, 'error': None}
            elif symbol == 'TCS':
                return {'success': True, 'symbol': symbol, 'news_count': 2, 'error': None}
            else:  # INFY
                return {'success': False, 'symbol': symbol, 'news_count': 0, 'error': 'No data'}
        
        # Mock the process_sentiment_for_symbol method
        self.integration.process_sentiment_for_symbol = Mock(side_effect=mock_process_single)
        
        # Test
        result = self.integration.process_sentiment_for_multiple_symbols(symbols, '2024-01-15', '2024-01-17')
        
        # Assertions
        self.assertEqual(result['total_symbols'], 3)
        self.assertEqual(result['successful_processing'], 2)
        self.assertEqual(result['failed_processing'], 1)
        self.assertEqual(result['total_news_items'], 5)  # 3 + 2 + 0
        self.assertEqual(len(result['detailed_results']), 3)
        self.assertEqual(len(result['errors']), 1)
        self.assertEqual(result['errors'][0]['symbol'], 'INFY')
    
    def test_get_sentiment_for_trading_date_success(self):
        """Test getting sentiment for a specific trading date."""
        # Setup mock data storage
        self.mock_data_storage.load_news_data.return_value = self.sample_news_with_sentiment
        self.mock_sentiment_analyzer.get_aggregated_sentiment.return_value = 0.4
        
        # Test
        result = self.integration.get_sentiment_for_trading_date('RELIANCE', '2024-01-16')
        
        # Assertions
        self.assertEqual(result['symbol'], 'RELIANCE')
        self.assertEqual(result['trading_date'], '2024-01-16')
        self.assertEqual(result['sentiment_score'], 0.4)
        self.assertGreater(result['news_count'], 0)
        self.assertEqual(result['sentiment_label'], 'Positive')
        self.assertIn('news_items', result)
    
    def test_get_sentiment_for_trading_date_no_news(self):
        """Test getting sentiment when no news data is available."""
        # Setup mock data storage to return empty list
        self.mock_data_storage.load_news_data.return_value = []
        
        # Test
        result = self.integration.get_sentiment_for_trading_date('RELIANCE', '2024-01-16')
        
        # Assertions
        self.assertEqual(result['symbol'], 'RELIANCE')
        self.assertEqual(result['trading_date'], '2024-01-16')
        self.assertEqual(result['sentiment_score'], 0.0)
        self.assertEqual(result['news_count'], 0)
        self.assertEqual(result['sentiment_label'], 'Neutral')
    
    def test_get_sentiment_for_trading_date_with_lookback(self):
        """Test getting sentiment with custom lookback days."""
        # Setup mock data storage
        self.mock_data_storage.load_news_data.return_value = self.sample_news_with_sentiment
        self.mock_sentiment_analyzer.get_aggregated_sentiment.return_value = 0.2
        
        # Test with 2-day lookback
        result = self.integration.get_sentiment_for_trading_date('RELIANCE', '2024-01-17', lookback_days=2)
        
        # Assertions
        self.assertEqual(result['sentiment_score'], 0.2)
        self.assertGreater(result['news_count'], 0)
    
    def test_update_stock_data_with_sentiment_success(self):
        """Test updating stock data with sentiment scores."""
        # Setup mocks
        self.mock_data_storage.load_news_data.return_value = self.sample_news_with_sentiment
        
        # Mock get_sentiment_for_trading_date method
        def mock_get_sentiment(symbol, trading_date):
            return {
                'sentiment_score': 0.5,
                'news_count': 2,
                'sentiment_label': 'Positive'
            }
        
        self.integration.get_sentiment_for_trading_date = Mock(side_effect=mock_get_sentiment)
        
        # Test
        result = self.integration.update_stock_data_with_sentiment('RELIANCE', self.sample_stock_data)
        
        # Assertions
        self.assertFalse(result.empty)
        self.assertIn('sentiment_score', result.columns)
        self.assertIn('news_count', result.columns)
        
        # Check that sentiment scores were added
        self.assertTrue(all(result['sentiment_score'] == 0.5))
        self.assertTrue(all(result['news_count'] == 2))
    
    def test_update_stock_data_with_sentiment_no_news(self):
        """Test updating stock data when no news data is available."""
        # Setup mock to return empty news data
        self.mock_data_storage.load_news_data.return_value = []
        
        # Test
        result = self.integration.update_stock_data_with_sentiment('RELIANCE', self.sample_stock_data)
        
        # Assertions
        self.assertFalse(result.empty)
        self.assertIn('sentiment_score', result.columns)
        self.assertIn('news_count', result.columns)
        
        # All sentiment scores should be neutral (0.0)
        self.assertTrue(all(result['sentiment_score'] == 0.0))
        self.assertTrue(all(result['news_count'] == 0))
    
    def test_update_stock_data_with_sentiment_empty_stock_data(self):
        """Test updating empty stock data."""
        empty_stock_data = pd.DataFrame()
        
        # Test
        result = self.integration.update_stock_data_with_sentiment('RELIANCE', empty_stock_data)
        
        # Assertions
        self.assertTrue(result.empty)
    
    def test_calculate_sentiment_summary(self):
        """Test sentiment summary calculation."""
        # Test with sample news data
        summary = self.integration._calculate_sentiment_summary(self.sample_news_with_sentiment)
        
        # Assertions
        self.assertIn('avg_sentiment', summary)
        self.assertIn('sentiment_count', summary)
        self.assertIn('positive_count', summary)
        self.assertIn('negative_count', summary)
        self.assertIn('neutral_count', summary)
        
        self.assertEqual(summary['sentiment_count'], 3)
        self.assertEqual(summary['positive_count'], 1)  # 0.8 > 0.1
        self.assertEqual(summary['negative_count'], 1)  # -0.7 < -0.1
        self.assertEqual(summary['neutral_count'], 1)   # 0.1 is neutral
        
        # Check average calculation
        expected_avg = (0.8 + (-0.7) + 0.1) / 3
        self.assertAlmostEqual(summary['avg_sentiment'], expected_avg, places=5)
    
    def test_calculate_sentiment_summary_empty(self):
        """Test sentiment summary calculation with empty data."""
        summary = self.integration._calculate_sentiment_summary([])
        
        # Assertions
        self.assertEqual(summary['avg_sentiment'], 0.0)
        self.assertEqual(summary['sentiment_count'], 0)
        self.assertEqual(summary['positive_count'], 0)
        self.assertEqual(summary['negative_count'], 0)
        self.assertEqual(summary['neutral_count'], 0)
    
    def test_get_sentiment_label(self):
        """Test sentiment score to label conversion."""
        # Test different score ranges
        self.assertEqual(self.integration._get_sentiment_label(0.5), "Positive")
        self.assertEqual(self.integration._get_sentiment_label(-0.5), "Negative")
        self.assertEqual(self.integration._get_sentiment_label(0.05), "Neutral")
        self.assertEqual(self.integration._get_sentiment_label(-0.05), "Neutral")
        self.assertEqual(self.integration._get_sentiment_label(0.0), "Neutral")
    
    def test_integrate_sentiment_with_stock_data_success(self):
        """Test integration of sentiment with stock data."""
        # Test the private method directly
        result = self.integration._integrate_sentiment_with_stock_data(
            'RELIANCE', self.sample_stock_data, self.sample_news_with_sentiment, '2024-01-15', '2024-01-17'
        )
        
        # Assertions
        self.assertFalse(result.empty)
        self.assertIn('sentiment_score', result.columns)
        self.assertIn('news_count', result.columns)
        
        # Check that data was filtered by date range
        self.assertLessEqual(len(result), len(self.sample_stock_data))
    
    def test_integrate_sentiment_with_stock_data_empty_stock_data(self):
        """Test integration with empty stock data."""
        empty_stock_data = pd.DataFrame()
        
        result = self.integration._integrate_sentiment_with_stock_data(
            'RELIANCE', empty_stock_data, self.sample_news_with_sentiment, '2024-01-15', '2024-01-17'
        )
        
        # Assertions
        self.assertTrue(result.empty)
    
    def test_integrate_sentiment_with_stock_data_no_news(self):
        """Test integration with no news data."""
        result = self.integration._integrate_sentiment_with_stock_data(
            'RELIANCE', self.sample_stock_data, [], '2024-01-15', '2024-01-17'
        )
        
        # Assertions
        self.assertFalse(result.empty)
        self.assertIn('sentiment_score', result.columns)
        self.assertIn('news_count', result.columns)
        
        # All sentiment scores should be neutral
        self.assertTrue(all(result['sentiment_score'] == 0.0))
        self.assertTrue(all(result['news_count'] == 0))
    
    def test_get_integration_stats(self):
        """Test getting integration system statistics."""
        # Setup mocks
        self.mock_data_storage.get_storage_stats.return_value = {
            'total_symbols': 10,
            'total_files': 30,
            'total_size_mb': 5.2
        }
        self.mock_sentiment_analyzer.get_model_info.return_value = {
            'model_name': 'distilbert-base-uncased-finetuned-sst-2-english',
            'device': 'cpu',
            'model_loaded': True
        }
        
        # Test
        stats = self.integration.get_integration_stats()
        
        # Assertions
        self.assertIn('storage_stats', stats)
        self.assertIn('sentiment_model_info', stats)
        self.assertIn('components_loaded', stats)
        
        # Check components loaded
        components = stats['components_loaded']
        self.assertTrue(components['news_fetcher'])
        self.assertTrue(components['sentiment_analyzer'])
        self.assertTrue(components['data_storage'])
    
    def test_error_handling_in_process_sentiment(self):
        """Test error handling in sentiment processing."""
        # Setup mock to raise exception
        self.mock_news_fetcher.fetch_news_for_symbol.side_effect = Exception("API Error")
        
        # Test
        result = self.integration.process_sentiment_for_symbol('RELIANCE', '2024-01-15', '2024-01-17')
        
        # Assertions
        self.assertFalse(result['success'])
        self.assertIsNotNone(result['error'])
        self.assertEqual(result['symbol'], 'RELIANCE')
    
    def test_error_handling_in_get_sentiment_for_trading_date(self):
        """Test error handling in getting sentiment for trading date."""
        # Setup mock to raise exception
        self.mock_data_storage.load_news_data.side_effect = Exception("Storage Error")
        
        # Test
        result = self.integration.get_sentiment_for_trading_date('RELIANCE', '2024-01-16')
        
        # Assertions
        self.assertEqual(result['sentiment_score'], 0.0)
        self.assertEqual(result['news_count'], 0)
        self.assertEqual(result['sentiment_label'], 'Neutral')
        self.assertIn('error', result)


if __name__ == '__main__':
    unittest.main()