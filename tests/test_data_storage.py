"""
Unit tests for DataStorage class.
Tests CSV-based data storage functionality, validation, and backup capabilities.
"""

import unittest
import tempfile
import shutil
import json
from pathlib import Path
import pandas as pd
from datetime import datetime
import logging

from src.data.data_storage import DataStorage


class TestDataStorage(unittest.TestCase):
    """Test cases for DataStorage class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        self.storage = DataStorage(base_path=self.temp_dir, enable_backup=True)
        
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
            },
            {
                'symbol': 'RELIANCE',
                'title': 'Market outlook positive',
                'summary': 'Analysts upgrade rating',
                'publisher': 'Reuters',
                'publish_time': int(datetime(2023, 1, 2).timestamp()),
                'url': 'https://example.com/news2'
            }
        ]
        
        # Sample indicators data for testing
        self.sample_indicators_data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=3),
            'symbol': ['RELIANCE', 'RELIANCE', 'RELIANCE'],
            'rsi_14': [45.5, 52.3, 48.7],
            'sma_50': [102.5, 103.2, 104.1],
            'bb_upper': [110.0, 111.0, 112.0],
            'bb_middle': [105.0, 106.0, 107.0],
            'bb_lower': [100.0, 101.0, 102.0]
        })
    
    def tearDown(self):
        """Clean up test fixtures."""
        # Remove temporary directory
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test DataStorage initialization."""
        # Check directory structure was created
        base_path = Path(self.temp_dir)
        expected_dirs = ['stocks', 'news', 'indicators', 'backups', 'temp']
        
        for dir_name in expected_dirs:
            self.assertTrue((base_path / dir_name).exists())
            
        # Check metadata file exists
        self.assertTrue((base_path / "metadata.json").exists())
        
        # Check metadata structure
        self.assertIn("created", self.storage.metadata)
        self.assertIn("symbols", self.storage.metadata)
        self.assertIn("version", self.storage.metadata)
    
    def test_validate_symbol(self):
        """Test symbol validation and cleaning."""
        # Test basic symbol
        self.assertEqual(self.storage._validate_symbol('RELIANCE'), 'RELIANCE')
        
        # Test symbol with .NS suffix
        self.assertEqual(self.storage._validate_symbol('TCS.NS'), 'TCS')
        
        # Test symbol with .BO suffix
        self.assertEqual(self.storage._validate_symbol('INFY.BO'), 'INFY')
        
        # Test lowercase symbol
        self.assertEqual(self.storage._validate_symbol('hdfc'), 'HDFC')
        
        # Test symbol with whitespace
        self.assertEqual(self.storage._validate_symbol(' WIPRO '), 'WIPRO')
        
        # Test symbol with invalid characters
        self.assertEqual(self.storage._validate_symbol('TEST<>:'), 'TEST___')
        
        # Test invalid symbols
        with self.assertRaises(ValueError):
            self.storage._validate_symbol('')
        
        with self.assertRaises(ValueError):
            self.storage._validate_symbol(None)
    
    def test_save_and_load_stock_data(self):
        """Test saving and loading stock data."""
        # Save stock data
        result = self.storage.save_stock_data('RELIANCE', self.sample_stock_data)
        self.assertTrue(result)
        
        # Check file was created
        file_path = self.storage._get_stock_file_path('RELIANCE')
        self.assertTrue(file_path.exists())
        
        # Load stock data
        loaded_data = self.storage.load_stock_data('RELIANCE')
        self.assertFalse(loaded_data.empty)
        self.assertEqual(len(loaded_data), len(self.sample_stock_data))
        
        # Check data integrity
        self.assertIn('symbol', loaded_data.columns)
        self.assertIn('open', loaded_data.columns)
        self.assertIn('close', loaded_data.columns)
        
        # Check metadata was updated
        symbol_info = self.storage.get_symbol_info('RELIANCE')
        self.assertIn('stock', symbol_info)
        self.assertEqual(symbol_info['stock']['record_count'], len(self.sample_stock_data))
    
    def test_save_invalid_stock_data(self):
        """Test handling of invalid stock data."""
        # Test empty data
        empty_data = pd.DataFrame()
        result = self.storage.save_stock_data('TEST', empty_data)
        self.assertFalse(result)
        
        # Test missing required columns
        invalid_data = pd.DataFrame({
            'open': [100.0, 101.0],
            'high': [105.0, 106.0]
            # Missing low, close, volume
        })
        result = self.storage.save_stock_data('TEST', invalid_data)
        self.assertFalse(result)
        
        # Test negative values
        invalid_data = self.sample_stock_data.copy()
        invalid_data.loc[0, 'close'] = -10.0
        result = self.storage.save_stock_data('TEST', invalid_data)
        self.assertFalse(result)
        
        # Test high < low
        invalid_data = self.sample_stock_data.copy()
        invalid_data.loc[0, 'high'] = 90.0  # Less than low
        result = self.storage.save_stock_data('TEST', invalid_data)
        self.assertFalse(result)
    
    def test_save_and_load_news_data(self):
        """Test saving and loading news data."""
        # Save news data
        result = self.storage.save_news_data('RELIANCE', self.sample_news_data)
        self.assertTrue(result)
        
        # Check file was created
        file_path = self.storage._get_news_file_path('RELIANCE')
        self.assertTrue(file_path.exists())
        
        # Load news data
        loaded_news = self.storage.load_news_data('RELIANCE')
        self.assertEqual(len(loaded_news), len(self.sample_news_data))
        
        # Check data structure
        news_item = loaded_news[0]
        self.assertIn('title', news_item)
        self.assertIn('publisher', news_item)
        self.assertIn('publish_time', news_item)
        
        # Check metadata was updated
        symbol_info = self.storage.get_symbol_info('RELIANCE')
        self.assertIn('news', symbol_info)
        self.assertEqual(symbol_info['news']['record_count'], len(self.sample_news_data))
    
    def test_save_empty_news_data(self):
        """Test handling of empty news data."""
        result = self.storage.save_news_data('TEST', [])
        self.assertTrue(result)  # Should succeed but not create file
        
        # Check no file was created
        file_path = self.storage._get_news_file_path('TEST')
        self.assertFalse(file_path.exists())
    
    def test_save_and_load_indicators_data(self):
        """Test saving and loading indicators data."""
        # Save indicators data
        result = self.storage.save_indicators_data('RELIANCE', self.sample_indicators_data)
        self.assertTrue(result)
        
        # Check file was created
        file_path = self.storage._get_indicators_file_path('RELIANCE')
        self.assertTrue(file_path.exists())
        
        # Load indicators data
        loaded_data = self.storage.load_indicators_data('RELIANCE')
        self.assertFalse(loaded_data.empty)
        self.assertEqual(len(loaded_data), len(self.sample_indicators_data))
        
        # Check data integrity
        self.assertIn('rsi_14', loaded_data.columns)
        self.assertIn('sma_50', loaded_data.columns)
        self.assertIn('bb_upper', loaded_data.columns)
        
        # Check metadata was updated
        symbol_info = self.storage.get_symbol_info('RELIANCE')
        self.assertIn('indicators', symbol_info)
        self.assertEqual(symbol_info['indicators']['record_count'], len(self.sample_indicators_data))
    
    def test_load_nonexistent_data(self):
        """Test loading data that doesn't exist."""
        # Load nonexistent stock data
        stock_data = self.storage.load_stock_data('NONEXISTENT')
        self.assertTrue(stock_data.empty)
        
        # Load nonexistent news data
        news_data = self.storage.load_news_data('NONEXISTENT')
        self.assertEqual(len(news_data), 0)
        
        # Load nonexistent indicators data
        indicators_data = self.storage.load_indicators_data('NONEXISTENT')
        self.assertTrue(indicators_data.empty)
    
    def test_backup_functionality(self):
        """Test backup creation and cleanup."""
        # Save data first time
        self.storage.save_stock_data('RELIANCE', self.sample_stock_data)
        
        # Save data second time (should create backup)
        modified_data = self.sample_stock_data.copy()
        modified_data.loc[0, 'close'] = 999.0
        self.storage.save_stock_data('RELIANCE', modified_data)
        
        # Check backup was created
        backup_dir = Path(self.temp_dir) / "backups"
        backup_files = list(backup_dir.glob("RELIANCE_*.csv"))
        self.assertGreater(len(backup_files), 0)
    
    def test_get_available_symbols(self):
        """Test getting list of available symbols."""
        # Initially should be empty
        symbols = self.storage.get_available_symbols()
        self.assertEqual(len(symbols), 0)
        
        # Save data for multiple symbols
        self.storage.save_stock_data('RELIANCE', self.sample_stock_data)
        self.storage.save_news_data('TCS', self.sample_news_data)
        
        # Check symbols are listed
        symbols = self.storage.get_available_symbols()
        self.assertIn('RELIANCE', symbols)
        self.assertIn('TCS', symbols)
    
    def test_delete_symbol_data(self):
        """Test deleting symbol data."""
        # Save data for symbol
        self.storage.save_stock_data('RELIANCE', self.sample_stock_data)
        self.storage.save_news_data('RELIANCE', self.sample_news_data)
        
        # Delete specific data type
        result = self.storage.delete_symbol_data('RELIANCE', 'stock')
        self.assertTrue(result)
        
        # Check stock file was deleted
        stock_file = self.storage._get_stock_file_path('RELIANCE')
        self.assertFalse(stock_file.exists())
        
        # Check news file still exists
        news_file = self.storage._get_news_file_path('RELIANCE')
        self.assertTrue(news_file.exists())
        
        # Delete all data for symbol
        result = self.storage.delete_symbol_data('RELIANCE')
        self.assertTrue(result)
        
        # Check news file was deleted
        self.assertFalse(news_file.exists())
        
        # Check symbol removed from metadata
        symbols = self.storage.get_available_symbols()
        self.assertNotIn('RELIANCE', symbols)
    
    def test_get_storage_stats(self):
        """Test storage statistics calculation."""
        # Save some data
        self.storage.save_stock_data('RELIANCE', self.sample_stock_data)
        self.storage.save_news_data('RELIANCE', self.sample_news_data)
        self.storage.save_indicators_data('TCS', self.sample_indicators_data)
        
        # Get stats
        stats = self.storage.get_storage_stats()
        
        # Check stats structure
        self.assertIn('total_symbols', stats)
        self.assertIn('total_files', stats)
        self.assertIn('total_size_mb', stats)
        self.assertIn('data_types', stats)
        
        # Check values
        self.assertGreater(stats['total_files'], 0)
        self.assertGreater(stats['total_size_mb'], 0)
        self.assertGreater(stats['data_types']['stock'], 0)
        self.assertGreater(stats['data_types']['news'], 0)
        self.assertGreater(stats['data_types']['indicators'], 0)
    
    def test_metadata_persistence(self):
        """Test metadata persistence across instances."""
        # Save data
        self.storage.save_stock_data('RELIANCE', self.sample_stock_data)
        
        # Create new storage instance
        new_storage = DataStorage(base_path=self.temp_dir)
        
        # Check metadata was loaded
        symbols = new_storage.get_available_symbols()
        self.assertIn('RELIANCE', symbols)
        
        symbol_info = new_storage.get_symbol_info('RELIANCE')
        self.assertIn('stock', symbol_info)
    
    def test_file_path_generation(self):
        """Test file path generation for different data types."""
        base_path = Path(self.temp_dir)
        
        # Test stock file path
        stock_path = self.storage._get_stock_file_path('RELIANCE')
        expected_stock_path = base_path / "stocks" / "RELIANCE.csv"
        self.assertEqual(stock_path, expected_stock_path)
        
        # Test news file path
        news_path = self.storage._get_news_file_path('TCS')
        expected_news_path = base_path / "news" / "TCS_news.csv"
        self.assertEqual(news_path, expected_news_path)
        
        # Test indicators file path
        indicators_path = self.storage._get_indicators_file_path('HDFC')
        expected_indicators_path = base_path / "indicators" / "HDFC_indicators.csv"
        self.assertEqual(indicators_path, expected_indicators_path)
    
    def test_data_validation(self):
        """Test comprehensive data validation."""
        # Test valid stock data
        valid_data = self.sample_stock_data.copy()
        self.assertTrue(self.storage._validate_stock_data(valid_data))
        
        # Test empty data
        empty_data = pd.DataFrame()
        self.assertFalse(self.storage._validate_stock_data(empty_data))
        
        # Test missing columns
        incomplete_data = valid_data[['open', 'high']].copy()
        self.assertFalse(self.storage._validate_stock_data(incomplete_data))
        
        # Test negative values
        negative_data = valid_data.copy()
        negative_data.loc[0, 'volume'] = -1000
        self.assertFalse(self.storage._validate_stock_data(negative_data))
        
        # Test invalid OHLC relationships
        invalid_ohlc = valid_data.copy()
        invalid_ohlc.loc[0, 'high'] = 50.0  # Less than low
        self.assertFalse(self.storage._validate_stock_data(invalid_ohlc))


if __name__ == '__main__':
    # Configure logging for tests
    logging.basicConfig(level=logging.WARNING)
    unittest.main()