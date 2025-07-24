"""
Integration tests for feature builder and indicator integration system.
Tests the complete indicator calculation pipeline.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import shutil
import sys
import os

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.feature_builder import FeatureBuilder
from data.data_storage import DataStorage
from data.technical_indicators import TechnicalIndicators


class TestFeatureBuilder(unittest.TestCase):
    """Integration tests for FeatureBuilder class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temporary directory for test data storage
        self.temp_dir = tempfile.mkdtemp()
        self.data_storage = DataStorage(base_path=self.temp_dir, enable_backup=False)
        self.feature_builder = FeatureBuilder(data_storage=self.data_storage)
        
        # Create sample stock data
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        
        # Create realistic price data with trends
        base_price = 100
        prices = []
        for i in range(100):
            # Add trend and some volatility
            trend = i * 0.1
            volatility = np.sin(i * 0.1) * 2
            noise = np.random.normal(0, 0.5)
            price = base_price + trend + volatility + noise
            prices.append(max(price, 1))  # Ensure positive prices
        
        self.sample_data = pd.DataFrame({
            'date': dates,
            'symbol': ['TEST'] * 100,
            'open': [p + np.random.normal(0, 0.2) for p in prices],
            'high': [p + abs(np.random.normal(0, 0.5)) for p in prices],
            'low': [p - abs(np.random.normal(0, 0.5)) for p in prices],
            'close': prices,
            'volume': [1000 + i * 10 + np.random.randint(-100, 100) for i in range(100)]
        })
        
        # Ensure high >= low
        self.sample_data['high'] = np.maximum(self.sample_data['high'], self.sample_data['low'])
        
        # Create sample news data
        self.sample_news = [
            {'title': 'Positive news about TEST stock', 'publish_time': 1640995200},
            {'title': 'Market outlook remains strong', 'publish_time': 1641081600},
            {'title': 'Company reports good earnings', 'publish_time': 1641168000}
        ]
        
        # Save sample data to storage
        self.data_storage.save_stock_data('TEST', self.sample_data)
        self.data_storage.save_news_data('TEST', self.sample_news)
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_calculate_technical_indicators_integration(self):
        """Test technical indicators calculation integration."""
        result = self.feature_builder.calculate_technical_indicators(self.sample_data)
        
        # Check that all indicator columns are added
        expected_indicators = ['rsi_14', 'sma_50', 'bb_upper', 'bb_middle', 'bb_lower']
        for indicator in expected_indicators:
            self.assertIn(indicator, result.columns)
        
        # Check that original columns are preserved
        for col in self.sample_data.columns:
            self.assertIn(col, result.columns)
        
        # Check that indicators have some valid values
        self.assertGreater(result['rsi_14'].notna().sum(), 0)
        self.assertGreater(result['sma_50'].notna().sum(), 0)
        self.assertGreater(result['bb_middle'].notna().sum(), 0)
    
    def test_calculate_technical_indicators_error_handling(self):
        """Test error handling in technical indicators calculation."""
        # Test with empty data
        with self.assertRaises(ValueError):
            self.feature_builder.calculate_technical_indicators(pd.DataFrame())
        
        # Test with missing columns
        invalid_data = pd.DataFrame({'invalid_col': [1, 2, 3]})
        with self.assertRaises(ValueError):
            self.feature_builder.calculate_technical_indicators(invalid_data)
    
    def test_calculate_sentiment_scores_placeholder(self):
        """Test sentiment scores calculation (placeholder implementation)."""
        scores = self.feature_builder.calculate_sentiment_scores(self.sample_news)
        
        # Should return neutral scores (placeholder implementation)
        self.assertEqual(len(scores), len(self.sample_news))
        self.assertTrue(all(score == 0.0 for score in scores))
        
        # Test with empty news
        empty_scores = self.feature_builder.calculate_sentiment_scores([])
        self.assertEqual(len(empty_scores), 0)
    
    def test_build_features_basic(self):
        """Test basic feature building functionality."""
        # First calculate indicators
        data_with_indicators = self.feature_builder.calculate_technical_indicators(self.sample_data)
        
        # Calculate sentiment scores
        sentiment_scores = [0.1, 0.2, -0.1] + [0.0] * 97  # Match data length
        
        # Build features
        features = self.feature_builder.build_features(data_with_indicators, sentiment_scores)
        
        # Check that sentiment scores are added
        self.assertIn('sentiment_score', features.columns)
        self.assertEqual(features['sentiment_score'].iloc[0], 0.1)
        
        # Check that derived features are added
        derived_features = ['price_change', 'price_return', 'volatility_10d', 'bb_position']
        for feature in derived_features:
            self.assertIn(feature, features.columns)
    
    def test_build_features_mismatched_sentiment(self):
        """Test feature building with mismatched sentiment data length."""
        data_with_indicators = self.feature_builder.calculate_technical_indicators(self.sample_data)
        
        # Provide sentiment data with wrong length
        wrong_length_sentiment = [0.1, 0.2]  # Only 2 values for 100 data points
        
        features = self.feature_builder.build_features(data_with_indicators, wrong_length_sentiment)
        
        # Should fill with neutral sentiment
        self.assertIn('sentiment_score', features.columns)
        self.assertTrue((features['sentiment_score'] == 0.0).all())
    
    def test_build_features_no_sentiment(self):
        """Test feature building without sentiment data."""
        data_with_indicators = self.feature_builder.calculate_technical_indicators(self.sample_data)
        
        features = self.feature_builder.build_features(data_with_indicators, [])
        
        # Should fill with neutral sentiment
        self.assertIn('sentiment_score', features.columns)
        self.assertTrue((features['sentiment_score'] == 0.0).all())
    
    def test_process_symbol_features_complete_pipeline(self):
        """Test complete feature processing pipeline for a symbol."""
        features = self.feature_builder.process_symbol_features('TEST', save_indicators=True)
        
        # Check that features are generated
        self.assertFalse(features.empty)
        self.assertEqual(len(features), len(self.sample_data))
        
        # Check that all expected columns exist
        expected_columns = [
            'open', 'high', 'low', 'close', 'volume',  # Original price data
            'rsi_14', 'sma_50', 'bb_upper', 'bb_middle', 'bb_lower',  # Technical indicators
            'sentiment_score',  # Sentiment
            'price_change', 'price_return', 'volatility_10d'  # Derived features
        ]
        
        for col in expected_columns:
            self.assertIn(col, features.columns)
        
        # Check that indicators were saved to storage
        saved_indicators = self.data_storage.load_indicators_data('TEST')
        self.assertFalse(saved_indicators.empty)
    
    def test_process_symbol_features_with_provided_data(self):
        """Test feature processing with provided data (not loaded from storage)."""
        features = self.feature_builder.process_symbol_features(
            'TEST2',  # Symbol not in storage
            price_data=self.sample_data,
            news_data=self.sample_news,
            save_indicators=False
        )
        
        # Should still generate features
        self.assertFalse(features.empty)
        self.assertEqual(len(features), len(self.sample_data))
    
    def test_process_symbol_features_missing_data(self):
        """Test feature processing with missing data."""
        # Try to process symbol that doesn't exist in storage
        features = self.feature_builder.process_symbol_features('NONEXISTENT')
        
        # Should return empty DataFrame
        self.assertTrue(features.empty)
    
    def test_process_multiple_symbols(self):
        """Test processing features for multiple symbols."""
        # Add another symbol to storage
        sample_data_2 = self.sample_data.copy()
        sample_data_2['symbol'] = 'TEST2'
        self.data_storage.save_stock_data('TEST2', sample_data_2)
        
        results = self.feature_builder.process_multiple_symbols(['TEST', 'TEST2', 'NONEXISTENT'])
        
        # Should process existing symbols successfully
        self.assertIn('TEST', results)
        self.assertIn('TEST2', results)
        self.assertNotIn('NONEXISTENT', results)  # Should skip missing symbols
        
        # Check that features are generated for existing symbols
        self.assertFalse(results['TEST'].empty)
        self.assertFalse(results['TEST2'].empty)
    
    def test_validate_features(self):
        """Test feature validation functionality."""
        features = self.feature_builder.process_symbol_features('TEST')
        validation_results = self.feature_builder.validate_features(features)
        
        # Check validation structure
        self.assertIn('valid', validation_results)
        self.assertIn('total_rows', validation_results)
        self.assertIn('total_columns', validation_results)
        self.assertIn('missing_data', validation_results)
        self.assertIn('data_quality', validation_results)
        self.assertIn('warnings', validation_results)
        
        # Should be valid
        self.assertTrue(validation_results['valid'])
        self.assertEqual(validation_results['total_rows'], len(features))
        
        # Check missing data analysis
        self.assertIsInstance(validation_results['missing_data'], dict)
        
        # Check data quality for RSI
        if 'rsi_14' in validation_results['data_quality']:
            rsi_quality = validation_results['data_quality']['rsi_14']
            self.assertIn('min', rsi_quality)
            self.assertIn('max', rsi_quality)
            self.assertIn('valid_count', rsi_quality)
    
    def test_validate_features_empty_data(self):
        """Test feature validation with empty data."""
        validation_results = self.feature_builder.validate_features(pd.DataFrame())
        
        self.assertFalse(validation_results['valid'])
        self.assertIn('error', validation_results)
    
    def test_get_feature_summary(self):
        """Test feature summary generation."""
        features = self.feature_builder.process_symbol_features('TEST')
        summary = self.feature_builder.get_feature_summary(features)
        
        # Check summary structure
        self.assertIn('shape', summary)
        self.assertIn('date_range', summary)
        self.assertIn('feature_stats', summary)
        self.assertIn('data_completeness', summary)
        
        # Check shape information
        self.assertEqual(summary['shape'], features.shape)
        
        # Check date range (if date column exists)
        if 'date' in features.columns:
            self.assertIn('start', summary['date_range'])
            self.assertIn('end', summary['date_range'])
            self.assertIn('days', summary['date_range'])
        
        # Check feature statistics
        self.assertIsInstance(summary['feature_stats'], dict)
        
        # Check data completeness
        self.assertIsInstance(summary['data_completeness'], dict)
        for col in features.columns:
            self.assertIn(col, summary['data_completeness'])
    
    def test_get_feature_summary_empty_data(self):
        """Test feature summary with empty data."""
        summary = self.feature_builder.get_feature_summary(pd.DataFrame())
        
        self.assertIn('error', summary)
    
    def test_derived_features_calculation(self):
        """Test derived features calculation."""
        data_with_indicators = self.feature_builder.calculate_technical_indicators(self.sample_data)
        features = self.feature_builder.build_features(data_with_indicators, [])
        
        # Check price-based derived features
        self.assertIn('price_change', features.columns)
        self.assertIn('price_return', features.columns)
        self.assertIn('volatility_10d', features.columns)
        
        # Check that price changes are calculated correctly
        expected_change = features['close'].diff()
        pd.testing.assert_series_equal(features['price_change'], expected_change, check_names=False)
        
        # Check that returns are calculated correctly
        expected_return = features['close'].pct_change()
        pd.testing.assert_series_equal(features['price_return'], expected_return, check_names=False)
        
        # Check Bollinger Band derived features
        if features['bb_upper'].notna().any():
            self.assertIn('bb_position', features.columns)
            self.assertIn('bb_width', features.columns)
            
            # BB position can be outside 0-1 range when price is outside bands
            # Just check that we have valid numeric values
            valid_bb_pos = features['bb_position'].dropna()
            if len(valid_bb_pos) > 0:
                self.assertTrue(np.isfinite(valid_bb_pos).all())
                
            # BB width should be positive
            valid_bb_width = features['bb_width'].dropna()
            if len(valid_bb_width) > 0:
                self.assertTrue((valid_bb_width > 0).all())
    
    def test_error_handling_integration(self):
        """Test error handling throughout the integration pipeline."""
        # Test with corrupted data
        corrupted_data = self.sample_data.copy()
        corrupted_data.loc[10:20, 'close'] = np.nan
        corrupted_data.loc[30:40, 'high'] = -1  # Invalid negative high
        
        # Should handle errors gracefully
        try:
            features = self.feature_builder.calculate_technical_indicators(corrupted_data)
            self.assertIsInstance(features, pd.DataFrame)
            
            # Build features should also handle errors
            final_features = self.feature_builder.build_features(features, [])
            self.assertIsInstance(final_features, pd.DataFrame)
            
        except Exception as e:
            self.fail(f"Error handling failed: {str(e)}")
    
    def test_storage_integration(self):
        """Test integration with data storage system."""
        # Process features and save indicators
        features = self.feature_builder.process_symbol_features('TEST', save_indicators=True)
        
        # Verify indicators were saved
        saved_indicators = self.data_storage.load_indicators_data('TEST')
        self.assertFalse(saved_indicators.empty)
        self.assertEqual(len(saved_indicators), len(features))
        
        # Check that key indicator columns are saved
        indicator_columns = ['rsi_14', 'sma_50', 'bb_upper', 'bb_middle', 'bb_lower']
        for col in indicator_columns:
            self.assertIn(col, saved_indicators.columns)
        
        # Verify we can load and use saved indicators
        loaded_features = self.data_storage.load_indicators_data('TEST')
        self.assertFalse(loaded_features.empty)


if __name__ == '__main__':
    unittest.main()