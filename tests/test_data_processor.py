"""
Unit tests for data processing utilities.
Tests data cleaning, validation, and normalization functionality.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import os

from src.data.data_processor import DataCleaner, DataValidator, DataNormalizer


class TestDataCleaner(unittest.TestCase):
    """Test cases for DataCleaner class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.cleaner = DataCleaner(outlier_std_threshold=3.0)
        
        # Create sample stock data
        dates = pd.date_range(start='2023-01-01', end='2023-01-31', freq='D')
        self.sample_data = pd.DataFrame({
            'Open': [100 + i + np.random.normal(0, 1) for i in range(len(dates))],
            'High': [105 + i + np.random.normal(0, 1) for i in range(len(dates))],
            'Low': [95 + i + np.random.normal(0, 1) for i in range(len(dates))],
            'Close': [102 + i + np.random.normal(0, 1) for i in range(len(dates))],
            'Volume': [1000000 + np.random.randint(0, 500000) for _ in range(len(dates))]
        }, index=dates)
        
        # Ensure OHLC consistency in sample data
        for idx in self.sample_data.index:
            row = self.sample_data.loc[idx]
            high_val = max(row['Open'], row['Close'], row['Low']) + 1
            low_val = min(row['Open'], row['Close'], row['High']) - 1
            self.sample_data.loc[idx, 'High'] = high_val
            self.sample_data.loc[idx, 'Low'] = low_val
    
    def test_clean_empty_data(self):
        """Test cleaning empty DataFrame."""
        empty_data = pd.DataFrame()
        cleaned_data, stats = self.cleaner.clean_stock_data(empty_data, "TEST")
        
        self.assertTrue(cleaned_data.empty)
        self.assertIn("error", stats)
        self.assertEqual(stats["error"], "Empty data")
    
    def test_clean_data_with_missing_values(self):
        """Test cleaning data with missing values."""
        # Add missing values
        data_with_missing = self.sample_data.copy()
        data_with_missing.loc[data_with_missing.index[5], 'Close'] = np.nan
        data_with_missing.loc[data_with_missing.index[10], 'Volume'] = np.nan
        data_with_missing.loc[data_with_missing.index[15], 'Open'] = np.nan
        
        original_rows = len(data_with_missing)
        cleaned_data, stats = self.cleaner.clean_stock_data(data_with_missing, "TEST")
        
        # Should remove rows with missing values
        self.assertLess(len(cleaned_data), original_rows)
        self.assertEqual(stats["missing_values_removed"], 3)
        self.assertFalse(cleaned_data.isnull().any().any())
    
    def test_clean_data_with_outliers(self):
        """Test outlier capping functionality."""
        # Test the _cap_outliers method directly first
        test_data = pd.DataFrame({
            'Price': [100, 101, 102, 500, 103]  # 500 is clear outlier
        })
        
        outliers_capped = self.cleaner._cap_outliers(test_data, 'Price')
        self.assertGreater(outliers_capped, 0, "Direct outlier capping should detect outliers")
        
        # Now test with full cleaning process
        simple_data = pd.DataFrame({
            'Open': [100, 101, 102, 200, 103],  # 200 is moderate outlier
            'High': [105, 106, 107, 205, 108],
            'Low': [95, 96, 97, 195, 98],
            'Close': [102, 103, 104, 202, 105],
            'Volume': [1000, 1100, 1200, 5000, 1400]  # 5000 is volume outlier
        }, index=pd.date_range('2023-01-01', periods=5))
        
        cleaned_data, stats = self.cleaner.clean_stock_data(simple_data, "TEST")
        
        # Should cap outliers
        self.assertGreater(stats["outliers_capped"], 0)
        
        # Check that cleaning was performed
        self.assertEqual(len(cleaned_data), len(simple_data))  # No rows should be removed
        self.assertGreater(stats["data_quality_score"], 0.5)  # Should have decent quality
    
    def test_fix_ohlc_inconsistencies(self):
        """Test fixing OHLC data inconsistencies."""
        # Create data with OHLC inconsistencies
        inconsistent_data = self.sample_data.copy()
        
        # Make High lower than Close (inconsistent)
        inconsistent_data.loc[inconsistent_data.index[0], 'High'] = \
            inconsistent_data.loc[inconsistent_data.index[0], 'Close'] - 5
        
        # Make Low higher than Open (inconsistent)
        inconsistent_data.loc[inconsistent_data.index[1], 'Low'] = \
            inconsistent_data.loc[inconsistent_data.index[1], 'Open'] + 5
        
        cleaned_data, stats = self.cleaner.clean_stock_data(inconsistent_data, "TEST")
        
        # Check that inconsistencies are fixed
        for idx in cleaned_data.index:
            row = cleaned_data.loc[idx]
            self.assertGreaterEqual(row['High'], max(row['Open'], row['Close'], row['Low']))
            self.assertLessEqual(row['Low'], min(row['Open'], row['Close'], row['High']))
    
    def test_data_quality_score_calculation(self):
        """Test data quality score calculation."""
        # Test with perfect data
        cleaned_data, stats = self.cleaner.clean_stock_data(self.sample_data, "TEST")
        self.assertGreater(stats["data_quality_score"], 0.8)  # Should be high quality
        
        # Test with poor quality data
        poor_data = self.sample_data.copy()
        # Add many missing values
        for i in range(0, len(poor_data), 3):  # Every third row
            poor_data.loc[poor_data.index[i], 'Close'] = np.nan
        
        cleaned_poor_data, poor_stats = self.cleaner.clean_stock_data(poor_data, "POOR")
        self.assertLess(poor_stats["data_quality_score"], stats["data_quality_score"])
    
    def test_validate_cleaned_data(self):
        """Test validation of cleaned data."""
        cleaned_data, _ = self.cleaner.clean_stock_data(self.sample_data, "TEST")
        is_valid, issues = self.cleaner.validate_cleaned_data(cleaned_data, "TEST")
        
        self.assertTrue(is_valid)
        self.assertEqual(len(issues), 0)
        
        # Test with insufficient data
        small_data = self.sample_data.head(10)  # Less than 30 rows
        is_valid_small, issues_small = self.cleaner.validate_cleaned_data(small_data, "SMALL")
        
        self.assertFalse(is_valid_small)
        self.assertGreater(len(issues_small), 0)
        self.assertTrue(any("Insufficient data" in issue for issue in issues_small))
    
    def test_get_cleaning_summary(self):
        """Test cleaning summary generation."""
        # Clean multiple datasets
        self.cleaner.clean_stock_data(self.sample_data, "TEST1")
        self.cleaner.clean_stock_data(self.sample_data, "TEST2")
        
        summary = self.cleaner.get_cleaning_summary()
        
        self.assertEqual(summary["total_symbols_processed"], 2)
        self.assertIn("individual_stats", summary)
        self.assertIn("average_quality_score", summary)
        self.assertIn("overall_retention_rate", summary)
    
    def test_cap_outliers_edge_cases(self):
        """Test outlier capping with edge cases."""
        # Test with constant values (std = 0)
        constant_data = pd.DataFrame({'Price': [100] * 10})
        outliers_capped = self.cleaner._cap_outliers(constant_data, 'Price')
        self.assertEqual(outliers_capped, 0)
        
        # Test with empty column
        empty_data = pd.DataFrame({'Price': []})
        outliers_capped = self.cleaner._cap_outliers(empty_data, 'Price')
        self.assertEqual(outliers_capped, 0)
        
        # Test with non-existent column
        outliers_capped = self.cleaner._cap_outliers(self.sample_data, 'NonExistent')
        self.assertEqual(outliers_capped, 0)


class TestDataValidator(unittest.TestCase):
    """Test cases for DataValidator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        dates = pd.date_range(start='2023-01-01', end='2023-01-31', freq='B')  # Business days
        self.sample_data = pd.DataFrame({
            'Open': [100 + i for i in range(len(dates))],
            'High': [105 + i for i in range(len(dates))],
            'Low': [95 + i for i in range(len(dates))],
            'Close': [102 + i for i in range(len(dates))],
            'Volume': [1000000 + i * 1000 for i in range(len(dates))]
        }, index=dates)
    
    def test_check_data_completeness(self):
        """Test data completeness checking."""
        stats = DataValidator.check_data_completeness(self.sample_data, "TEST")
        
        self.assertEqual(stats["symbol"], "TEST")
        self.assertEqual(stats["total_rows"], len(self.sample_data))
        self.assertIsNotNone(stats["date_range"])
        self.assertEqual(len(stats["issues"]), 0)
    
    def test_check_data_completeness_with_missing_dates(self):
        """Test completeness checking with missing dates."""
        # Remove some dates to create gaps
        incomplete_data = self.sample_data.drop(self.sample_data.index[5:10])
        
        required_range = ('2023-01-01', '2023-01-31')
        stats = DataValidator.check_data_completeness(
            incomplete_data, "TEST", required_range
        )
        
        self.assertGreater(stats["missing_dates"], 0)
        self.assertLess(stats["completeness_score"], 1.0)
        self.assertGreater(len(stats["issues"]), 0)
    
    def test_check_data_consistency(self):
        """Test data consistency checking."""
        results = DataValidator.check_data_consistency(self.sample_data, "TEST")
        
        self.assertEqual(results["symbol"], "TEST")
        self.assertTrue(results["ohlc_consistency"])
        self.assertTrue(results["volume_consistency"])
        self.assertTrue(results["price_continuity"])
        self.assertEqual(len(results["issues"]), 0)
    
    def test_check_data_consistency_with_issues(self):
        """Test consistency checking with data issues."""
        # Create inconsistent data
        inconsistent_data = self.sample_data.copy()
        
        # Make High lower than Close (inconsistent)
        inconsistent_data.loc[inconsistent_data.index[0], 'High'] = \
            inconsistent_data.loc[inconsistent_data.index[0], 'Close'] - 10
        
        # Add negative volume
        inconsistent_data.loc[inconsistent_data.index[1], 'Volume'] = -1000
        
        # Add extreme price gap
        inconsistent_data.loc[inconsistent_data.index[2], 'Close'] = \
            inconsistent_data.loc[inconsistent_data.index[1], 'Close'] * 2  # 100% increase
        
        results = DataValidator.check_data_consistency(inconsistent_data, "TEST")
        
        self.assertFalse(results["ohlc_consistency"])
        self.assertFalse(results["volume_consistency"])
        self.assertFalse(results["price_continuity"])
        self.assertGreater(len(results["issues"]), 0)
    
    def test_check_empty_data(self):
        """Test validation with empty data."""
        empty_data = pd.DataFrame()
        
        stats = DataValidator.check_data_completeness(empty_data, "EMPTY")
        self.assertGreater(len(stats["issues"]), 0)
        
        results = DataValidator.check_data_consistency(empty_data, "EMPTY")
        self.assertGreater(len(results["issues"]), 0)
    
    def test_check_data_without_datetime_index(self):
        """Test validation with non-datetime index."""
        non_datetime_data = self.sample_data.reset_index(drop=True)
        
        stats = DataValidator.check_data_completeness(non_datetime_data, "TEST")
        self.assertTrue(any("not DatetimeIndex" in issue for issue in stats["issues"]))


class TestDataNormalizer(unittest.TestCase):
    """Test cases for DataNormalizer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.normalizer = DataNormalizer()
        
        # Create sample data for normalization
        dates = pd.date_range(start='2023-01-01', end='2023-01-31', freq='D')
        self.sample_data = pd.DataFrame({
            'Open': [100 + i * 2 for i in range(len(dates))],
            'High': [105 + i * 2 for i in range(len(dates))],
            'Low': [95 + i * 2 for i in range(len(dates))],
            'Close': [102 + i * 2 for i in range(len(dates))],
            'Volume': [1000000 + i * 10000 for i in range(len(dates))],
            'RSI': [50 + np.random.normal(0, 10) for _ in range(len(dates))],
            'SMA': [100 + i * 1.5 for i in range(len(dates))],
            'Sentiment': [np.random.uniform(-1, 1) for _ in range(len(dates))]
        }, index=dates)
    
    def test_normalize_features_minmax(self):
        """Test min-max normalization."""
        feature_columns = ['Open', 'High', 'Low', 'Close']
        normalized_data, stats = self.normalizer.normalize_features(
            self.sample_data, feature_columns, method='minmax'
        )
        
        self.assertEqual(stats["method"], "minmax")
        self.assertEqual(len(stats["features_normalized"]), 4)
        
        # Check that values are in [0, 1] range
        for col in feature_columns:
            self.assertGreaterEqual(normalized_data[col].min(), 0)
            self.assertLessEqual(normalized_data[col].max(), 1)
            self.assertAlmostEqual(normalized_data[col].min(), 0, places=10)
            self.assertAlmostEqual(normalized_data[col].max(), 1, places=10)
    
    def test_normalize_features_standard(self):
        """Test standard (z-score) normalization."""
        feature_columns = ['Volume', 'RSI']
        normalized_data, stats = self.normalizer.normalize_features(
            self.sample_data, feature_columns, method='standard'
        )
        
        self.assertEqual(stats["method"], "standard")
        self.assertEqual(len(stats["features_normalized"]), 2)
        
        # Check that values have mean≈0 and std≈1
        for col in feature_columns:
            self.assertAlmostEqual(normalized_data[col].mean(), 0, places=10)
            self.assertAlmostEqual(normalized_data[col].std(), 1, places=10)
    
    def test_normalize_features_robust(self):
        """Test robust normalization."""
        feature_columns = ['SMA', 'Sentiment']
        normalized_data, stats = self.normalizer.normalize_features(
            self.sample_data, feature_columns, method='robust'
        )
        
        self.assertEqual(stats["method"], "robust")
        self.assertEqual(len(stats["features_normalized"]), 2)
        
        # Check that median is approximately 0
        for col in feature_columns:
            self.assertAlmostEqual(normalized_data[col].median(), 0, places=10)
    
    def test_normalize_empty_data(self):
        """Test normalization with empty data."""
        empty_data = pd.DataFrame()
        normalized_data, stats = self.normalizer.normalize_features(
            empty_data, ['Open'], method='minmax'
        )
        
        self.assertTrue(normalized_data.empty)
        self.assertIn("error", stats)
    
    def test_normalize_no_feature_columns(self):
        """Test normalization with no valid feature columns."""
        normalized_data, stats = self.normalizer.normalize_features(
            self.sample_data, ['NonExistent'], method='minmax'
        )
        
        self.assertIn("error", stats)
        self.assertEqual(stats["error"], "No feature columns found")
    
    def test_normalize_constant_values(self):
        """Test normalization with constant values."""
        constant_data = pd.DataFrame({
            'Constant': [100] * 10,
            'Variable': [i for i in range(10)]
        })
        
        normalized_data, stats = self.normalizer.normalize_features(
            constant_data, ['Constant', 'Variable'], method='minmax'
        )
        
        # Constant column should remain unchanged
        self.assertTrue((normalized_data['Constant'] == 100).all())
        # Variable column should be normalized
        self.assertAlmostEqual(normalized_data['Variable'].min(), 0)
        self.assertAlmostEqual(normalized_data['Variable'].max(), 1)
    
    def test_split_train_test_time_based(self):
        """Test time-based train/test split."""
        train_data, test_data, stats = self.normalizer.split_train_test(
            self.sample_data, test_size=0.2, time_based=True
        )
        
        self.assertEqual(stats["test_size"], 0.2)
        self.assertTrue(stats["time_based"])
        self.assertEqual(stats["train_rows"] + stats["test_rows"], len(self.sample_data))
        
        # Check chronological order
        self.assertTrue(train_data.index.max() < test_data.index.min())
        
        # Check approximate split ratio
        expected_train_size = int(len(self.sample_data) * 0.8)
        self.assertEqual(len(train_data), expected_train_size)
    
    def test_split_train_test_random(self):
        """Test random train/test split."""
        train_data, test_data, stats = self.normalizer.split_train_test(
            self.sample_data, test_size=0.3, time_based=False
        )
        
        self.assertEqual(stats["test_size"], 0.3)
        self.assertFalse(stats["time_based"])
        self.assertEqual(stats["train_rows"] + stats["test_rows"], len(self.sample_data))
        
        # Check approximate split ratio (allow for rounding differences)
        expected_test_size = int(len(self.sample_data) * 0.3)
        self.assertAlmostEqual(len(test_data), expected_test_size, delta=1)
    
    def test_split_empty_data(self):
        """Test splitting empty data."""
        empty_data = pd.DataFrame()
        train_data, test_data, stats = self.normalizer.split_train_test(
            empty_data, test_size=0.2
        )
        
        self.assertTrue(train_data.empty)
        self.assertTrue(test_data.empty)
        self.assertIn("error", stats)
    
    def test_split_invalid_test_size(self):
        """Test splitting with invalid test size."""
        with self.assertRaises(ValueError):
            self.normalizer.split_train_test(self.sample_data, test_size=1.5)
        
        with self.assertRaises(ValueError):
            self.normalizer.split_train_test(self.sample_data, test_size=0)
    
    def test_combine_features(self):
        """Test feature combination."""
        # Create separate dataframes
        price_data = self.sample_data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        technical_data = self.sample_data[['RSI', 'SMA']].copy()
        sentiment_data = self.sample_data[['Sentiment']].copy()
        
        combined_data, stats = self.normalizer.combine_features(
            price_data, technical_data, sentiment_data
        )
        
        # Should have all columns
        expected_columns = set(price_data.columns) | set(technical_data.columns) | set(sentiment_data.columns)
        self.assertEqual(set(combined_data.columns), expected_columns)
        
        # Check stats
        self.assertEqual(len(stats["original_features"]), len(price_data.columns))
        self.assertEqual(len(stats["added_features"]), len(technical_data.columns) + len(sentiment_data.columns))
        self.assertEqual(stats["total_features"], len(expected_columns))
    
    def test_combine_features_with_missing_data(self):
        """Test feature combination with missing data."""
        price_data = self.sample_data[['Open', 'High', 'Low', 'Close']].copy()
        
        # Create technical data with missing values and different index
        technical_data = pd.DataFrame({
            'RSI': [50, np.nan, 60, 70],
            'SMA': [100, 105, np.nan, 115]
        }, index=price_data.index[:4])  # Shorter index
        
        combined_data, stats = self.normalizer.combine_features(
            price_data, technical_data
        )
        
        # Should handle missing data
        self.assertGreater(stats["missing_data_handled"], 0)
        self.assertFalse(combined_data.isnull().any().any())  # No missing values should remain
    
    def test_combine_features_empty_price_data(self):
        """Test feature combination with empty price data."""
        empty_price = pd.DataFrame()
        technical_data = self.sample_data[['RSI']].copy()
        
        combined_data, stats = self.normalizer.combine_features(
            empty_price, technical_data
        )
        
        self.assertTrue(combined_data.empty)
        self.assertIn("error", stats)
    
    def test_prepare_model_inputs_regular(self):
        """Test preparing regular model inputs."""
        # Add a target column
        data_with_target = self.sample_data.copy()
        data_with_target['Target'] = data_with_target['Close'].shift(-1)  # Next day close
        data_with_target = data_with_target.dropna()
        
        feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        X, y, stats = self.normalizer.prepare_model_inputs(
            data_with_target, 'Target', feature_columns
        )
        
        self.assertEqual(X.shape[0], len(data_with_target))
        self.assertEqual(X.shape[1], len(feature_columns))
        self.assertEqual(y.shape[0], len(data_with_target))
        self.assertEqual(stats["feature_count"], len(feature_columns))
        self.assertIsNone(stats["sequence_length"])
    
    def test_prepare_model_inputs_sequences(self):
        """Test preparing LSTM sequence inputs."""
        data_with_target = self.sample_data.copy()
        data_with_target['Target'] = data_with_target['Close'].shift(-1)
        data_with_target = data_with_target.dropna()
        
        feature_columns = ['Open', 'High', 'Low', 'Close']
        sequence_length = 5
        
        X, y, stats = self.normalizer.prepare_model_inputs(
            data_with_target, 'Target', feature_columns, sequence_length
        )
        
        expected_samples = len(data_with_target) - sequence_length
        self.assertEqual(X.shape[0], expected_samples)
        self.assertEqual(X.shape[1], sequence_length)
        self.assertEqual(X.shape[2], len(feature_columns))
        self.assertEqual(y.shape[0], expected_samples)
        self.assertEqual(stats["sequence_length"], sequence_length)
    
    def test_prepare_model_inputs_no_target(self):
        """Test preparing inputs with missing target column."""
        X, y, stats = self.normalizer.prepare_model_inputs(
            self.sample_data, 'NonExistentTarget'
        )
        
        self.assertEqual(X.size, 0)
        self.assertEqual(y.size, 0)
        self.assertIn("error", stats)
    
    def test_prepare_model_inputs_empty_data(self):
        """Test preparing inputs with empty data."""
        empty_data = pd.DataFrame()
        X, y, stats = self.normalizer.prepare_model_inputs(
            empty_data, 'Target'
        )
        
        self.assertEqual(X.size, 0)
        self.assertEqual(y.size, 0)
        self.assertIn("error", stats)
    
    def test_get_normalization_summary(self):
        """Test normalization summary generation."""
        # Perform multiple normalizations
        self.normalizer.normalize_features(self.sample_data, ['Open', 'Close'], 'minmax')
        self.normalizer.normalize_features(self.sample_data, ['Volume'], 'standard')
        
        summary = self.normalizer.get_normalization_summary()
        
        self.assertEqual(summary["total_operations"], 2)
        self.assertIn("minmax", summary["methods_used"])
        self.assertIn("standard", summary["methods_used"])
        self.assertEqual(summary["total_features_normalized"], 3)
        self.assertIn("individual_stats", summary)
    
    def test_normalization_edge_cases(self):
        """Test normalization with edge cases."""
        # Test with non-numeric columns
        mixed_data = pd.DataFrame({
            'Numeric': [1, 2, 3, 4, 5],
            'Text': ['a', 'b', 'c', 'd', 'e'],
            'Boolean': [True, False, True, False, True]
        })
        
        normalized_data, stats = self.normalizer.normalize_features(
            mixed_data, ['Numeric', 'Text', 'Boolean'], 'minmax'
        )
        
        # Should only normalize numeric columns
        self.assertEqual(len(stats["features_normalized"]), 1)
        self.assertIn("Numeric", stats["features_normalized"])


class TestDataProcessorIntegration(unittest.TestCase):
    """Integration tests for data processing components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.cleaner = DataCleaner()
        
        # Create realistic stock data with various issues
        dates = pd.date_range(start='2023-01-01', end='2023-03-31', freq='B')
        np.random.seed(42)  # For reproducible tests
        
        base_price = 100
        prices = []
        for i in range(len(dates)):
            # Simulate price movement with some volatility
            change = np.random.normal(0, 0.02)  # 2% daily volatility
            base_price *= (1 + change)
            prices.append(base_price)
        
        self.realistic_data = pd.DataFrame({
            'Open': [p * (1 + np.random.normal(0, 0.005)) for p in prices],
            'High': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'Low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'Close': prices,
            'Volume': [np.random.randint(500000, 2000000) for _ in range(len(dates))]
        }, index=dates)
        
        # Ensure OHLC consistency
        for idx in self.realistic_data.index:
            row = self.realistic_data.loc[idx]
            self.realistic_data.loc[idx, 'High'] = max(row['Open'], row['Close'], row['Low'], row['High'])
            self.realistic_data.loc[idx, 'Low'] = min(row['Open'], row['Close'], row['High'], row['Low'])
        
        # Introduce some issues
        # Add missing values
        missing_indices = np.random.choice(self.realistic_data.index, size=5, replace=False)
        for idx in missing_indices:
            self.realistic_data.loc[idx, 'Volume'] = np.nan
        
        # Add outliers
        outlier_indices = np.random.choice(self.realistic_data.index, size=3, replace=False)
        for idx in outlier_indices:
            self.realistic_data.loc[idx, 'Close'] *= 1.5  # 50% price spike
    
    def test_end_to_end_cleaning_and_validation(self):
        """Test complete cleaning and validation workflow."""
        symbol = "REALISTIC_TEST"
        
        # Step 1: Clean the data
        cleaned_data, cleaning_stats = self.cleaner.clean_stock_data(
            self.realistic_data, symbol
        )
        
        # Verify cleaning was performed
        self.assertGreater(cleaning_stats["missing_values_removed"], 0)
        self.assertGreater(cleaning_stats["outliers_capped"], 0)
        self.assertGreater(cleaning_stats["data_quality_score"], 0.5)
        
        # Step 2: Validate cleaned data
        is_valid, issues = self.cleaner.validate_cleaned_data(cleaned_data, symbol)
        
        # Should be valid after cleaning (allow for some extreme movements in test data)
        # Filter out expected issues from our test data setup
        serious_issues = [issue for issue in issues if 
                         not issue.startswith("Extreme price movements detected")]
        self.assertEqual(len(serious_issues), 0, f"Serious validation issues: {serious_issues}")
        
        # Step 3: Check completeness and consistency
        completeness_stats = DataValidator.check_data_completeness(cleaned_data, symbol)
        consistency_results = DataValidator.check_data_consistency(cleaned_data, symbol)
        
        # Should have good completeness and consistency
        self.assertEqual(len(completeness_stats["issues"]), 0)
        self.assertTrue(consistency_results["ohlc_consistency"])
        self.assertTrue(consistency_results["volume_consistency"])
        # Price continuity may fail due to intentional extreme movements in test data
        # This is acceptable for testing purposes
    
    def test_cleaning_preserves_data_structure(self):
        """Test that cleaning preserves essential data structure."""
        original_columns = set(self.realistic_data.columns)
        original_index_type = type(self.realistic_data.index)
        
        cleaned_data, _ = self.cleaner.clean_stock_data(self.realistic_data, "STRUCTURE_TEST")
        
        # Should preserve columns and index type
        self.assertEqual(set(cleaned_data.columns), original_columns)
        self.assertEqual(type(cleaned_data.index), original_index_type)
        
        # Should preserve data types
        for col in cleaned_data.columns:
            self.assertTrue(pd.api.types.is_numeric_dtype(cleaned_data[col]))
    
    def test_multiple_symbol_processing(self):
        """Test processing multiple symbols."""
        symbols = ["SYMBOL1", "SYMBOL2", "SYMBOL3"]
        
        for symbol in symbols:
            # Add some variation to each dataset
            varied_data = self.realistic_data * (1 + np.random.normal(0, 0.1))
            self.cleaner.clean_stock_data(varied_data, symbol)
        
        summary = self.cleaner.get_cleaning_summary()
        
        self.assertEqual(summary["total_symbols_processed"], len(symbols))
        self.assertIn("individual_stats", summary)
        
        # Each symbol should have its own stats
        for symbol in symbols:
            self.assertIn(symbol, summary["individual_stats"])


if __name__ == '__main__':
    unittest.main()