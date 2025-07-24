"""
Unit tests for technical indicators module.
Tests RSI, SMA, and Bollinger Bands calculations with known values.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.technical_indicators import TechnicalIndicators


class TestTechnicalIndicators(unittest.TestCase):
    """Test cases for TechnicalIndicators class."""
    
    def setUp(self):
        """Set up test data."""
        # Create sample price data for testing
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        
        # Create predictable price data for testing
        # Using a simple pattern that makes calculations verifiable
        prices = []
        base_price = 100
        for i in range(100):
            # Create some variation but keep it predictable
            price = base_price + (i % 10) - 5 + (i * 0.1)
            prices.append(price)
        
        self.test_data = pd.DataFrame({
            'date': dates,
            'open': [p + 1 for p in prices],
            'high': [p + 2 for p in prices],
            'low': [p - 1 for p in prices],
            'close': prices,
            'volume': [1000 + i * 10 for i in range(100)]
        })
        self.test_data.set_index('date', inplace=True)
        
        # Create minimal data for edge case testing
        self.minimal_data = pd.DataFrame({
            'close': [100, 101, 99, 102, 98]
        })
        
        # Create empty data for edge case testing
        self.empty_data = pd.DataFrame()
    
    def test_rsi_calculation_basic(self):
        """Test basic RSI calculation."""
        rsi = TechnicalIndicators.calculate_rsi(self.test_data, period=14)
        
        # Check that RSI is calculated
        self.assertIsInstance(rsi, pd.Series)
        self.assertEqual(len(rsi), len(self.test_data))
        
        # Check that first 14 values are NaN (as expected)
        self.assertTrue(pd.isna(rsi.iloc[:14]).all())
        
        # Check that RSI values are within valid range (0-100)
        valid_rsi = rsi.dropna()
        self.assertTrue((valid_rsi >= 0).all())
        self.assertTrue((valid_rsi <= 100).all())
    
    def test_rsi_known_values(self):
        """Test RSI calculation with known values."""
        # Create specific data pattern for RSI testing
        # Alternating gains and losses
        test_prices = [100, 105, 102, 108, 104, 110, 106, 112, 108, 114, 
                      110, 116, 112, 118, 114, 120, 116, 122, 118, 124]
        
        test_df = pd.DataFrame({'close': test_prices})
        rsi = TechnicalIndicators.calculate_rsi(test_df, period=14)
        
        # The last RSI value should be reasonable (not extreme)
        # Given the alternating pattern, RSI should be around 50
        last_rsi = rsi.iloc[-1]
        self.assertFalse(pd.isna(last_rsi))
        self.assertGreater(last_rsi, 30)
        self.assertLess(last_rsi, 70)
    
    def test_rsi_insufficient_data(self):
        """Test RSI calculation with insufficient data."""
        rsi = TechnicalIndicators.calculate_rsi(self.minimal_data, period=14)
        
        # Should return series with all NaN values
        self.assertTrue(pd.isna(rsi).all())
    
    def test_rsi_empty_data(self):
        """Test RSI calculation with empty data."""
        with self.assertRaises(ValueError):
            TechnicalIndicators.calculate_rsi(self.empty_data)
    
    def test_rsi_invalid_column(self):
        """Test RSI calculation with invalid column name."""
        with self.assertRaises(ValueError):
            TechnicalIndicators.calculate_rsi(self.test_data, price_col='invalid_column')
    
    def test_sma_calculation_basic(self):
        """Test basic SMA calculation."""
        sma = TechnicalIndicators.calculate_sma(self.test_data, period=20)
        
        # Check that SMA is calculated
        self.assertIsInstance(sma, pd.Series)
        self.assertEqual(len(sma), len(self.test_data))
        
        # Check that first 19 values are NaN
        self.assertTrue(pd.isna(sma.iloc[:19]).all())
        
        # Check that SMA values are reasonable
        valid_sma = sma.dropna()
        self.assertTrue(len(valid_sma) > 0)
        self.assertTrue((valid_sma > 0).all())
    
    def test_sma_known_values(self):
        """Test SMA calculation with known values."""
        # Simple test case: constant values
        constant_prices = [100] * 25
        test_df = pd.DataFrame({'close': constant_prices})
        sma = TechnicalIndicators.calculate_sma(test_df, period=10)
        
        # SMA of constant values should equal the constant
        valid_sma = sma.dropna()
        self.assertTrue((valid_sma == 100).all())
        
        # Test with simple increasing sequence
        increasing_prices = list(range(1, 26))  # 1 to 25
        test_df = pd.DataFrame({'close': increasing_prices})
        sma = TechnicalIndicators.calculate_sma(test_df, period=5)
        
        # SMA of first 5 values (1,2,3,4,5) should be 3
        first_valid_sma = sma.dropna().iloc[0]
        self.assertAlmostEqual(first_valid_sma, 3.0, places=1)
    
    def test_sma_insufficient_data(self):
        """Test SMA calculation with insufficient data."""
        sma = TechnicalIndicators.calculate_sma(self.minimal_data, period=10)
        
        # Should return series with all NaN values
        self.assertTrue(pd.isna(sma).all())
    
    def test_bollinger_bands_calculation_basic(self):
        """Test basic Bollinger Bands calculation."""
        bb_upper, bb_middle, bb_lower = TechnicalIndicators.calculate_bollinger_bands(
            self.test_data, period=20
        )
        
        # Check that all bands are calculated
        self.assertIsInstance(bb_upper, pd.Series)
        self.assertIsInstance(bb_middle, pd.Series)
        self.assertIsInstance(bb_lower, pd.Series)
        
        # Check lengths
        self.assertEqual(len(bb_upper), len(self.test_data))
        self.assertEqual(len(bb_middle), len(self.test_data))
        self.assertEqual(len(bb_lower), len(self.test_data))
        
        # Check that first 19 values are NaN
        self.assertTrue(pd.isna(bb_upper.iloc[:19]).all())
        self.assertTrue(pd.isna(bb_middle.iloc[:19]).all())
        self.assertTrue(pd.isna(bb_lower.iloc[:19]).all())
        
        # Check band relationships (upper > middle > lower)
        valid_indices = ~pd.isna(bb_middle)
        if valid_indices.any():
            self.assertTrue((bb_upper[valid_indices] >= bb_middle[valid_indices]).all())
            self.assertTrue((bb_middle[valid_indices] >= bb_lower[valid_indices]).all())
    
    def test_bollinger_bands_known_values(self):
        """Test Bollinger Bands calculation with known values."""
        # Test with constant values - bands should converge to the constant
        constant_prices = [100] * 25
        test_df = pd.DataFrame({'close': constant_prices})
        bb_upper, bb_middle, bb_lower = TechnicalIndicators.calculate_bollinger_bands(
            test_df, period=10, std_dev=2.0
        )
        
        # With constant prices, standard deviation is 0, so all bands should equal the constant
        valid_middle = bb_middle.dropna()
        self.assertTrue((valid_middle == 100).all())
        
        # Upper and lower bands should also equal middle when std dev is 0
        valid_upper = bb_upper.dropna()
        valid_lower = bb_lower.dropna()
        self.assertTrue(np.allclose(valid_upper, valid_middle))
        self.assertTrue(np.allclose(valid_lower, valid_middle))
    
    def test_bollinger_bands_insufficient_data(self):
        """Test Bollinger Bands calculation with insufficient data."""
        bb_upper, bb_middle, bb_lower = TechnicalIndicators.calculate_bollinger_bands(
            self.minimal_data, period=10
        )
        
        # Should return series with all NaN values
        self.assertTrue(pd.isna(bb_upper).all())
        self.assertTrue(pd.isna(bb_middle).all())
        self.assertTrue(pd.isna(bb_lower).all())
    
    def test_calculate_all_indicators(self):
        """Test calculation of all indicators together."""
        result = TechnicalIndicators.calculate_all_indicators(self.test_data)
        
        # Check that original columns are preserved
        for col in self.test_data.columns:
            self.assertIn(col, result.columns)
        
        # Check that new indicator columns are added
        expected_indicators = ['rsi_14', 'sma_50', 'bb_upper', 'bb_middle', 'bb_lower']
        for indicator in expected_indicators:
            self.assertIn(indicator, result.columns)
        
        # Check that indicators have reasonable values where not NaN
        if not pd.isna(result['rsi_14']).all():
            valid_rsi = result['rsi_14'].dropna()
            self.assertTrue((valid_rsi >= 0).all())
            self.assertTrue((valid_rsi <= 100).all())
        
        if not pd.isna(result['sma_50']).all():
            valid_sma = result['sma_50'].dropna()
            self.assertTrue((valid_sma > 0).all())
    
    def test_calculate_all_indicators_empty_data(self):
        """Test calculation of all indicators with empty data."""
        result = TechnicalIndicators.calculate_all_indicators(self.empty_data)
        
        # Should return empty DataFrame
        self.assertTrue(result.empty)
    
    def test_edge_cases(self):
        """Test various edge cases."""
        # Test with single row
        single_row = pd.DataFrame({'close': [100]})
        rsi = TechnicalIndicators.calculate_rsi(single_row, period=14)
        self.assertTrue(pd.isna(rsi).all())
        
        # Test with NaN values in data
        data_with_nan = self.test_data.copy()
        data_with_nan.loc[data_with_nan.index[10], 'close'] = np.nan
        
        result = TechnicalIndicators.calculate_all_indicators(data_with_nan)
        # Should handle NaN values gracefully
        self.assertIsInstance(result, pd.DataFrame)


if __name__ == '__main__':
    unittest.main()