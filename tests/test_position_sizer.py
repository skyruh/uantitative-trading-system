"""
Unit tests for position sizing functionality.
Tests position sizing calculations and risk management rules.
"""

import unittest
from datetime import datetime
from unittest.mock import Mock, patch

from src.risk.position_sizer import PositionSizer, PositionSizingConfig
from src.interfaces.trading_interfaces import TradingSignal


class TestPositionSizer(unittest.TestCase):
    """Test cases for PositionSizer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = PositionSizingConfig(
            min_position_pct=0.01,
            max_position_pct=0.02,
            base_position_pct=0.015,
            sentiment_adjustment_pct=0.20,
            min_trade_value=1000.0,
            max_trade_value=50000.0
        )
        self.position_sizer = PositionSizer(self.config)
        
        # Sample trading signal
        self.sample_signal = TradingSignal(
            symbol="RELIANCE",
            timestamp=datetime.now(),
            action="buy",
            confidence=0.8,
            lstm_prediction=0.6,
            dqn_q_values={"buy": 0.7, "sell": 0.2, "hold": 0.1},
            sentiment_score=0.5,
            risk_adjusted_size=0.0  # Will be calculated
        )
        
        self.portfolio_value = 1000000.0  # 10 lakh INR
        self.stock_price = 2500.0  # Sample stock price
    
    def test_basic_position_sizing(self):
        """Test basic position sizing calculation."""
        position_size = self.position_sizer.calculate_position_size(
            self.sample_signal, self.portfolio_value, self.stock_price
        )
        
        # Should be between min and max position size
        self.assertGreaterEqual(position_size, self.config.min_position_pct)
        self.assertLessEqual(position_size, self.config.max_position_pct)
        
        # Should be reasonable value
        self.assertGreater(position_size, 0)
        self.assertLess(position_size, 0.1)  # Should not exceed 10%
    
    def test_sentiment_adjustment_positive(self):
        """Test position sizing with positive sentiment."""
        # Positive sentiment should increase position size
        positive_signal = TradingSignal(
            symbol="RELIANCE",
            timestamp=datetime.now(),
            action="buy",
            confidence=0.8,
            lstm_prediction=0.6,
            dqn_q_values={"buy": 0.7, "sell": 0.2, "hold": 0.1},
            sentiment_score=1.0,  # Maximum positive sentiment
            risk_adjusted_size=0.0
        )
        
        neutral_signal = TradingSignal(
            symbol="RELIANCE",
            timestamp=datetime.now(),
            action="buy",
            confidence=0.8,
            lstm_prediction=0.6,
            dqn_q_values={"buy": 0.7, "sell": 0.2, "hold": 0.1},
            sentiment_score=0.0,  # Neutral sentiment
            risk_adjusted_size=0.0
        )
        
        positive_size = self.position_sizer.calculate_position_size(
            positive_signal, self.portfolio_value, self.stock_price
        )
        neutral_size = self.position_sizer.calculate_position_size(
            neutral_signal, self.portfolio_value, self.stock_price
        )
        
        # Positive sentiment should result in larger position size
        self.assertGreater(positive_size, neutral_size)
    
    def test_sentiment_adjustment_negative(self):
        """Test position sizing with negative sentiment."""
        # Negative sentiment should decrease position size
        negative_signal = TradingSignal(
            symbol="RELIANCE",
            timestamp=datetime.now(),
            action="buy",
            confidence=0.8,
            lstm_prediction=0.6,
            dqn_q_values={"buy": 0.7, "sell": 0.2, "hold": 0.1},
            sentiment_score=-1.0,  # Maximum negative sentiment
            risk_adjusted_size=0.0
        )
        
        neutral_signal = TradingSignal(
            symbol="RELIANCE",
            timestamp=datetime.now(),
            action="buy",
            confidence=0.8,
            lstm_prediction=0.6,
            dqn_q_values={"buy": 0.7, "sell": 0.2, "hold": 0.1},
            sentiment_score=0.0,  # Neutral sentiment
            risk_adjusted_size=0.0
        )
        
        negative_size = self.position_sizer.calculate_position_size(
            negative_signal, self.portfolio_value, self.stock_price
        )
        neutral_size = self.position_sizer.calculate_position_size(
            neutral_signal, self.portfolio_value, self.stock_price
        )
        
        # Negative sentiment should result in smaller position size
        self.assertLess(negative_size, neutral_size)
    
    def test_confidence_adjustment(self):
        """Test position sizing with different confidence levels."""
        high_confidence_signal = TradingSignal(
            symbol="RELIANCE",
            timestamp=datetime.now(),
            action="buy",
            confidence=1.0,  # Maximum confidence
            lstm_prediction=0.6,
            dqn_q_values={"buy": 0.7, "sell": 0.2, "hold": 0.1},
            sentiment_score=0.0,
            risk_adjusted_size=0.0
        )
        
        low_confidence_signal = TradingSignal(
            symbol="RELIANCE",
            timestamp=datetime.now(),
            action="buy",
            confidence=0.1,  # Low confidence
            lstm_prediction=0.6,
            dqn_q_values={"buy": 0.7, "sell": 0.2, "hold": 0.1},
            sentiment_score=0.0,
            risk_adjusted_size=0.0
        )
        
        high_conf_size = self.position_sizer.calculate_position_size(
            high_confidence_signal, self.portfolio_value, self.stock_price
        )
        low_conf_size = self.position_sizer.calculate_position_size(
            low_confidence_signal, self.portfolio_value, self.stock_price
        )
        
        # High confidence should result in larger position size
        self.assertGreater(high_conf_size, low_conf_size)
    
    def test_min_max_bounds(self):
        """Test that position sizes stay within min/max bounds."""
        # Test with extreme values that should trigger bounds
        extreme_positive_signal = TradingSignal(
            symbol="RELIANCE",
            timestamp=datetime.now(),
            action="buy",
            confidence=1.0,
            lstm_prediction=0.6,
            dqn_q_values={"buy": 0.7, "sell": 0.2, "hold": 0.1},
            sentiment_score=1.0,  # Maximum positive
            risk_adjusted_size=0.0
        )
        
        extreme_negative_signal = TradingSignal(
            symbol="RELIANCE",
            timestamp=datetime.now(),
            action="buy",
            confidence=0.0,
            lstm_prediction=0.6,
            dqn_q_values={"buy": 0.7, "sell": 0.2, "hold": 0.1},
            sentiment_score=-1.0,  # Maximum negative
            risk_adjusted_size=0.0
        )
        
        max_size = self.position_sizer.calculate_position_size(
            extreme_positive_signal, self.portfolio_value, self.stock_price
        )
        min_size = self.position_sizer.calculate_position_size(
            extreme_negative_signal, self.portfolio_value, self.stock_price
        )
        
        # Should not exceed bounds
        self.assertLessEqual(max_size, self.config.max_position_pct)
        self.assertGreaterEqual(min_size, self.config.min_position_pct)
    
    def test_trade_value_limits(self):
        """Test minimum and maximum trade value limits."""
        # Test with small portfolio that would result in trade below minimum
        small_portfolio = 50000.0  # 50k INR
        
        position_size = self.position_sizer.calculate_position_size(
            self.sample_signal, small_portfolio, self.stock_price
        )
        
        trade_value = position_size * small_portfolio
        
        # Should meet minimum trade value
        self.assertGreaterEqual(trade_value, self.config.min_trade_value)
    
    def test_get_max_shares(self):
        """Test maximum shares calculation."""
        max_shares = self.position_sizer.get_max_shares(
            self.portfolio_value, self.stock_price
        )
        
        # Should be reasonable number
        self.assertGreater(max_shares, 0)
        
        # Should not exceed maximum position value
        max_value = max_shares * self.stock_price
        max_allowed_value = self.portfolio_value * self.config.max_position_pct
        self.assertLessEqual(max_value, max_allowed_value)
    
    def test_validate_position_size(self):
        """Test position size validation."""
        # Valid position size
        valid_size = 0.015  # 1.5%
        self.assertTrue(
            self.position_sizer.validate_position_size(valid_size, self.portfolio_value)
        )
        
        # Too small position size
        too_small = 0.005  # 0.5%
        self.assertFalse(
            self.position_sizer.validate_position_size(too_small, self.portfolio_value)
        )
        
        # Too large position size
        too_large = 0.05  # 5%
        self.assertFalse(
            self.position_sizer.validate_position_size(too_large, self.portfolio_value)
        )
    
    def test_sentiment_adjustment_calculation(self):
        """Test sentiment adjustment calculation."""
        # Test various sentiment scores
        test_cases = [
            (-1.0, -0.20),  # Maximum negative
            (-0.5, -0.10),  # Moderate negative
            (0.0, 0.0),     # Neutral
            (0.5, 0.10),    # Moderate positive
            (1.0, 0.20),    # Maximum positive
        ]
        
        for sentiment_score, expected_adjustment in test_cases:
            adjustment = self.position_sizer._calculate_sentiment_adjustment(sentiment_score)
            self.assertAlmostEqual(adjustment, expected_adjustment, places=3)
    
    def test_confidence_adjustment_calculation(self):
        """Test confidence adjustment calculation."""
        # Test various confidence levels
        test_cases = [
            (0.0, 0.5),   # No confidence -> 50% of base size
            (0.5, 0.75),  # Medium confidence -> 75% of base size
            (1.0, 1.0),   # Full confidence -> 100% of base size
        ]
        
        for confidence, expected_adjustment in test_cases:
            adjustment = self.position_sizer._calculate_confidence_adjustment(confidence)
            self.assertAlmostEqual(adjustment, expected_adjustment, places=3)
    
    @patch('src.risk.position_sizer.logging.getLogger')
    def test_error_handling(self, mock_logger):
        """Test error handling in position sizing."""
        # Test with invalid signal (None values)
        invalid_signal = TradingSignal(
            symbol="INVALID",
            timestamp=datetime.now(),
            action="buy",
            confidence=None,  # Invalid
            lstm_prediction=0.6,
            dqn_q_values={"buy": 0.7, "sell": 0.2, "hold": 0.1},
            sentiment_score=None,  # Invalid
            risk_adjusted_size=0.0
        )
        
        # Should handle gracefully and return minimum position size
        position_size = self.position_sizer.calculate_position_size(
            invalid_signal, self.portfolio_value, self.stock_price
        )
        
        self.assertEqual(position_size, self.config.min_position_pct)


if __name__ == '__main__':
    unittest.main()