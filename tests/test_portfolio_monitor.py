"""
Unit tests for portfolio monitoring and diversification controls.
Tests portfolio diversification limits and concentration controls.
"""

import unittest
from datetime import datetime
from unittest.mock import Mock, patch

from src.risk.portfolio_monitor import PortfolioMonitor, PortfolioConfig, PortfolioMetrics
from src.interfaces.trading_interfaces import TradingSignal, Position


class TestPortfolioMonitor(unittest.TestCase):
    """Test cases for PortfolioMonitor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = PortfolioConfig(
            max_positions=30,
            min_positions=20,
            max_position_pct=0.05,  # 5%
            max_sector_pct=0.20,    # 20%
            max_market_cap_pct=0.60, # 60%
            sentiment_position_adjustment=0.20,
            rebalance_threshold=0.02
        )
        self.portfolio_monitor = PortfolioMonitor(self.config)
        self.portfolio_value = 1000000.0  # 10 lakh INR
        
        # Sample trading signal
        self.sample_signal = TradingSignal(
            symbol="RELIANCE",
            timestamp=datetime.now(),
            action="buy",
            confidence=0.8,
            lstm_prediction=0.6,
            dqn_q_values={"buy": 0.7, "sell": 0.2, "hold": 0.1},
            sentiment_score=0.5,
            risk_adjusted_size=0.02  # 2%
        )
    
    def test_check_position_limits_basic(self):
        """Test basic position limits checking."""
        current_positions = []  # Empty portfolio
        
        is_allowed, reason = self.portfolio_monitor.check_position_limits(
            self.sample_signal, current_positions, self.portfolio_value
        )
        
        self.assertTrue(is_allowed)
        self.assertEqual(reason, "Position within diversification limits")
    
    def test_check_position_limits_max_positions(self):
        """Test maximum positions limit."""
        # Create positions at the limit
        current_positions = []
        for i in range(self.config.max_positions):
            position = Position(
                symbol=f"STOCK_{i}",
                entry_date=datetime.now(),
                entry_price=1000.0,
                quantity=10,
                stop_loss_price=950.0,
                current_value=10000.0,
                unrealized_pnl=0.0,
                status="open"
            )
            current_positions.append(position)
        
        is_allowed, reason = self.portfolio_monitor.check_position_limits(
            self.sample_signal, current_positions, self.portfolio_value
        )
        
        self.assertFalse(is_allowed)
        self.assertIn("Maximum positions limit reached", reason)
    
    def test_check_position_limits_existing_position(self):
        """Test rejection of duplicate positions."""
        # Create existing position in same stock
        existing_position = Position(
            symbol="RELIANCE",  # Same as sample signal
            entry_date=datetime.now(),
            entry_price=2500.0,
            quantity=10,
            stop_loss_price=2375.0,
            current_value=25000.0,
            unrealized_pnl=0.0,
            status="open"
        )
        
        current_positions = [existing_position]
        
        is_allowed, reason = self.portfolio_monitor.check_position_limits(
            self.sample_signal, current_positions, self.portfolio_value
        )
        
        self.assertFalse(is_allowed)
        self.assertIn("Already have position in RELIANCE", reason)
    
    def test_sentiment_adjusted_size_positive(self):
        """Test sentiment-based position size adjustment (positive)."""
        base_size = 0.02  # 2%
        positive_sentiment = 1.0  # Maximum positive
        
        adjusted_size = self.portfolio_monitor._calculate_sentiment_adjusted_size(
            base_size, positive_sentiment
        )
        
        # Should be increased by 20% (2% * 1.2 = 2.4%)
        expected_size = base_size * (1 + self.config.sentiment_position_adjustment)
        self.assertAlmostEqual(adjusted_size, expected_size, places=4)
    
    def test_sentiment_adjusted_size_negative(self):
        """Test sentiment-based position size adjustment (negative)."""
        base_size = 0.02  # 2%
        negative_sentiment = -1.0  # Maximum negative
        
        adjusted_size = self.portfolio_monitor._calculate_sentiment_adjusted_size(
            base_size, negative_sentiment
        )
        
        # Should be decreased by 20% (2% * 0.8 = 1.6%)
        expected_size = base_size * (1 - self.config.sentiment_position_adjustment)
        self.assertAlmostEqual(adjusted_size, expected_size, places=4)
    
    def test_sentiment_adjusted_size_neutral(self):
        """Test sentiment-based position size adjustment (neutral)."""
        base_size = 0.02  # 2%
        neutral_sentiment = 0.0  # Neutral
        
        adjusted_size = self.portfolio_monitor._calculate_sentiment_adjusted_size(
            base_size, neutral_sentiment
        )
        
        # Should remain unchanged
        self.assertAlmostEqual(adjusted_size, base_size, places=4)
    
    def test_sector_limits_within_bounds(self):
        """Test sector concentration within limits."""
        # Create positions in different sectors
        current_positions = [
            Position(
                symbol="TCS",  # IT sector
                entry_date=datetime.now(),
                entry_price=3000.0,
                quantity=5,
                stop_loss_price=2850.0,
                current_value=15000.0,  # 1.5% of portfolio
                unrealized_pnl=0.0,
                status="open"
            ),
            Position(
                symbol="INFY",  # IT sector
                entry_date=datetime.now(),
                entry_price=1500.0,
                quantity=10,
                stop_loss_price=1425.0,
                current_value=15000.0,  # 1.5% of portfolio
                unrealized_pnl=0.0,
                status="open"
            )
        ]
        
        # Add another IT stock (total IT would be 5%)
        it_signal = TradingSignal(
            symbol="WIPRO",
            timestamp=datetime.now(),
            action="buy",
            confidence=0.8,
            lstm_prediction=0.6,
            dqn_q_values={"buy": 0.7, "sell": 0.2, "hold": 0.1},
            sentiment_score=0.0,
            risk_adjusted_size=0.02  # 2%
        )
        
        is_allowed, reason = self.portfolio_monitor.check_position_limits(
            it_signal, current_positions, self.portfolio_value
        )
        
        self.assertTrue(is_allowed)  # 5% total IT is within 20% limit
    
    def test_sector_limits_exceeded(self):
        """Test sector concentration limits exceeded."""
        # Create positions that would exceed sector limit
        current_positions = []
        it_stocks = ["TCS", "INFY", "WIPRO"]
        
        for stock in it_stocks:
            position = Position(
                symbol=stock,
                entry_date=datetime.now(),
                entry_price=2000.0,
                quantity=30,  # Large position
                stop_loss_price=1900.0,
                current_value=60000.0,  # 6% each = 18% total IT
                unrealized_pnl=0.0,
                status="open"
            )
            current_positions.append(position)
        
        # Try to add another IT stock that would exceed 20% limit
        new_it_signal = TradingSignal(
            symbol="TECHM",  # Another IT stock
            timestamp=datetime.now(),
            action="buy",
            confidence=0.8,
            lstm_prediction=0.6,
            dqn_q_values={"buy": 0.7, "sell": 0.2, "hold": 0.1},
            sentiment_score=0.0,
            risk_adjusted_size=0.05  # 5% - would make total IT 23%
        )
        
        is_allowed, reason = self.portfolio_monitor.check_position_limits(
            new_it_signal, current_positions, self.portfolio_value
        )
        
        self.assertFalse(is_allowed)
        self.assertIn("Sector IT exposure", reason)
        self.assertIn("would exceed limit", reason)
    
    def test_market_cap_limits_within_bounds(self):
        """Test market cap concentration within limits."""
        # Create positions in large cap stocks
        current_positions = [
            Position(
                symbol="RELIANCE",  # Large cap
                entry_date=datetime.now(),
                entry_price=2500.0,
                quantity=20,
                stop_loss_price=2375.0,
                current_value=50000.0,  # 5% of portfolio
                unrealized_pnl=0.0,
                status="open"
            )
        ]
        
        # Add another large cap stock
        large_cap_signal = TradingSignal(
            symbol="TCS",
            timestamp=datetime.now(),
            action="buy",
            confidence=0.8,
            lstm_prediction=0.6,
            dqn_q_values={"buy": 0.7, "sell": 0.2, "hold": 0.1},
            sentiment_score=0.0,
            risk_adjusted_size=0.05  # 5%
        )
        
        is_allowed, reason = self.portfolio_monitor.check_position_limits(
            large_cap_signal, current_positions, self.portfolio_value
        )
        
        self.assertTrue(is_allowed)  # 10% total large cap is within 60% limit
    
    def test_get_portfolio_metrics_empty(self):
        """Test portfolio metrics calculation with empty portfolio."""
        metrics = self.portfolio_monitor.get_portfolio_metrics([], self.portfolio_value)
        
        self.assertEqual(metrics.total_positions, 0)
        self.assertEqual(metrics.total_exposure, 0.0)
        self.assertEqual(metrics.largest_position_pct, 0.0)
        self.assertEqual(metrics.concentration_risk, 0.0)
        self.assertEqual(metrics.diversification_score, 0.0)
        self.assertEqual(metrics.sector_distribution, {})
        self.assertEqual(metrics.market_cap_distribution, {})
    
    def test_get_portfolio_metrics_with_positions(self):
        """Test portfolio metrics calculation with positions."""
        positions = [
            Position(
                symbol="RELIANCE",
                entry_date=datetime.now(),
                entry_price=2500.0,
                quantity=20,
                stop_loss_price=2375.0,
                current_value=50000.0,  # 5%
                unrealized_pnl=0.0,
                status="open"
            ),
            Position(
                symbol="TCS",
                entry_date=datetime.now(),
                entry_price=3000.0,
                quantity=10,
                stop_loss_price=2850.0,
                current_value=30000.0,  # 3%
                unrealized_pnl=0.0,
                status="open"
            ),
            Position(
                symbol="HDFCBANK",
                entry_date=datetime.now(),
                entry_price=1500.0,
                quantity=20,
                stop_loss_price=1425.0,
                current_value=30000.0,  # 3%
                unrealized_pnl=0.0,
                status="open"
            )
        ]
        
        metrics = self.portfolio_monitor.get_portfolio_metrics(positions, self.portfolio_value)
        
        self.assertEqual(metrics.total_positions, 3)
        self.assertAlmostEqual(metrics.total_exposure, 0.11, places=2)  # 11%
        self.assertAlmostEqual(metrics.largest_position_pct, 0.05, places=2)  # 5%
        self.assertGreater(metrics.concentration_risk, 0)
        self.assertGreater(metrics.diversification_score, 0)
        
        # Check sector distribution
        self.assertIn("Energy", metrics.sector_distribution)
        self.assertIn("IT", metrics.sector_distribution)
        self.assertIn("Banking", metrics.sector_distribution)
    
    def test_suggest_rebalancing_overweight_position(self):
        """Test rebalancing suggestions for overweight positions."""
        # Create an overweight position
        positions = [
            Position(
                symbol="RELIANCE",
                entry_date=datetime.now(),
                entry_price=2500.0,
                quantity=40,  # Large position
                stop_loss_price=2375.0,
                current_value=100000.0,  # 10% - exceeds 5% + 2% threshold
                unrealized_pnl=0.0,
                status="open"
            )
        ]
        
        suggestions = self.portfolio_monitor.suggest_rebalancing(positions, self.portfolio_value)
        
        self.assertGreater(len(suggestions), 0)
        self.assertTrue(any("Reduce largest position" in s for s in suggestions))
    
    def test_suggest_rebalancing_under_diversified(self):
        """Test rebalancing suggestions for under-diversified portfolio."""
        # Create portfolio with too few positions
        positions = [
            Position(
                symbol="RELIANCE",
                entry_date=datetime.now(),
                entry_price=2500.0,
                quantity=20,
                stop_loss_price=2375.0,
                current_value=50000.0,
                unrealized_pnl=0.0,
                status="open"
            )
        ]
        
        suggestions = self.portfolio_monitor.suggest_rebalancing(positions, self.portfolio_value)
        
        self.assertGreater(len(suggestions), 0)
        self.assertTrue(any("Add more positions" in s for s in suggestions))
    
    def test_get_diversification_score(self):
        """Test diversification score calculation."""
        # Well-diversified portfolio
        positions = []
        for i in range(25):  # Good number of positions
            position = Position(
                symbol=f"STOCK_{i}",
                entry_date=datetime.now(),
                entry_price=1000.0,
                quantity=4,  # Equal weights
                stop_loss_price=950.0,
                current_value=4000.0,  # 0.4% each
                unrealized_pnl=0.0,
                status="open"
            )
            positions.append(position)
        
        score = self.portfolio_monitor.get_diversification_score(positions, self.portfolio_value)
        
        self.assertGreater(score, 0)
        self.assertLessEqual(score, 1.0)
    
    def test_update_stock_metadata(self):
        """Test updating stock metadata."""
        new_metadata = {
            "NEWSTOCK": {"sector": "Technology", "market_cap": "Mid"},
            "ANOTHER": {"sector": "Healthcare", "market_cap": "Small"}
        }
        
        original_count = len(self.portfolio_monitor.stock_metadata)
        self.portfolio_monitor.update_stock_metadata(new_metadata)
        
        # Should have added new stocks
        self.assertEqual(len(self.portfolio_monitor.stock_metadata), original_count + 2)
        self.assertIn("NEWSTOCK", self.portfolio_monitor.stock_metadata)
        self.assertIn("ANOTHER", self.portfolio_monitor.stock_metadata)
    
    def test_unknown_stock_handling(self):
        """Test handling of stocks not in metadata."""
        unknown_signal = TradingSignal(
            symbol="UNKNOWN_STOCK",
            timestamp=datetime.now(),
            action="buy",
            confidence=0.8,
            lstm_prediction=0.6,
            dqn_q_values={"buy": 0.7, "sell": 0.2, "hold": 0.1},
            sentiment_score=0.0,
            risk_adjusted_size=0.02
        )
        
        current_positions = []
        
        is_allowed, reason = self.portfolio_monitor.check_position_limits(
            unknown_signal, current_positions, self.portfolio_value
        )
        
        # Should allow unknown stocks but log warning
        self.assertTrue(is_allowed)
    
    def test_position_size_bounds(self):
        """Test position size adjustment bounds."""
        # Test extreme sentiment values
        extreme_positive = self.portfolio_monitor._calculate_sentiment_adjusted_size(0.01, 10.0)  # Invalid sentiment
        extreme_negative = self.portfolio_monitor._calculate_sentiment_adjusted_size(0.01, -10.0)  # Invalid sentiment
        
        # Should be clamped to reasonable bounds
        self.assertGreaterEqual(extreme_positive, 0.005)  # Minimum 0.5%
        self.assertLessEqual(extreme_positive, 0.1)       # Maximum 10%
        self.assertGreaterEqual(extreme_negative, 0.005)  # Minimum 0.5%
        self.assertLessEqual(extreme_negative, 0.1)       # Maximum 10%
    
    def test_concentration_risk_calculation(self):
        """Test Herfindahl-Hirschman Index calculation."""
        # Create positions with known concentration
        positions = [
            Position(
                symbol="STOCK1",
                entry_date=datetime.now(),
                entry_price=1000.0,
                quantity=50,  # 50% of portfolio
                stop_loss_price=950.0,
                current_value=500000.0,
                unrealized_pnl=0.0,
                status="open"
            ),
            Position(
                symbol="STOCK2",
                entry_date=datetime.now(),
                entry_price=1000.0,
                quantity=30,  # 30% of portfolio
                stop_loss_price=950.0,
                current_value=300000.0,
                unrealized_pnl=0.0,
                status="open"
            ),
            Position(
                symbol="STOCK3",
                entry_date=datetime.now(),
                entry_price=1000.0,
                quantity=20,  # 20% of portfolio
                stop_loss_price=950.0,
                current_value=200000.0,
                unrealized_pnl=0.0,
                status="open"
            )
        ]
        
        metrics = self.portfolio_monitor.get_portfolio_metrics(positions, self.portfolio_value)
        
        # HHI = 0.5^2 + 0.3^2 + 0.2^2 = 0.25 + 0.09 + 0.04 = 0.38
        expected_hhi = 0.38
        self.assertAlmostEqual(metrics.concentration_risk, expected_hhi, places=2)
    
    @patch('src.risk.portfolio_monitor.logging.getLogger')
    def test_error_handling(self, mock_logger):
        """Test error handling in portfolio monitoring."""
        # Test with invalid position data
        invalid_position = Position(
            symbol=None,  # Invalid
            entry_date=None,  # Invalid
            entry_price=None,  # Invalid
            quantity=10,
            stop_loss_price=0.0,
            current_value=None,  # Invalid
            unrealized_pnl=0.0,
            status="open"
        )
        
        positions = [invalid_position]
        
        # Should handle gracefully
        metrics = self.portfolio_monitor.get_portfolio_metrics(positions, self.portfolio_value)
        
        # Should return default metrics
        self.assertEqual(metrics.total_positions, 0)


if __name__ == '__main__':
    unittest.main()