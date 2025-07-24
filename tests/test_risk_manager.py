"""
Unit tests for the main RiskManager class.
Tests integration of position sizing and stop-loss management.
"""

import unittest
from datetime import datetime
from unittest.mock import Mock, patch

from src.risk.risk_manager import RiskManager
from src.risk.position_sizer import PositionSizingConfig
from src.risk.stop_loss_manager import StopLossConfig
from src.risk.portfolio_monitor import PortfolioConfig
from src.interfaces.trading_interfaces import TradingSignal, Position


class TestRiskManager(unittest.TestCase):
    """Test cases for RiskManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.position_config = PositionSizingConfig(
            min_position_pct=0.01,
            max_position_pct=0.02,
            base_position_pct=0.015
        )
        
        self.stop_loss_config = StopLossConfig(
            stop_loss_pct=0.05
        )
        
        self.portfolio_config = PortfolioConfig(
            max_positions=30,
            max_position_pct=0.05
        )
        
        self.risk_manager = RiskManager(self.position_config, self.stop_loss_config, self.portfolio_config)
        
        # Sample trading signal
        self.sample_signal = TradingSignal(
            symbol="RELIANCE",
            timestamp=datetime.now(),
            action="buy",
            confidence=0.8,
            lstm_prediction=0.6,
            dqn_q_values={"buy": 0.7, "sell": 0.2, "hold": 0.1},
            sentiment_score=0.5,
            risk_adjusted_size=0.0
        )
        
        # Sample position
        self.sample_position = Position(
            symbol="RELIANCE",
            entry_date=datetime.now(),
            entry_price=2500.0,
            quantity=10,
            stop_loss_price=0.0,
            current_value=25000.0,
            unrealized_pnl=0.0,
            status="open"
        )
        
        self.portfolio_value = 1000000.0  # 10 lakh INR
    
    def test_calculate_position_size_integration(self):
        """Test position size calculation through risk manager."""
        position_size = self.risk_manager.calculate_position_size(
            self.sample_signal, self.portfolio_value
        )
        
        # Should return reasonable position size
        self.assertGreater(position_size, 0)
        self.assertLess(position_size, 0.1)  # Should not exceed 10%
        
        # Should be within configured bounds
        self.assertGreaterEqual(position_size, self.position_config.min_position_pct)
        self.assertLessEqual(position_size, self.position_config.max_position_pct)
    
    def test_set_stop_loss_integration(self):
        """Test stop-loss setting through risk manager."""
        stop_price = self.risk_manager.set_stop_loss(self.sample_position)
        
        # Should be 5% below entry price
        expected_stop = self.sample_position.entry_price * (1 - self.stop_loss_config.stop_loss_pct)
        self.assertAlmostEqual(stop_price, expected_stop, places=2)
    
    def test_should_close_position_integration(self):
        """Test position closing decision through risk manager."""
        # Set stop-loss first
        self.risk_manager.set_stop_loss(self.sample_position)
        
        # Test with price below stop-loss
        low_price = 2300.0  # Below 5% stop-loss
        should_close = self.risk_manager.should_close_position(self.sample_position, low_price)
        self.assertTrue(should_close)
        
        # Test with price above stop-loss
        high_price = 2600.0  # Above stop-loss
        should_close = self.risk_manager.should_close_position(self.sample_position, high_price)
        self.assertFalse(should_close)
    
    def test_execute_stop_loss_integration(self):
        """Test stop-loss execution through risk manager."""
        # Set stop-loss first
        self.risk_manager.set_stop_loss(self.sample_position)
        
        # Execute stop-loss
        current_price = 2300.0  # Below stop-loss
        success, message = self.risk_manager.execute_stop_loss(self.sample_position, current_price)
        
        self.assertTrue(success)
        self.assertIn("Stop-loss executed", message)
    
    def test_validate_trade_buy_signal(self):
        """Test trade validation for buy signals."""
        current_positions = []  # Empty portfolio
        
        is_valid, reason = self.risk_manager.validate_trade(
            self.sample_signal, self.portfolio_value, current_positions
        )
        
        self.assertTrue(is_valid)
        self.assertEqual(reason, "Trade validation passed")
    
    def test_validate_trade_hold_signal(self):
        """Test trade validation for hold signals."""
        hold_signal = TradingSignal(
            symbol="RELIANCE",
            timestamp=datetime.now(),
            action="hold",
            confidence=0.8,
            lstm_prediction=0.6,
            dqn_q_values={"buy": 0.3, "sell": 0.2, "hold": 0.5},
            sentiment_score=0.0,
            risk_adjusted_size=0.0
        )
        
        current_positions = []
        
        is_valid, reason = self.risk_manager.validate_trade(
            hold_signal, self.portfolio_value, current_positions
        )
        
        self.assertTrue(is_valid)
        self.assertEqual(reason, "Hold signal - no validation needed")
    
    def test_validate_trade_invalid_action(self):
        """Test trade validation with invalid action."""
        invalid_signal = TradingSignal(
            symbol="RELIANCE",
            timestamp=datetime.now(),
            action="invalid_action",
            confidence=0.8,
            lstm_prediction=0.6,
            dqn_q_values={"buy": 0.7, "sell": 0.2, "hold": 0.1},
            sentiment_score=0.0,
            risk_adjusted_size=0.0
        )
        
        current_positions = []
        
        is_valid, reason = self.risk_manager.validate_trade(
            invalid_signal, self.portfolio_value, current_positions
        )
        
        self.assertFalse(is_valid)
        self.assertIn("Invalid signal action", reason)
    
    def test_validate_trade_portfolio_limits(self):
        """Test trade validation with portfolio limits."""
        # Create many positions to exceed limits
        current_positions = []
        for i in range(35):  # Exceed max of 30
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
        
        is_valid, reason = self.risk_manager.validate_trade(
            self.sample_signal, self.portfolio_value, current_positions
        )
        
        self.assertFalse(is_valid)
        self.assertIn("Portfolio diversification limits exceeded", reason)
    
    def test_get_risk_metrics_empty_portfolio(self):
        """Test risk metrics calculation with empty portfolio."""
        metrics = self.risk_manager.get_risk_metrics([], self.portfolio_value)
        
        expected_metrics = {
            'total_exposure': 0.0,
            'number_of_positions': 0,
            'average_position_size': 0.0,
            'largest_position_pct': 0.0,
            'stop_loss_coverage': 0.0
        }
        
        self.assertEqual(metrics, expected_metrics)
    
    def test_get_risk_metrics_with_positions(self):
        """Test risk metrics calculation with positions."""
        # Create test positions
        positions = []
        for i, symbol in enumerate(["RELIANCE", "TCS", "INFY"], 1):
            position = Position(
                symbol=symbol,
                entry_date=datetime.now(),
                entry_price=2000.0,
                quantity=10,
                stop_loss_price=1900.0,
                current_value=20000.0,
                unrealized_pnl=0.0,
                status="open"
            )
            positions.append(position)
            # Set stop-loss for coverage calculation
            self.risk_manager.set_stop_loss(position)
        
        metrics = self.risk_manager.get_risk_metrics(positions, self.portfolio_value)
        
        # Verify metrics
        self.assertEqual(metrics['number_of_positions'], 3)
        self.assertGreater(metrics['total_exposure'], 0)
        self.assertGreater(metrics['average_position_size'], 0)
        self.assertGreater(metrics['largest_position_pct'], 0)
        self.assertEqual(metrics['stop_loss_coverage'], 1.0)  # 100% coverage
    
    def test_get_position_sizer_access(self):
        """Test access to position sizer component."""
        position_sizer = self.risk_manager.get_position_sizer()
        self.assertIsNotNone(position_sizer)
        
        # Test that it's the same instance
        self.assertEqual(position_sizer, self.risk_manager.position_sizer)
    
    def test_get_stop_loss_manager_access(self):
        """Test access to stop-loss manager component."""
        stop_loss_manager = self.risk_manager.get_stop_loss_manager()
        self.assertIsNotNone(stop_loss_manager)
        
        # Test that it's the same instance
        self.assertEqual(stop_loss_manager, self.risk_manager.stop_loss_manager)
    
    def test_check_portfolio_limits_basic(self):
        """Test basic portfolio limits checking."""
        # Test with reasonable number of positions
        current_positions = []
        for i in range(10):  # Well below limit
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
        
        # New position should be allowed
        within_limits = self.risk_manager.check_portfolio_limits(
            self.sample_position, current_positions
        )
        
        self.assertTrue(within_limits)
    
    @patch('src.risk.risk_manager.logging.getLogger')
    def test_error_handling_position_size(self, mock_logger):
        """Test error handling in position size calculation."""
        # Test with invalid signal
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
        
        # Should handle gracefully
        position_size = self.risk_manager.calculate_position_size(
            invalid_signal, self.portfolio_value
        )
        
        # Should return default minimum size
        self.assertEqual(position_size, 0.01)
    
    @patch('src.risk.risk_manager.logging.getLogger')
    def test_error_handling_trade_validation(self, mock_logger):
        """Test error handling in trade validation."""
        # Test with None signal
        is_valid, reason = self.risk_manager.validate_trade(
            None, self.portfolio_value, []
        )
        
        self.assertFalse(is_valid)
        self.assertIn("Validation error", reason)
    
    def test_risk_manager_initialization_default_configs(self):
        """Test risk manager initialization with default configurations."""
        default_risk_manager = RiskManager()
        
        # Should initialize successfully
        self.assertIsNotNone(default_risk_manager.position_sizer)
        self.assertIsNotNone(default_risk_manager.stop_loss_manager)
    
    def test_get_portfolio_monitor_access(self):
        """Test access to portfolio monitor component."""
        portfolio_monitor = self.risk_manager.get_portfolio_monitor()
        self.assertIsNotNone(portfolio_monitor)
        
        # Test that it's the same instance
        self.assertEqual(portfolio_monitor, self.risk_manager.portfolio_monitor)
    
    def test_get_portfolio_metrics_integration(self):
        """Test portfolio metrics through risk manager."""
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
            ),
            Position(
                symbol="TCS",
                entry_date=datetime.now(),
                entry_price=3000.0,
                quantity=10,
                stop_loss_price=2850.0,
                current_value=30000.0,
                unrealized_pnl=0.0,
                status="open"
            )
        ]
        
        metrics = self.risk_manager.get_portfolio_metrics(positions, self.portfolio_value)
        
        # Should return portfolio metrics
        self.assertEqual(metrics.total_positions, 2)
        self.assertGreater(metrics.total_exposure, 0)
        self.assertGreater(metrics.diversification_score, 0)
    
    def test_suggest_rebalancing_integration(self):
        """Test rebalancing suggestions through risk manager."""
        # Create under-diversified portfolio
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
        
        suggestions = self.risk_manager.suggest_rebalancing(positions, self.portfolio_value)
        
        # Should suggest adding more positions
        self.assertGreater(len(suggestions), 0)
        self.assertTrue(any("Add more positions" in s for s in suggestions))
    
    def test_risk_manager_initialization_with_portfolio_config(self):
        """Test risk manager initialization with portfolio configuration."""
        # Should initialize successfully with portfolio config
        self.assertIsNotNone(self.risk_manager.portfolio_monitor)
        
        # Check that configuration was applied
        self.assertEqual(self.risk_manager.portfolio_monitor.config.max_positions, 30)
        self.assertEqual(self.risk_manager.portfolio_monitor.config.max_position_pct, 0.05)
    
    def test_comprehensive_workflow(self):
        """Test comprehensive risk management workflow."""
        # 1. Calculate position size
        position_size = self.risk_manager.calculate_position_size(
            self.sample_signal, self.portfolio_value
        )
        self.assertGreater(position_size, 0)
        
        # 2. Validate trade
        is_valid, reason = self.risk_manager.validate_trade(
            self.sample_signal, self.portfolio_value, []
        )
        self.assertTrue(is_valid)
        
        # 3. Set stop-loss for position
        stop_price = self.risk_manager.set_stop_loss(self.sample_position)
        self.assertGreater(stop_price, 0)
        
        # 4. Check if position should be closed
        should_close = self.risk_manager.should_close_position(
            self.sample_position, 2600.0  # Above stop-loss
        )
        self.assertFalse(should_close)
        
        # 5. Get risk metrics
        metrics = self.risk_manager.get_risk_metrics([self.sample_position], self.portfolio_value)
        self.assertGreater(metrics['number_of_positions'], 0)
        
        # 6. Get portfolio metrics
        portfolio_metrics = self.risk_manager.get_portfolio_metrics([self.sample_position], self.portfolio_value)
        self.assertGreater(portfolio_metrics.total_positions, 0)
        
        # 7. Get rebalancing suggestions
        suggestions = self.risk_manager.suggest_rebalancing([self.sample_position], self.portfolio_value)
        self.assertIsInstance(suggestions, list)


if __name__ == '__main__':
    unittest.main()