"""
Unit tests for stop-loss management functionality.
Tests stop-loss setting, execution, and automatic triggering.
"""

import unittest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.risk.stop_loss_manager import StopLossManager, StopLossConfig, StopLossOrder
from src.interfaces.trading_interfaces import Position


class TestStopLossManager(unittest.TestCase):
    """Test cases for StopLossManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = StopLossConfig(
            stop_loss_pct=0.05,  # 5% stop-loss
            trailing_stop_enabled=False,
            trailing_stop_pct=0.03,
            min_profit_for_trailing=0.02
        )
        self.stop_loss_manager = StopLossManager(self.config)
        
        # Sample position
        self.sample_position = Position(
            symbol="RELIANCE",
            entry_date=datetime.now(),
            entry_price=2500.0,
            quantity=10,
            stop_loss_price=0.0,  # Will be set by manager
            current_value=25000.0,
            unrealized_pnl=0.0,
            status="open"
        )
    
    def test_set_stop_loss_basic(self):
        """Test basic stop-loss setting."""
        stop_price = self.stop_loss_manager.set_stop_loss(self.sample_position)
        
        # Should be 5% below entry price
        expected_stop_price = self.sample_position.entry_price * (1 - self.config.stop_loss_pct)
        self.assertAlmostEqual(stop_price, expected_stop_price, places=2)
        
        # Should be 2375.0 (2500 * 0.95)
        self.assertAlmostEqual(stop_price, 2375.0, places=2)
    
    def test_stop_loss_order_creation(self):
        """Test that stop-loss order is created correctly."""
        stop_price = self.stop_loss_manager.set_stop_loss(self.sample_position)
        
        # Check that order was created
        orders = self.stop_loss_manager.get_all_stop_losses()
        self.assertEqual(len(orders), 1)
        
        # Get the created order
        order = list(orders.values())[0]
        self.assertEqual(order.symbol, self.sample_position.symbol)
        self.assertEqual(order.stop_price, stop_price)
        self.assertEqual(order.original_stop_price, stop_price)
        self.assertFalse(order.is_trailing)
    
    def test_should_execute_stop_loss_triggered(self):
        """Test stop-loss execution when price hits stop level."""
        # Set stop-loss
        stop_price = self.stop_loss_manager.set_stop_loss(self.sample_position)
        
        # Test with price below stop-loss
        current_price = stop_price - 10.0  # Below stop-loss
        should_execute = self.stop_loss_manager.should_execute_stop_loss(
            self.sample_position, current_price
        )
        
        self.assertTrue(should_execute)
    
    def test_should_execute_stop_loss_not_triggered(self):
        """Test stop-loss not executed when price is above stop level."""
        # Set stop-loss
        stop_price = self.stop_loss_manager.set_stop_loss(self.sample_position)
        
        # Test with price above stop-loss
        current_price = stop_price + 10.0  # Above stop-loss
        should_execute = self.stop_loss_manager.should_execute_stop_loss(
            self.sample_position, current_price
        )
        
        self.assertFalse(should_execute)
    
    def test_execute_stop_loss_success(self):
        """Test successful stop-loss execution."""
        # Set stop-loss
        stop_price = self.stop_loss_manager.set_stop_loss(self.sample_position)
        
        # Execute stop-loss
        current_price = stop_price - 5.0
        success, message = self.stop_loss_manager.execute_stop_loss(
            self.sample_position, current_price
        )
        
        self.assertTrue(success)
        self.assertIn("Stop-loss executed", message)
        self.assertIn(self.sample_position.symbol, message)
        
        # Order should be removed after execution
        orders = self.stop_loss_manager.get_all_stop_losses()
        self.assertEqual(len(orders), 0)
    
    def test_execute_stop_loss_no_order(self):
        """Test stop-loss execution when no order exists."""
        # Try to execute without setting stop-loss first
        current_price = 2400.0
        success, message = self.stop_loss_manager.execute_stop_loss(
            self.sample_position, current_price
        )
        
        self.assertFalse(success)
        self.assertIn("No stop-loss order found", message)
    
    def test_get_stop_loss_price(self):
        """Test getting stop-loss price for a position."""
        # Initially should return None
        stop_price = self.stop_loss_manager.get_stop_loss_price(self.sample_position)
        self.assertIsNone(stop_price)
        
        # After setting stop-loss
        expected_stop_price = self.stop_loss_manager.set_stop_loss(self.sample_position)
        actual_stop_price = self.stop_loss_manager.get_stop_loss_price(self.sample_position)
        
        self.assertEqual(actual_stop_price, expected_stop_price)
    
    def test_remove_stop_loss(self):
        """Test removing stop-loss order."""
        # Set stop-loss
        self.stop_loss_manager.set_stop_loss(self.sample_position)
        
        # Verify it exists
        orders = self.stop_loss_manager.get_all_stop_losses()
        self.assertEqual(len(orders), 1)
        
        # Remove it
        success = self.stop_loss_manager.remove_stop_loss(self.sample_position)
        self.assertTrue(success)
        
        # Verify it's gone
        orders = self.stop_loss_manager.get_all_stop_losses()
        self.assertEqual(len(orders), 0)
    
    def test_remove_stop_loss_not_exists(self):
        """Test removing stop-loss that doesn't exist."""
        success = self.stop_loss_manager.remove_stop_loss(self.sample_position)
        self.assertFalse(success)
    
    def test_trailing_stop_loss_enabled(self):
        """Test trailing stop-loss functionality."""
        # Enable trailing stop
        config_with_trailing = StopLossConfig(
            stop_loss_pct=0.05,
            trailing_stop_enabled=True,
            trailing_stop_pct=0.03,
            min_profit_for_trailing=0.02
        )
        trailing_manager = StopLossManager(config_with_trailing)
        
        # Set initial stop-loss
        initial_stop = trailing_manager.set_stop_loss(self.sample_position)
        
        # Price moves up significantly (more than min profit threshold)
        higher_price = self.sample_position.entry_price * 1.05  # 5% profit
        
        # Check if stop-loss should execute (it shouldn't, price is higher)
        should_execute = trailing_manager.should_execute_stop_loss(
            self.sample_position, higher_price
        )
        
        self.assertFalse(should_execute)
        
        # Get updated stop-loss price (should be higher due to trailing)
        updated_stop = trailing_manager.get_stop_loss_price(self.sample_position)
        
        # Updated stop should be higher than initial stop
        self.assertGreater(updated_stop, initial_stop)
    
    def test_trailing_stop_insufficient_profit(self):
        """Test trailing stop when profit is insufficient."""
        # Enable trailing stop
        config_with_trailing = StopLossConfig(
            stop_loss_pct=0.05,
            trailing_stop_enabled=True,
            trailing_stop_pct=0.03,
            min_profit_for_trailing=0.02
        )
        trailing_manager = StopLossManager(config_with_trailing)
        
        # Set initial stop-loss
        initial_stop = trailing_manager.set_stop_loss(self.sample_position)
        
        # Price moves up slightly (less than min profit threshold)
        slightly_higher_price = self.sample_position.entry_price * 1.01  # 1% profit
        
        # Check stop-loss (should not trigger trailing)
        trailing_manager.should_execute_stop_loss(self.sample_position, slightly_higher_price)
        
        # Stop-loss price should remain unchanged
        current_stop = trailing_manager.get_stop_loss_price(self.sample_position)
        self.assertEqual(current_stop, initial_stop)
    
    def test_get_stop_loss_summary(self):
        """Test stop-loss summary statistics."""
        # Initially empty
        summary = self.stop_loss_manager.get_stop_loss_summary()
        self.assertEqual(summary['total_orders'], 0)
        self.assertEqual(summary['trailing_orders'], 0)
        self.assertEqual(summary['fixed_orders'], 0)
        
        # Add some stop-loss orders
        position1 = Position(
            symbol="RELIANCE",
            entry_date=datetime.now(),
            entry_price=2500.0,
            quantity=10,
            stop_loss_price=0.0,
            current_value=25000.0,
            unrealized_pnl=0.0,
            status="open"
        )
        
        position2 = Position(
            symbol="TCS",
            entry_date=datetime.now(),
            entry_price=3000.0,
            quantity=5,
            stop_loss_price=0.0,
            current_value=15000.0,
            unrealized_pnl=0.0,
            status="open"
        )
        
        self.stop_loss_manager.set_stop_loss(position1)
        self.stop_loss_manager.set_stop_loss(position2)
        
        summary = self.stop_loss_manager.get_stop_loss_summary()
        self.assertEqual(summary['total_orders'], 2)
        self.assertEqual(summary['trailing_orders'], 0)
        self.assertEqual(summary['fixed_orders'], 2)
    
    def test_multiple_positions_different_symbols(self):
        """Test managing stop-losses for multiple positions."""
        # Create multiple positions
        positions = []
        for i, symbol in enumerate(["RELIANCE", "TCS", "INFY"], 1):
            position = Position(
                symbol=symbol,
                entry_date=datetime.now() - timedelta(days=i),
                entry_price=2000.0 + (i * 100),
                quantity=10,
                stop_loss_price=0.0,
                current_value=20000.0 + (i * 1000),
                unrealized_pnl=0.0,
                status="open"
            )
            positions.append(position)
            self.stop_loss_manager.set_stop_loss(position)
        
        # Verify all orders created
        orders = self.stop_loss_manager.get_all_stop_losses()
        self.assertEqual(len(orders), 3)
        
        # Test individual stop-loss prices
        for position in positions:
            stop_price = self.stop_loss_manager.get_stop_loss_price(position)
            expected_stop = position.entry_price * (1 - self.config.stop_loss_pct)
            self.assertAlmostEqual(stop_price, expected_stop, places=2)
    
    @patch('src.risk.stop_loss_manager.logging.getLogger')
    def test_error_handling(self, mock_logger):
        """Test error handling in stop-loss management."""
        # Test with invalid position (None values)
        invalid_position = Position(
            symbol=None,  # Invalid
            entry_date=None,  # Invalid
            entry_price=None,  # Invalid
            quantity=10,
            stop_loss_price=0.0,
            current_value=25000.0,
            unrealized_pnl=0.0,
            status="open"
        )
        
        # Should handle gracefully
        stop_price = self.stop_loss_manager.set_stop_loss(invalid_position)
        
        # Should return some default value or handle the error
        self.assertIsNotNone(stop_price)
    
    def test_stop_loss_percentage_accuracy(self):
        """Test accuracy of stop-loss percentage calculation."""
        test_cases = [
            (1000.0, 950.0),   # 1000 * 0.95 = 950
            (2500.0, 2375.0),  # 2500 * 0.95 = 2375
            (5000.0, 4750.0),  # 5000 * 0.95 = 4750
        ]
        
        for entry_price, expected_stop in test_cases:
            position = Position(
                symbol="TEST",
                entry_date=datetime.now(),
                entry_price=entry_price,
                quantity=10,
                stop_loss_price=0.0,
                current_value=entry_price * 10,
                unrealized_pnl=0.0,
                status="open"
            )
            
            stop_price = self.stop_loss_manager.set_stop_loss(position)
            self.assertAlmostEqual(stop_price, expected_stop, places=2)


if __name__ == '__main__':
    unittest.main()