"""
Integration tests for the PositionManager class.
Tests complete position lifecycle from opening to closing positions.
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.trading.position_manager import PositionManager
from src.interfaces.trading_interfaces import Position, TradingSignal


class TestPositionManager(unittest.TestCase):
    """Test cases for PositionManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.initial_capital = 1000000.0  # ₹10 lakh
        self.position_manager = PositionManager(self.initial_capital)
        
        # Sample trading signal
        self.sample_signal = TradingSignal(
            symbol='RELIANCE',
            timestamp=datetime.now(),
            action='buy',
            confidence=0.8,
            lstm_prediction=0.7,
            dqn_q_values={'buy': 0.9, 'sell': 0.1, 'hold': 0.3},
            sentiment_score=0.5,
            risk_adjusted_size=0.02
        )
    
    def test_initialization(self):
        """Test PositionManager initialization."""
        self.assertEqual(self.position_manager.initial_capital, self.initial_capital)
        self.assertEqual(self.position_manager.current_capital, self.initial_capital)
        self.assertEqual(len(self.position_manager.open_positions), 0)
        self.assertEqual(len(self.position_manager.closed_positions), 0)
        self.assertEqual(len(self.position_manager.position_history), 0)
    
    @patch('src.trading.position_manager.PositionManager._log_position_event')
    def test_open_position_success(self, mock_log):
        """Test successful position opening."""
        position_size = 0.02  # 2% of capital
        
        position = self.position_manager.open_position(self.sample_signal, position_size)
        
        # Verify position properties
        self.assertIsInstance(position, Position)
        self.assertEqual(position.symbol, 'RELIANCE')
        self.assertEqual(position.status, 'open')
        self.assertGreater(position.quantity, 0)
        self.assertGreater(position.entry_price, 0)
        self.assertEqual(position.stop_loss_price, position.entry_price * 0.95)
        self.assertEqual(position.unrealized_pnl, 0.0)
        
        # Verify position is tracked
        self.assertIn('RELIANCE', self.position_manager.open_positions)
        self.assertEqual(len(self.position_manager.open_positions), 1)
        
        # Verify capital reduction
        expected_capital = self.initial_capital - (position.quantity * position.entry_price)
        self.assertEqual(self.position_manager.current_capital, expected_capital)
        
        # Verify logging was called
        mock_log.assert_called_once()
    
    def test_open_position_error_handling(self):
        """Test position opening with error handling."""
        # Test with zero position size (should still work but with minimal position)
        position = self.position_manager.open_position(self.sample_signal, 0.0)
        
        # Should create position but with zero quantity
        self.assertEqual(position.symbol, 'RELIANCE')
        self.assertEqual(position.quantity, 0)
        
        # Test with very small position size
        small_position = self.position_manager.open_position(self.sample_signal, 0.0001)
        self.assertIsInstance(small_position, Position)
        self.assertEqual(small_position.symbol, 'RELIANCE')
    
    def test_close_position_success(self):
        """Test successful position closing."""
        # First open a position
        position = self.position_manager.open_position(self.sample_signal, 0.02)
        initial_capital = self.position_manager.current_capital
        
        # Close the position at a higher price (profit)
        close_price = position.entry_price * 1.1  # 10% gain
        closed_position = self.position_manager.close_position(position, close_price)
        
        # Verify position closure
        self.assertEqual(closed_position.status, 'closed')
        self.assertGreater(closed_position.unrealized_pnl, 0)  # Should be profitable
        
        # Verify position tracking
        self.assertNotIn('RELIANCE', self.position_manager.open_positions)
        self.assertEqual(len(self.position_manager.closed_positions), 1)
        
        # Verify capital update
        expected_final_value = position.quantity * close_price
        expected_capital = initial_capital + expected_final_value
        self.assertEqual(self.position_manager.current_capital, expected_capital)
    
    def test_close_position_with_loss(self):
        """Test closing position with a loss."""
        # Open position
        position = self.position_manager.open_position(self.sample_signal, 0.02)
        
        # Close at lower price (loss)
        close_price = position.entry_price * 0.9  # 10% loss
        closed_position = self.position_manager.close_position(position, close_price)
        
        # Verify loss
        self.assertEqual(closed_position.status, 'closed')
        self.assertLess(closed_position.unrealized_pnl, 0)  # Should be negative
    
    def test_close_non_open_position(self):
        """Test attempting to close non-open position."""
        # Create a closed position
        closed_position = Position(
            symbol='TEST',
            entry_date=datetime.now(),
            entry_price=100.0,
            quantity=100,
            stop_loss_price=95.0,
            current_value=10000.0,
            unrealized_pnl=0.0,
            status='closed'
        )
        
        result = self.position_manager.close_position(closed_position, 110.0)
        
        # Should return unchanged position
        self.assertEqual(result.status, 'closed')
        self.assertEqual(result.unrealized_pnl, 0.0)
    
    def test_update_position_value(self):
        """Test updating position value with current price."""
        # Open position
        position = self.position_manager.open_position(self.sample_signal, 0.02)
        original_value = position.current_value
        
        # Update with higher price
        new_price = position.entry_price * 1.05  # 5% increase
        updated_position = self.position_manager.update_position_value(position, new_price)
        
        # Verify updates
        self.assertGreater(updated_position.current_value, original_value)
        self.assertGreater(updated_position.unrealized_pnl, 0)
        
        # Verify position is updated in tracking
        tracked_position = self.position_manager.open_positions['RELIANCE']
        self.assertEqual(tracked_position.current_value, updated_position.current_value)
    
    def test_update_closed_position_value(self):
        """Test that closed positions are not updated."""
        # Open and close position
        position = self.position_manager.open_position(self.sample_signal, 0.02)
        closed_position = self.position_manager.close_position(position, position.entry_price)
        
        # Attempt to update closed position
        result = self.position_manager.update_position_value(closed_position, 150.0)
        
        # Should remain unchanged
        self.assertEqual(result.status, 'closed')
        self.assertEqual(result.current_value, closed_position.current_value)
    
    def test_get_open_positions(self):
        """Test getting all open positions."""
        # Initially empty
        positions = self.position_manager.get_open_positions()
        self.assertEqual(len(positions), 0)
        
        # Open multiple positions
        signal1 = self.sample_signal
        signal2 = TradingSignal(
            symbol='TCS',
            timestamp=datetime.now(),
            action='buy',
            confidence=0.7,
            lstm_prediction=0.6,
            dqn_q_values={'buy': 0.8, 'sell': 0.2, 'hold': 0.4},
            sentiment_score=0.3,
            risk_adjusted_size=0.015
        )
        
        self.position_manager.open_position(signal1, 0.02)
        self.position_manager.open_position(signal2, 0.015)
        
        positions = self.position_manager.get_open_positions()
        self.assertEqual(len(positions), 2)
        
        symbols = [pos.symbol for pos in positions]
        self.assertIn('RELIANCE', symbols)
        self.assertIn('TCS', symbols)
    
    def test_get_position_by_symbol(self):
        """Test getting position by symbol."""
        # No position initially
        position = self.position_manager.get_position_by_symbol('RELIANCE')
        self.assertIsNone(position)
        
        # Open position
        self.position_manager.open_position(self.sample_signal, 0.02)
        
        # Should find position
        position = self.position_manager.get_position_by_symbol('RELIANCE')
        self.assertIsNotNone(position)
        self.assertEqual(position.symbol, 'RELIANCE')
        
        # Should not find non-existent position
        position = self.position_manager.get_position_by_symbol('NONEXISTENT')
        self.assertIsNone(position)
    
    def test_has_position(self):
        """Test checking if position exists."""
        # Initially no positions
        self.assertFalse(self.position_manager.has_position('RELIANCE'))
        
        # Open position
        self.position_manager.open_position(self.sample_signal, 0.02)
        
        # Should have position
        self.assertTrue(self.position_manager.has_position('RELIANCE'))
        self.assertFalse(self.position_manager.has_position('TCS'))
    
    def test_get_portfolio_value(self):
        """Test calculating total portfolio value."""
        # Initially should equal initial capital
        portfolio_value = self.position_manager.get_portfolio_value()
        self.assertEqual(portfolio_value, self.initial_capital)
        
        # Open position
        position = self.position_manager.open_position(self.sample_signal, 0.02)
        
        # Portfolio value should still equal initial capital (cash + position value)
        portfolio_value = self.position_manager.get_portfolio_value()
        self.assertAlmostEqual(portfolio_value, self.initial_capital, places=2)
        
        # Update position value
        new_price = position.entry_price * 1.1
        self.position_manager.update_position_value(position, new_price)
        
        # Portfolio value should increase
        portfolio_value = self.position_manager.get_portfolio_value()
        self.assertGreater(portfolio_value, self.initial_capital)
    
    def test_get_total_pnl(self):
        """Test calculating total P&L."""
        # Initially zero
        pnl = self.position_manager.get_total_pnl()
        self.assertEqual(pnl, 0.0)
        
        # Open position and update value
        position = self.position_manager.open_position(self.sample_signal, 0.02)
        new_price = position.entry_price * 1.1  # 10% gain
        self.position_manager.update_position_value(position, new_price)
        
        # Should have positive unrealized P&L
        pnl = self.position_manager.get_total_pnl()
        self.assertGreater(pnl, 0)
        
        # Close position
        self.position_manager.close_position(position, new_price)
        
        # Should still have positive P&L (now realized)
        pnl = self.position_manager.get_total_pnl()
        self.assertGreater(pnl, 0)
    
    def test_get_position_count(self):
        """Test getting number of open positions."""
        self.assertEqual(self.position_manager.get_position_count(), 0)
        
        # Open positions
        self.position_manager.open_position(self.sample_signal, 0.02)
        self.assertEqual(self.position_manager.get_position_count(), 1)
        
        signal2 = TradingSignal(
            symbol='TCS',
            timestamp=datetime.now(),
            action='buy',
            confidence=0.7,
            lstm_prediction=0.6,
            dqn_q_values={'buy': 0.8, 'sell': 0.2, 'hold': 0.4},
            sentiment_score=0.3,
            risk_adjusted_size=0.015
        )
        self.position_manager.open_position(signal2, 0.015)
        self.assertEqual(self.position_manager.get_position_count(), 2)
    
    def test_get_available_capital(self):
        """Test getting available capital."""
        # Initially should equal initial capital
        available = self.position_manager.get_available_capital()
        self.assertEqual(available, self.initial_capital)
        
        # Open position
        position = self.position_manager.open_position(self.sample_signal, 0.02)
        
        # Available capital should decrease
        available = self.position_manager.get_available_capital()
        expected = self.initial_capital - (position.quantity * position.entry_price)
        self.assertEqual(available, expected)
    
    def test_update_positions_batch(self):
        """Test batch updating of positions."""
        # Open multiple positions
        signal1 = self.sample_signal
        signal2 = TradingSignal(
            symbol='TCS',
            timestamp=datetime.now(),
            action='buy',
            confidence=0.7,
            lstm_prediction=0.6,
            dqn_q_values={'buy': 0.8, 'sell': 0.2, 'hold': 0.4},
            sentiment_score=0.3,
            risk_adjusted_size=0.015
        )
        
        pos1 = self.position_manager.open_position(signal1, 0.02)
        pos2 = self.position_manager.open_position(signal2, 0.015)
        
        # Update with new prices
        price_data = {
            'RELIANCE': pos1.entry_price * 1.05,
            'TCS': pos2.entry_price * 0.98
        }
        
        updated_positions = self.position_manager.update_positions_batch(price_data)
        
        # Verify updates
        self.assertEqual(len(updated_positions), 2)
        self.assertIn('RELIANCE', updated_positions)
        self.assertIn('TCS', updated_positions)
        
        # Verify P&L changes
        self.assertGreater(updated_positions['RELIANCE'].unrealized_pnl, 0)  # Profit
        self.assertLess(updated_positions['TCS'].unrealized_pnl, 0)  # Loss
    
    def test_check_stop_losses(self):
        """Test stop-loss checking functionality."""
        # Open position
        position = self.position_manager.open_position(self.sample_signal, 0.02)
        
        # Price above stop-loss (no trigger)
        price_data = {'RELIANCE': position.stop_loss_price * 1.01}
        stop_loss_positions = self.position_manager.check_stop_losses(price_data)
        self.assertEqual(len(stop_loss_positions), 0)
        
        # Price at stop-loss (trigger)
        price_data = {'RELIANCE': position.stop_loss_price}
        stop_loss_positions = self.position_manager.check_stop_losses(price_data)
        self.assertEqual(len(stop_loss_positions), 1)
        self.assertEqual(stop_loss_positions[0].symbol, 'RELIANCE')
        
        # Price below stop-loss (trigger)
        price_data = {'RELIANCE': position.stop_loss_price * 0.99}
        stop_loss_positions = self.position_manager.check_stop_losses(price_data)
        self.assertEqual(len(stop_loss_positions), 1)
    
    def test_close_positions_batch(self):
        """Test batch closing of positions."""
        # Open multiple positions
        signal1 = self.sample_signal
        signal2 = TradingSignal(
            symbol='TCS',
            timestamp=datetime.now(),
            action='buy',
            confidence=0.7,
            lstm_prediction=0.6,
            dqn_q_values={'buy': 0.8, 'sell': 0.2, 'hold': 0.4},
            sentiment_score=0.3,
            risk_adjusted_size=0.015
        )
        
        pos1 = self.position_manager.open_position(signal1, 0.02)
        pos2 = self.position_manager.open_position(signal2, 0.015)
        
        # Close both positions
        positions_to_close = [pos1, pos2]
        price_data = {
            'RELIANCE': pos1.entry_price * 1.1,
            'TCS': pos2.entry_price * 0.9
        }
        
        closed_positions = self.position_manager.close_positions_batch(positions_to_close, price_data)
        
        # Verify closures
        self.assertEqual(len(closed_positions), 2)
        self.assertEqual(len(self.position_manager.open_positions), 0)
        self.assertEqual(len(self.position_manager.closed_positions), 2)
        
        # Verify P&L
        reliance_pos = next(pos for pos in closed_positions if pos.symbol == 'RELIANCE')
        tcs_pos = next(pos for pos in closed_positions if pos.symbol == 'TCS')
        
        self.assertGreater(reliance_pos.unrealized_pnl, 0)  # Profit
        self.assertLess(tcs_pos.unrealized_pnl, 0)  # Loss
    
    def test_get_position_statistics(self):
        """Test getting comprehensive position statistics."""
        # Open and close some positions
        pos1 = self.position_manager.open_position(self.sample_signal, 0.02)
        
        # Close with profit
        self.position_manager.close_position(pos1, pos1.entry_price * 1.1)
        
        # Open another position
        signal2 = TradingSignal(
            symbol='TCS',
            timestamp=datetime.now(),
            action='buy',
            confidence=0.7,
            lstm_prediction=0.6,
            dqn_q_values={'buy': 0.8, 'sell': 0.2, 'hold': 0.4},
            sentiment_score=0.3,
            risk_adjusted_size=0.015
        )
        pos2 = self.position_manager.open_position(signal2, 0.015)
        
        # Close with loss
        self.position_manager.close_position(pos2, pos2.entry_price * 0.9)
        
        stats = self.position_manager.get_position_statistics()
        
        # Verify statistics
        self.assertIsInstance(stats, dict)
        self.assertEqual(stats['open_positions'], 0)
        self.assertEqual(stats['closed_positions'], 2)
        self.assertEqual(stats['total_positions'], 2)
        self.assertEqual(stats['win_count'], 1)
        self.assertEqual(stats['loss_count'], 1)
        self.assertEqual(stats['win_rate'], 50.0)
        self.assertGreater(stats['average_win'], 0)
        self.assertLess(stats['average_loss'], 0)
        self.assertIn('portfolio_value', stats)
        self.assertIn('total_return_pct', stats)
    
    def test_position_history_logging(self):
        """Test that position events are logged correctly."""
        # Open position
        position = self.position_manager.open_position(self.sample_signal, 0.02)
        
        # Update position
        self.position_manager.update_position_value(position, position.entry_price * 1.05)
        
        # Close position
        self.position_manager.close_position(position, position.entry_price * 1.1)
        
        # Check history
        history = self.position_manager.get_position_history()
        self.assertGreater(len(history), 0)
        
        # Verify event types
        event_types = [event['event_type'] for event in history]
        self.assertIn('OPEN', event_types)
        self.assertIn('CLOSE', event_types)
    
    def test_reset_portfolio(self):
        """Test portfolio reset functionality."""
        # Open position and modify state
        self.position_manager.open_position(self.sample_signal, 0.02)
        
        # Verify state is modified
        self.assertGreater(len(self.position_manager.open_positions), 0)
        self.assertLess(self.position_manager.current_capital, self.initial_capital)
        
        # Reset portfolio
        new_capital = 2000000.0
        self.position_manager.reset_portfolio(new_capital)
        
        # Verify reset
        self.assertEqual(self.position_manager.initial_capital, new_capital)
        self.assertEqual(self.position_manager.current_capital, new_capital)
        self.assertEqual(len(self.position_manager.open_positions), 0)
        self.assertEqual(len(self.position_manager.closed_positions), 0)
        self.assertEqual(len(self.position_manager.position_history), 0)
    
    def test_export_positions_to_dataframe(self):
        """Test exporting position history to DataFrame."""
        # Initially empty
        df = self.position_manager.export_positions_to_dataframe()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 0)
        
        # Open and close position
        position = self.position_manager.open_position(self.sample_signal, 0.02)
        self.position_manager.close_position(position, position.entry_price * 1.1)
        
        # Export to DataFrame
        df = self.position_manager.export_positions_to_dataframe()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)
        
        # Verify columns
        expected_columns = ['timestamp', 'event_type', 'symbol', 'quantity', 'entry_price']
        for col in expected_columns:
            self.assertIn(col, df.columns)
    
    def test_complete_position_lifecycle(self):
        """Integration test for complete position lifecycle."""
        initial_capital = self.position_manager.current_capital
        
        # 1. Open position
        position = self.position_manager.open_position(self.sample_signal, 0.02)
        self.assertEqual(position.status, 'open')
        self.assertLess(self.position_manager.current_capital, initial_capital)
        
        # 2. Update position value (simulate price movement)
        new_price = position.entry_price * 1.08  # 8% gain
        updated_position = self.position_manager.update_position_value(position, new_price)
        self.assertGreater(updated_position.unrealized_pnl, 0)
        
        # 3. Check portfolio value
        portfolio_value = self.position_manager.get_portfolio_value()
        self.assertGreater(portfolio_value, initial_capital)
        
        # 4. Close position
        final_price = position.entry_price * 1.12  # 12% final gain
        closed_position = self.position_manager.close_position(position, final_price)
        self.assertEqual(closed_position.status, 'closed')
        self.assertGreater(closed_position.unrealized_pnl, 0)
        
        # 5. Verify final state
        self.assertEqual(len(self.position_manager.open_positions), 0)
        self.assertEqual(len(self.position_manager.closed_positions), 1)
        self.assertGreater(self.position_manager.current_capital, initial_capital)
        
        # 6. Verify statistics
        stats = self.position_manager.get_position_statistics()
        self.assertEqual(stats['win_count'], 1)
        self.assertEqual(stats['loss_count'], 0)
        self.assertEqual(stats['win_rate'], 100.0)
        self.assertGreater(stats['total_return_pct'], 0)


if __name__ == '__main__':
    unittest.main()