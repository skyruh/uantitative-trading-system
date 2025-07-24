"""
Integration tests for DQN Training System.
Tests complete DQN training workflow including state representation,
reward calculation, and integration with LSTM predictions.
"""

import unittest
import numpy as np
import pandas as pd
import tempfile
import shutil
import os
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime, timedelta

# Add src to path for imports
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.dqn_trainer import (
    DQNTrainer, PortfolioSimulator, StateBuilder, 
    TradingState, TradingAction, TradingReward
)
from models.dqn_agent import DQNAgent
from models.lstm_model import LSTMModel


class TestPortfolioSimulator(unittest.TestCase):
    """Test cases for Portfolio Simulator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.portfolio = PortfolioSimulator(initial_capital=100000.0, transaction_cost=0.001)
    
    def test_portfolio_initialization(self):
        """Test portfolio initialization."""
        self.assertEqual(self.portfolio.initial_capital, 100000.0)
        self.assertEqual(self.portfolio.cash, 100000.0)
        self.assertEqual(self.portfolio.portfolio_value, 100000.0)
        self.assertEqual(len(self.portfolio.positions), 0)
        self.assertEqual(len(self.portfolio.trade_history), 0)
    
    def test_buy_trade_execution(self):
        """Test buy trade execution."""
        success = self.portfolio.execute_trade('RELIANCE', 'buy', 2000.0, 10)
        
        self.assertTrue(success)
        self.assertEqual(self.portfolio.positions['RELIANCE'], 10)
        self.assertLess(self.portfolio.cash, 100000.0)  # Cash reduced
        self.assertEqual(len(self.portfolio.trade_history), 1)
        
        # Check trade details
        trade = self.portfolio.trade_history[0]
        self.assertEqual(trade['action'], 'buy')
        self.assertEqual(trade['quantity'], 10)
        self.assertEqual(trade['price'], 2000.0)
    
    def test_sell_trade_execution(self):
        """Test sell trade execution."""
        # First buy some shares
        self.portfolio.execute_trade('RELIANCE', 'buy', 2000.0, 10)
        initial_cash = self.portfolio.cash
        
        # Then sell some shares
        success = self.portfolio.execute_trade('RELIANCE', 'sell', 2100.0, 5)
        
        self.assertTrue(success)
        self.assertEqual(self.portfolio.positions['RELIANCE'], 5)
        self.assertGreater(self.portfolio.cash, initial_cash)  # Cash increased
        self.assertEqual(len(self.portfolio.trade_history), 2)
    
    def test_hold_action(self):
        """Test hold action."""
        initial_cash = self.portfolio.cash
        success = self.portfolio.execute_trade('RELIANCE', 'hold', 2000.0)
        
        self.assertTrue(success)
        self.assertEqual(self.portfolio.cash, initial_cash)
        self.assertEqual(len(self.portfolio.positions), 0)
        self.assertEqual(len(self.portfolio.trade_history), 0)
    
    def test_insufficient_cash_buy(self):
        """Test buy with insufficient cash."""
        # Try to buy more than available cash
        success = self.portfolio.execute_trade('RELIANCE', 'buy', 2000.0, 100)
        
        self.assertFalse(success)
        self.assertEqual(len(self.portfolio.positions), 0)
        self.assertEqual(self.portfolio.cash, 100000.0)
    
    def test_sell_without_position(self):
        """Test sell without holding position."""
        success = self.portfolio.execute_trade('RELIANCE', 'sell', 2000.0, 10)
        
        self.assertFalse(success)
        self.assertEqual(len(self.portfolio.positions), 0)
        self.assertEqual(len(self.portfolio.trade_history), 0)
    
    def test_portfolio_value_update(self):
        """Test portfolio value update."""
        # Buy some shares
        self.portfolio.execute_trade('RELIANCE', 'buy', 2000.0, 10)
        
        # Update with new prices
        new_prices = {'RELIANCE': 2200.0}
        portfolio_value = self.portfolio.update_portfolio_value(new_prices)
        
        expected_value = self.portfolio.cash + (10 * 2200.0)
        self.assertAlmostEqual(portfolio_value, expected_value, places=2)
        self.assertAlmostEqual(self.portfolio.portfolio_value, expected_value, places=2)
    
    def test_return_calculation(self):
        """Test return calculation."""
        # Initial return should be 0
        initial_return = self.portfolio.calculate_return()
        self.assertEqual(initial_return, 0.0)
        
        # Buy shares and update value
        self.portfolio.execute_trade('RELIANCE', 'buy', 2000.0, 10)
        self.portfolio.update_portfolio_value({'RELIANCE': 2200.0})
        
        # Calculate return
        portfolio_return = self.portfolio.calculate_return()
        self.assertGreater(portfolio_return, 0.0)  # Should be positive
    
    def test_portfolio_statistics(self):
        """Test portfolio statistics calculation."""
        # Execute some trades
        self.portfolio.execute_trade('RELIANCE', 'buy', 2000.0, 10)
        self.portfolio.update_portfolio_value({'RELIANCE': 2100.0})
        self.portfolio.calculate_return()
        
        stats = self.portfolio.get_portfolio_stats()
        
        self.assertIn('total_return', stats)
        self.assertIn('average_return', stats)
        self.assertIn('volatility', stats)
        self.assertIn('sharpe_ratio', stats)
        self.assertIn('num_trades', stats)
        self.assertEqual(stats['num_trades'], 1)
    
    def test_portfolio_reset(self):
        """Test portfolio reset functionality."""
        # Execute some trades
        self.portfolio.execute_trade('RELIANCE', 'buy', 2000.0, 10)
        
        # Reset portfolio
        self.portfolio.reset()
        
        self.assertEqual(self.portfolio.cash, 100000.0)
        self.assertEqual(len(self.portfolio.positions), 0)
        self.assertEqual(len(self.portfolio.trade_history), 0)
        self.assertEqual(self.portfolio.portfolio_value, 100000.0)


class TestStateBuilder(unittest.TestCase):
    """Test cases for State Builder."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_lstm = Mock()
        self.mock_lstm.is_trained = True
        self.mock_lstm.predict_price_movement.return_value = 0.7
        
        self.state_builder = StateBuilder(self.mock_lstm)
        self.portfolio = PortfolioSimulator()
        
        # Create sample data
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        self.sample_data = pd.DataFrame({
            'date': dates,
            'close': np.random.uniform(1900, 2100, 100),
            'high': np.random.uniform(2000, 2200, 100),
            'low': np.random.uniform(1800, 2000, 100),
            'volume': np.random.uniform(1000000, 5000000, 100),
            'rsi_14': np.random.uniform(20, 80, 100),
            'sma_50': np.random.uniform(1950, 2050, 100),
            'bb_upper': np.random.uniform(2050, 2150, 100),
            'bb_middle': np.random.uniform(2000, 2100, 100),
            'bb_lower': np.random.uniform(1950, 2050, 100),
            'sentiment_score': np.random.uniform(-0.5, 0.5, 100)
        })
    
    def test_state_building(self):
        """Test state vector building."""
        state = self.state_builder.build_state(
            self.sample_data, self.portfolio, 'RELIANCE', 70
        )
        
        self.assertIsInstance(state, np.ndarray)
        self.assertEqual(len(state), 10)  # Expected state size
        
        # Check that all values are reasonable
        self.assertTrue(all(np.isfinite(state)))
        self.assertGreaterEqual(state[0], 0.0)  # LSTM prediction
        self.assertLessEqual(state[0], 1.0)
    
    def test_state_without_lstm(self):
        """Test state building without LSTM model."""
        state_builder = StateBuilder(None)
        state = state_builder.build_state(
            self.sample_data, self.portfolio, 'RELIANCE', 70
        )
        
        self.assertIsInstance(state, np.ndarray)
        self.assertEqual(len(state), 10)
        self.assertEqual(state[0], 0.5)  # Default LSTM prediction
    
    def test_state_with_portfolio_positions(self):
        """Test state building with portfolio positions."""
        # Add position to portfolio
        self.portfolio.execute_trade('RELIANCE', 'buy', 2000.0, 10)
        self.portfolio.update_portfolio_value({'RELIANCE': 2100.0})
        
        state = self.state_builder.build_state(
            self.sample_data, self.portfolio, 'RELIANCE', 70
        )
        
        # Position ratio should be > 0
        position_ratio = state[8]
        self.assertGreater(position_ratio, 0.0)
    
    def test_bollinger_band_position_calculation(self):
        """Test Bollinger Band position calculation."""
        # Test data with known BB values
        test_data = pd.Series({
            'close': 2000.0,
            'bb_upper': 2100.0,
            'bb_lower': 1900.0
        })
        
        bb_position = self.state_builder._calculate_bb_position(test_data)
        expected_position = (2000.0 - 1900.0) / (2100.0 - 1900.0)
        
        self.assertAlmostEqual(bb_position, expected_position, places=3)
    
    def test_price_change_calculation(self):
        """Test price change calculation."""
        price_change = self.state_builder._calculate_price_change(self.sample_data, 70)
        
        self.assertIsInstance(price_change, float)
        self.assertGreaterEqual(price_change, -1.0)
        self.assertLessEqual(price_change, 1.0)
    
    def test_volume_ratio_calculation(self):
        """Test volume ratio calculation."""
        volume_ratio = self.state_builder._calculate_volume_ratio(self.sample_data, 70)
        
        self.assertIsInstance(volume_ratio, float)
        # Should be normalized around 0
        self.assertGreaterEqual(volume_ratio, -1.0)
        self.assertLessEqual(volume_ratio, 1.0)
    
    def test_error_handling_insufficient_data(self):
        """Test error handling with insufficient data."""
        # Test with very early index
        state = self.state_builder.build_state(
            self.sample_data, self.portfolio, 'RELIANCE', 2
        )
        
        self.assertIsInstance(state, np.ndarray)
        self.assertEqual(len(state), 10)
        # Should not crash and return valid state


class TestDQNTrainer(unittest.TestCase):
    """Test cases for DQN Trainer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.dqn_agent = DQNAgent(state_size=10, batch_size=4, random_seed=42)
        self.mock_lstm = Mock()
        self.mock_lstm.is_trained = True
        self.mock_lstm.predict_price_movement.return_value = 0.6
        
        self.trainer = DQNTrainer(
            dqn_agent=self.dqn_agent,
            lstm_model=self.mock_lstm,
            initial_capital=100000.0
        )
        
        # Create sample training data
        dates = pd.date_range('2023-01-01', periods=200, freq='D')
        self.training_data = pd.DataFrame({
            'date': dates,
            'close': np.cumsum(np.random.normal(0, 10, 200)) + 2000,
            'high': np.cumsum(np.random.normal(0, 10, 200)) + 2050,
            'low': np.cumsum(np.random.normal(0, 10, 200)) + 1950,
            'volume': np.random.uniform(1000000, 5000000, 200),
            'rsi_14': np.random.uniform(20, 80, 200),
            'sma_50': np.cumsum(np.random.normal(0, 8, 200)) + 2000,
            'bb_upper': np.cumsum(np.random.normal(0, 10, 200)) + 2100,
            'bb_middle': np.cumsum(np.random.normal(0, 10, 200)) + 2000,
            'bb_lower': np.cumsum(np.random.normal(0, 10, 200)) + 1900,
            'sentiment_score': np.random.uniform(-0.5, 0.5, 200)
        })
        
        # Ensure prices are positive
        self.training_data['close'] = np.abs(self.training_data['close'])
        self.training_data['high'] = np.abs(self.training_data['high'])
        self.training_data['low'] = np.abs(self.training_data['low'])
        self.training_data['sma_50'] = np.abs(self.training_data['sma_50'])
        
        # Create benchmark data
        self.benchmark_data = pd.DataFrame({
            'date': dates,
            'close': np.cumsum(np.random.normal(0, 5, 200)) + 15000  # NIFTY-like values
        })
        self.benchmark_data['close'] = np.abs(self.benchmark_data['close'])
    
    def test_trainer_initialization(self):
        """Test DQN trainer initialization."""
        self.assertIsNotNone(self.trainer.dqn_agent)
        self.assertIsNotNone(self.trainer.lstm_model)
        self.assertIsNotNone(self.trainer.portfolio)
        self.assertIsNotNone(self.trainer.state_builder)
        self.assertEqual(self.trainer.initial_capital, 100000.0)
    
    def test_sharpe_reward_calculation(self):
        """Test Sharpe ratio-based reward calculation."""
        # Test positive excess return
        reward_info = self.trainer.calculate_sharpe_reward(0.15, 0.10)
        
        self.assertIsInstance(reward_info, TradingReward)
        self.assertEqual(reward_info.portfolio_return, 0.15)
        self.assertEqual(reward_info.benchmark_return, 0.10)
        self.assertGreater(reward_info.reward_value, 0.0)  # Should be positive
        self.assertIn('excess_return', reward_info.calculation_details)
        
        # Test negative excess return
        reward_info_neg = self.trainer.calculate_sharpe_reward(0.05, 0.10)
        self.assertLess(reward_info_neg.reward_value, 0.0)  # Should be negative
    
    def test_single_episode_training(self):
        """Test single episode training."""
        episode_stats = self.trainer.train_episode(
            self.training_data, 'RELIANCE', self.benchmark_data
        )
        
        self.assertIsInstance(episode_stats, dict)
        self.assertIn('total_reward', episode_stats)
        self.assertIn('average_reward', episode_stats)
        self.assertIn('num_actions', episode_stats)
        self.assertIn('final_portfolio_value', episode_stats)
        self.assertIn('total_return', episode_stats)
        
        # Check action counts
        total_actions = (episode_stats['buy_actions'] + 
                        episode_stats['sell_actions'] + 
                        episode_stats['hold_actions'])
        self.assertEqual(total_actions, episode_stats['num_actions'])
    
    def test_multi_episode_training(self):
        """Test multi-episode training."""
        training_data_dict = {
            'RELIANCE': self.training_data,
            'TCS': self.training_data.copy()  # Use same data for simplicity
        }
        
        success = self.trainer.train(
            training_data_dict, self.benchmark_data, episodes=5
        )
        
        self.assertTrue(success)
        self.assertGreater(len(self.trainer.training_history), 0)
        self.assertGreater(len(self.trainer.episode_rewards), 0)
        
        # Check training history structure
        history_entry = self.trainer.training_history[0]
        self.assertIn('episode', history_entry)
        self.assertIn('average_reward', history_entry)
        self.assertIn('average_return', history_entry)
        self.assertIn('epsilon', history_entry)
    
    def test_training_with_insufficient_data(self):
        """Test training with insufficient data."""
        # Create small dataset
        small_data = self.training_data.head(50)
        
        episode_stats = self.trainer.train_episode(small_data, 'RELIANCE')
        
        # Should return empty dict due to insufficient data
        self.assertEqual(episode_stats, {})
    
    def test_training_metrics_retrieval(self):
        """Test training metrics retrieval."""
        # Train for a few episodes
        training_data_dict = {'RELIANCE': self.training_data}
        self.trainer.train(training_data_dict, episodes=3)
        
        metrics = self.trainer.get_training_metrics()
        
        self.assertIsInstance(metrics, dict)
        self.assertIn('total_episodes', metrics)
        self.assertIn('final_average_reward', metrics)
        self.assertIn('best_average_reward', metrics)
        self.assertIn('agent_metrics', metrics)
        
        self.assertEqual(metrics['total_episodes'], 3)
    
    def test_reward_calculation_edge_cases(self):
        """Test reward calculation with edge cases."""
        # Test with zero returns
        reward_zero = self.trainer.calculate_sharpe_reward(0.0, 0.0)
        self.assertAlmostEqual(reward_zero.reward_value, 0.0, places=3)
        
        # Test with extreme values
        reward_extreme = self.trainer.calculate_sharpe_reward(1.0, -1.0)
        self.assertGreaterEqual(reward_extreme.reward_value, -1.0)
        self.assertLessEqual(reward_extreme.reward_value, 1.0)
    
    def test_state_builder_integration(self):
        """Test integration with state builder."""
        # Test that state builder is properly integrated
        state = self.trainer.state_builder.build_state(
            self.training_data, self.trainer.portfolio, 'RELIANCE', 100
        )
        
        self.assertIsInstance(state, np.ndarray)
        self.assertEqual(len(state), 10)
        
        # LSTM prediction should be called
        self.mock_lstm.predict_price_movement.assert_called()
    
    def test_portfolio_integration(self):
        """Test integration with portfolio simulator."""
        # Execute a trade through trainer's portfolio
        success = self.trainer.portfolio.execute_trade('RELIANCE', 'buy', 2000.0, 10)
        self.assertTrue(success)
        
        # Update portfolio value
        self.trainer.portfolio.update_portfolio_value({'RELIANCE': 2100.0})
        
        # Calculate return
        portfolio_return = self.trainer.portfolio.calculate_return()
        self.assertGreater(portfolio_return, 0.0)
    
    def test_error_handling_in_training(self):
        """Test error handling during training."""
        # Test with corrupted data
        corrupted_data = self.training_data.copy()
        corrupted_data.loc[50:60, 'close'] = np.nan
        
        # Should not crash
        episode_stats = self.trainer.train_episode(corrupted_data, 'RELIANCE')
        
        # May return empty dict or partial results, but should not crash
        self.assertIsInstance(episode_stats, dict)
    
    def test_training_without_lstm(self):
        """Test training without LSTM model."""
        trainer_no_lstm = DQNTrainer(
            dqn_agent=self.dqn_agent,
            lstm_model=None,
            initial_capital=100000.0
        )
        
        episode_stats = trainer_no_lstm.train_episode(self.training_data, 'RELIANCE')
        
        self.assertIsInstance(episode_stats, dict)
        # Should work without LSTM, using default predictions
    
    def test_training_progress_tracking(self):
        """Test that training progress is properly tracked."""
        training_data_dict = {'RELIANCE': self.training_data}
        
        # Initial state
        initial_epsilon = self.trainer.dqn_agent.epsilon
        
        # Train
        self.trainer.train(training_data_dict, episodes=5)
        
        # Check that epsilon has decayed
        final_epsilon = self.trainer.dqn_agent.epsilon
        self.assertLessEqual(final_epsilon, initial_epsilon)
        
        # Check that training history is recorded
        self.assertEqual(len(self.trainer.training_history), 5)


if __name__ == '__main__':
    # Set up logging to reduce noise during testing
    import logging
    logging.getLogger().setLevel(logging.CRITICAL)
    
    unittest.main()