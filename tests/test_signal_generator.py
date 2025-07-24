"""
Unit tests for the SignalGenerator class.
Tests signal generation, validation, and confidence scoring functionality.
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime
from unittest.mock import Mock, MagicMock

from src.trading.signal_generator import SignalGenerator
from src.interfaces.trading_interfaces import TradingSignal
from src.models.lstm_model import LSTMModel
from src.models.dqn_agent import DQNAgent


class TestSignalGenerator(unittest.TestCase):
    """Test cases for SignalGenerator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock models
        self.mock_lstm = Mock(spec=LSTMModel)
        self.mock_lstm.is_trained = True
        self.mock_lstm.predict_price_movement.return_value = 0.7
        
        self.mock_dqn = Mock(spec=DQNAgent)
        self.mock_dqn.is_trained = True
        self.mock_dqn.get_q_values.return_value = {'buy': 0.8, 'sell': 0.2, 'hold': 0.5}
        
        # Create signal generator
        self.signal_generator = SignalGenerator(
            lstm_model=self.mock_lstm,
            dqn_agent=self.mock_dqn,
            min_confidence=0.6
        )
        
        # Sample market data
        self.sample_market_data = {
            'close': 100.0,
            'high': 105.0,
            'low': 95.0,
            'volume': 1000000,
            'rsi_14': 65.0,
            'sma_50': 98.0,
            'bb_upper': 110.0,
            'bb_middle': 100.0,
            'bb_lower': 90.0,
            'sentiment_score': 0.3
        }
    
    def test_initialization(self):
        """Test SignalGenerator initialization."""
        # Test default initialization
        sg = SignalGenerator()
        self.assertEqual(sg.min_confidence, 0.6)
        self.assertEqual(sg.sentiment_weight, 0.2)
        self.assertEqual(sg.lstm_weight, 0.4)
        self.assertEqual(sg.dqn_weight, 0.4)
        
        # Test custom initialization
        sg_custom = SignalGenerator(
            min_confidence=0.8,
            sentiment_weight=0.3,
            lstm_weight=0.35,
            dqn_weight=0.35
        )
        self.assertEqual(sg_custom.min_confidence, 0.8)
        self.assertEqual(sg_custom.sentiment_weight, 0.3)
    
    def test_weight_normalization(self):
        """Test that weights are normalized to sum to 1."""
        sg = SignalGenerator(
            sentiment_weight=0.4,
            lstm_weight=0.4,
            dqn_weight=0.4  # Sum = 1.2, should be normalized
        )
        
        total_weight = sg.sentiment_weight + sg.lstm_weight + sg.dqn_weight
        self.assertAlmostEqual(total_weight, 1.0, places=2)
    
    def test_set_models(self):
        """Test setting LSTM model and DQN agent."""
        sg = SignalGenerator()
        sg.set_models(self.mock_lstm, self.mock_dqn)
        
        self.assertEqual(sg.lstm_model, self.mock_lstm)
        self.assertEqual(sg.dqn_agent, self.mock_dqn)
    
    def test_prepare_state_vector(self):
        """Test state vector preparation for DQN."""
        state_vector = self.signal_generator._prepare_state_vector(
            self.sample_market_data, 0.7, 0.3
        )
        
        self.assertIsInstance(state_vector, np.ndarray)
        self.assertEqual(len(state_vector), 10)
        self.assertTrue(np.all(np.isfinite(state_vector)))
    
    def test_prepare_state_vector_with_missing_data(self):
        """Test state vector preparation with missing market data."""
        incomplete_data = {'close': 100.0}
        
        state_vector = self.signal_generator._prepare_state_vector(
            incomplete_data, 0.7, 0.3
        )
        
        self.assertIsInstance(state_vector, np.ndarray)
        self.assertEqual(len(state_vector), 10)
        self.assertTrue(np.all(np.isfinite(state_vector)))
    
    def test_calculate_confidence_score(self):
        """Test confidence score calculation."""
        confidence = self.signal_generator._calculate_confidence_score(
            lstm_prediction=0.8,
            dqn_q_values={'buy': 0.9, 'sell': 0.1, 'hold': 0.3},
            sentiment_score=0.5,
            market_data=self.sample_market_data
        )
        
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
    
    def test_calculate_confidence_score_extreme_rsi(self):
        """Test confidence score with extreme RSI values."""
        market_data_extreme_rsi = self.sample_market_data.copy()
        market_data_extreme_rsi['rsi_14'] = 25.0  # Oversold
        
        confidence = self.signal_generator._calculate_confidence_score(
            lstm_prediction=0.8,
            dqn_q_values={'buy': 0.9, 'sell': 0.1, 'hold': 0.3},
            sentiment_score=0.5,
            market_data=market_data_extreme_rsi
        )
        
        self.assertGreater(confidence, 0.0)
    
    def test_determine_action_buy_signal(self):
        """Test action determination for buy signal."""
        action = self.signal_generator._determine_action(
            lstm_prediction=0.8,  # Strong buy signal
            dqn_q_values={'buy': 0.9, 'sell': 0.1, 'hold': 0.3},
            sentiment_score=0.5  # Positive sentiment
        )
        
        self.assertEqual(action, 'buy')
    
    def test_determine_action_sell_signal(self):
        """Test action determination for sell signal."""
        action = self.signal_generator._determine_action(
            lstm_prediction=0.2,  # Strong sell signal
            dqn_q_values={'buy': 0.1, 'sell': 0.9, 'hold': 0.3},
            sentiment_score=-0.5  # Negative sentiment
        )
        
        self.assertEqual(action, 'sell')
    
    def test_determine_action_hold_signal(self):
        """Test action determination for hold signal."""
        action = self.signal_generator._determine_action(
            lstm_prediction=0.5,  # Neutral LSTM
            dqn_q_values={'buy': 0.3, 'sell': 0.3, 'hold': 0.8},
            sentiment_score=0.0  # Neutral sentiment
        )
        
        self.assertEqual(action, 'hold')
    
    def test_calculate_risk_adjusted_size(self):
        """Test risk-adjusted position size calculation."""
        size = self.signal_generator._calculate_risk_adjusted_size(
            confidence=0.8,
            sentiment_score=0.5,
            base_position_size=0.02
        )
        
        self.assertIsInstance(size, float)
        self.assertGreaterEqual(size, 0.005)  # Minimum bound
        self.assertLessEqual(size, 0.03)      # Maximum bound
        # With high confidence and positive sentiment, size should be adjusted upward
        self.assertGreaterEqual(size, 0.015)  # Should be reasonably sized
    
    def test_calculate_risk_adjusted_size_bounds(self):
        """Test that position size stays within bounds."""
        # Test minimum bound
        size_min = self.signal_generator._calculate_risk_adjusted_size(
            confidence=0.1,
            sentiment_score=0.0,
            base_position_size=0.001
        )
        self.assertGreaterEqual(size_min, 0.005)
        
        # Test maximum bound
        size_max = self.signal_generator._calculate_risk_adjusted_size(
            confidence=1.0,
            sentiment_score=1.0,
            base_position_size=0.05
        )
        self.assertLessEqual(size_max, 0.03)
    
    def test_generate_signal_success(self):
        """Test successful signal generation."""
        signal = self.signal_generator.generate_signal(
            symbol='RELIANCE',
            market_data=self.sample_market_data,
            lstm_prediction=0.8,
            dqn_q_values={'buy': 0.9, 'sell': 0.1, 'hold': 0.3},
            sentiment_score=0.5
        )
        
        self.assertIsInstance(signal, TradingSignal)
        self.assertEqual(signal.symbol, 'RELIANCE')
        self.assertIn(signal.action, ['buy', 'sell', 'hold'])
        self.assertGreaterEqual(signal.confidence, 0.0)
        self.assertLessEqual(signal.confidence, 1.0)
        self.assertEqual(signal.lstm_prediction, 0.8)
        self.assertEqual(signal.sentiment_score, 0.5)
        self.assertGreater(signal.risk_adjusted_size, 0.0)
    
    def test_generate_signal_with_error_handling(self):
        """Test signal generation with error handling."""
        # Test with invalid market data - the signal generation should still work
        # but may produce different results due to missing data
        signal = self.signal_generator.generate_signal(
            symbol='TEST',
            market_data={},  # Empty market data
            lstm_prediction=0.8,
            dqn_q_values={'buy': 0.9, 'sell': 0.1, 'hold': 0.3},
            sentiment_score=0.5
        )
        
        # Signal should be generated successfully even with empty market data
        self.assertIsInstance(signal, TradingSignal)
        self.assertEqual(signal.symbol, 'TEST')
        self.assertIn(signal.action, ['buy', 'sell', 'hold'])
        self.assertGreaterEqual(signal.confidence, 0.0)
        self.assertLessEqual(signal.confidence, 1.0)
    
    def test_validate_signal_valid(self):
        """Test validation of valid signal."""
        valid_signal = TradingSignal(
            symbol='RELIANCE',
            timestamp=datetime.now(),
            action='buy',
            confidence=0.8,
            lstm_prediction=0.7,
            dqn_q_values={'buy': 0.9, 'sell': 0.1, 'hold': 0.3},
            sentiment_score=0.5,
            risk_adjusted_size=0.02
        )
        
        self.assertTrue(self.signal_generator.validate_signal(valid_signal))
    
    def test_validate_signal_low_confidence(self):
        """Test validation rejection for low confidence."""
        low_confidence_signal = TradingSignal(
            symbol='RELIANCE',
            timestamp=datetime.now(),
            action='buy',
            confidence=0.4,  # Below threshold
            lstm_prediction=0.7,
            dqn_q_values={'buy': 0.9, 'sell': 0.1, 'hold': 0.3},
            sentiment_score=0.5,
            risk_adjusted_size=0.02
        )
        
        self.assertFalse(self.signal_generator.validate_signal(low_confidence_signal))
    
    def test_validate_signal_invalid_action(self):
        """Test validation rejection for invalid action."""
        invalid_action_signal = TradingSignal(
            symbol='RELIANCE',
            timestamp=datetime.now(),
            action='invalid_action',
            confidence=0.8,
            lstm_prediction=0.7,
            dqn_q_values={'buy': 0.9, 'sell': 0.1, 'hold': 0.3},
            sentiment_score=0.5,
            risk_adjusted_size=0.02
        )
        
        self.assertFalse(self.signal_generator.validate_signal(invalid_action_signal))
    
    def test_validate_signal_invalid_position_size(self):
        """Test validation rejection for invalid position size."""
        invalid_size_signal = TradingSignal(
            symbol='RELIANCE',
            timestamp=datetime.now(),
            action='buy',
            confidence=0.8,
            lstm_prediction=0.7,
            dqn_q_values={'buy': 0.9, 'sell': 0.1, 'hold': 0.3},
            sentiment_score=0.5,
            risk_adjusted_size=0.1  # Too large
        )
        
        self.assertFalse(self.signal_generator.validate_signal(invalid_size_signal))
    
    def test_validate_signal_invalid_lstm_prediction(self):
        """Test validation rejection for invalid LSTM prediction."""
        invalid_lstm_signal = TradingSignal(
            symbol='RELIANCE',
            timestamp=datetime.now(),
            action='buy',
            confidence=0.8,
            lstm_prediction=1.5,  # Out of bounds
            dqn_q_values={'buy': 0.9, 'sell': 0.1, 'hold': 0.3},
            sentiment_score=0.5,
            risk_adjusted_size=0.02
        )
        
        self.assertFalse(self.signal_generator.validate_signal(invalid_lstm_signal))
    
    def test_validate_signal_invalid_sentiment(self):
        """Test validation rejection for invalid sentiment score."""
        invalid_sentiment_signal = TradingSignal(
            symbol='RELIANCE',
            timestamp=datetime.now(),
            action='buy',
            confidence=0.8,
            lstm_prediction=0.7,
            dqn_q_values={'buy': 0.9, 'sell': 0.1, 'hold': 0.3},
            sentiment_score=2.0,  # Out of bounds
            risk_adjusted_size=0.02
        )
        
        self.assertFalse(self.signal_generator.validate_signal(invalid_sentiment_signal))
    
    def test_validate_signal_empty_symbol(self):
        """Test validation rejection for empty symbol."""
        empty_symbol_signal = TradingSignal(
            symbol='',  # Empty symbol
            timestamp=datetime.now(),
            action='buy',
            confidence=0.8,
            lstm_prediction=0.7,
            dqn_q_values={'buy': 0.9, 'sell': 0.1, 'hold': 0.3},
            sentiment_score=0.5,
            risk_adjusted_size=0.02
        )
        
        self.assertFalse(self.signal_generator.validate_signal(empty_symbol_signal))
    
    def test_validate_signal_hold_lower_threshold(self):
        """Test that hold signals have lower confidence requirements."""
        hold_signal = TradingSignal(
            symbol='RELIANCE',
            timestamp=datetime.now(),
            action='hold',
            confidence=0.65,  # Above min_confidence but below 0.7
            lstm_prediction=0.5,
            dqn_q_values={'buy': 0.3, 'sell': 0.3, 'hold': 0.8},
            sentiment_score=0.0,
            risk_adjusted_size=0.01
        )
        
        self.assertTrue(self.signal_generator.validate_signal(hold_signal))
    
    def test_validate_signal_buy_sell_higher_threshold(self):
        """Test that buy/sell signals require higher confidence."""
        buy_signal_low_conf = TradingSignal(
            symbol='RELIANCE',
            timestamp=datetime.now(),
            action='buy',
            confidence=0.65,  # Above min_confidence but below 0.7
            lstm_prediction=0.8,
            dqn_q_values={'buy': 0.9, 'sell': 0.1, 'hold': 0.3},
            sentiment_score=0.5,
            risk_adjusted_size=0.02
        )
        
        self.assertFalse(self.signal_generator.validate_signal(buy_signal_low_conf))
    
    def test_generate_signals_batch(self):
        """Test batch signal generation."""
        market_data_batch = {
            'RELIANCE': self.sample_market_data,
            'TCS': {
                'close': 200.0,
                'high': 205.0,
                'low': 195.0,
                'volume': 500000,
                'rsi_14': 45.0,
                'sma_50': 198.0,
                'bb_upper': 210.0,
                'bb_middle': 200.0,
                'bb_lower': 190.0,
                'sentiment_score': -0.2
            }
        }
        
        signals = self.signal_generator.generate_signals_batch(market_data_batch)
        
        self.assertIsInstance(signals, dict)
        self.assertLessEqual(len(signals), len(market_data_batch))
        
        for symbol, signal in signals.items():
            self.assertIsInstance(signal, TradingSignal)
            self.assertEqual(signal.symbol, symbol)
    
    def test_update_confidence_threshold(self):
        """Test updating confidence threshold."""
        # Valid threshold
        self.signal_generator.update_confidence_threshold(0.8)
        self.assertEqual(self.signal_generator.min_confidence, 0.8)
        
        # Invalid threshold (should not change)
        original_threshold = self.signal_generator.min_confidence
        self.signal_generator.update_confidence_threshold(1.5)
        self.assertEqual(self.signal_generator.min_confidence, original_threshold)
    
    def test_get_signal_statistics(self):
        """Test getting signal statistics."""
        stats = self.signal_generator.get_signal_statistics()
        
        self.assertIsInstance(stats, dict)
        self.assertIn('min_confidence', stats)
        self.assertIn('weights', stats)
        self.assertIn('models_available', stats)
        
        self.assertEqual(stats['min_confidence'], self.signal_generator.min_confidence)
        self.assertIn('sentiment', stats['weights'])
        self.assertIn('lstm', stats['weights'])
        self.assertIn('dqn', stats['weights'])
    
    def test_signal_generation_without_models(self):
        """Test signal generation without trained models."""
        sg_no_models = SignalGenerator()
        
        signal = sg_no_models.generate_signal(
            symbol='TEST',
            market_data=self.sample_market_data,
            lstm_prediction=0.5,
            dqn_q_values={'buy': 0.0, 'sell': 0.0, 'hold': 0.0},
            sentiment_score=0.0
        )
        
        self.assertIsInstance(signal, TradingSignal)
        self.assertEqual(signal.symbol, 'TEST')
    
    def test_confidence_calculation_edge_cases(self):
        """Test confidence calculation with edge cases."""
        # Test with identical Q-values
        confidence = self.signal_generator._calculate_confidence_score(
            lstm_prediction=0.5,
            dqn_q_values={'buy': 0.5, 'sell': 0.5, 'hold': 0.5},
            sentiment_score=0.0,
            market_data=self.sample_market_data
        )
        
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
        
        # Test with extreme values
        confidence_extreme = self.signal_generator._calculate_confidence_score(
            lstm_prediction=1.0,
            dqn_q_values={'buy': 1.0, 'sell': -1.0, 'hold': 0.0},
            sentiment_score=1.0,
            market_data=self.sample_market_data
        )
        
        self.assertGreaterEqual(confidence_extreme, 0.0)
        self.assertLessEqual(confidence_extreme, 1.0)


if __name__ == '__main__':
    unittest.main()