#!/usr/bin/env python3
"""
Debug script to test signal generation
"""

import sys
import os
sys.path.append('src')

from trading.signal_generator import SignalGenerator
from models.lstm_model import LSTMModel
import pandas as pd
import numpy as np

def test_signal_generation():
    """Test signal generation with sample data"""
    
    # Create signal generator
    signal_gen = SignalGenerator(min_confidence=0.1)
    print(f"LSTM model: {signal_gen.lstm_model}")
    print(f"DQN agent: {signal_gen.dqn_agent}")
    
    # Create sample market data
    sample_data = {
        'RELIANCE.NS': {
            'Close': 2500.0,
            'Open': 2480.0,
            'High': 2520.0,
            'Low': 2470.0,
            'Volume': 1000000,
            'rsi_14': 45.0,
            'sma_50': 2450.0
        },
        'TCS.NS': {
            'Close': 3200.0,
            'Open': 3180.0,
            'High': 3220.0,
            'Low': 3170.0,
            'Volume': 800000,
            'rsi_14': 65.0,
            'sma_50': 3150.0
        }
    }
    
    print("Testing signal generation...")
    print(f"Signal generator min_confidence: {signal_gen.min_confidence}")
    
    # Generate signals
    signals = signal_gen.generate_signals_batch(sample_data)
    
    print(f"\nGenerated {len(signals)} signals:")
    for symbol, signal in signals.items():
        print(f"{symbol}: {signal.action} (confidence: {signal.confidence:.3f}, "
              f"lstm_pred: {signal.lstm_prediction:.3f})")
    
    if not signals:
        print("No signals generated - debugging individual components...")
        
        # Test individual components
        for symbol, market_data in sample_data.items():
            print(f"\nDebugging {symbol}:")
            
            # Test LSTM prediction directly
            print(f"  Market data: {market_data}")
            lstm_pred = signal_gen._get_lstm_prediction(symbol, market_data)
            print(f"  LSTM prediction: {lstm_pred:.3f}")
            
            # Force the signal generator to use the LSTM prediction
            signal_gen.lstm_model = type('MockModel', (), {'is_trained': True})()  # Mock trained model
            
            # Test Q-values
            q_values = signal_gen._get_default_q_values(lstm_pred)
            print(f"  Q-values: {q_values}")
            
            # Test confidence
            confidence = signal_gen._calculate_confidence_score(lstm_pred, q_values, market_data)
            print(f"  Confidence: {confidence:.3f}")
            
            # Test action
            action = signal_gen._determine_action(lstm_pred, q_values)
            print(f"  Action: {action}")
            
            # Test validation
            try:
                signal = signal_gen.generate_signal(symbol, market_data, lstm_pred, q_values)
                is_valid = signal_gen.validate_signal(signal)
                print(f"  Signal valid: {is_valid}")
                if not is_valid:
                    print(f"    Signal details: action={signal.action}, confidence={signal.confidence:.3f}")
            except Exception as e:
                print(f"  Error generating signal: {e}")

if __name__ == "__main__":
    test_signal_generation()