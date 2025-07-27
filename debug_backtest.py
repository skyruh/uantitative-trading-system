#!/usr/bin/env python3
"""
Debug script to test backtest signal processing
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime, timedelta
import pandas as pd
from src.backtesting.backtest_engine import BacktestEngine
from src.trading.simple_strategy import SimpleStrategy
from src.config.settings import Settings

def main():
    """Test backtest signal processing"""
    
    # Initialize settings
    settings = Settings()
    
    # Create backtest engine
    engine = BacktestEngine(settings)
    
    # Initialize backtest
    if not engine.initialize_backtest():
        print("Failed to initialize backtest")
        return
    
    # Load market data
    if not engine.load_market_data():
        print("Failed to load market data")
        return
    
    print(f"Market data loaded for {len(engine.market_data)} symbols")
    
    # Create strategy
    strategy = SimpleStrategy(settings)
    
    # Test signal generation for a specific date
    test_date = datetime(2024, 6, 1)  # Use a date in the middle of 2024
    engine.current_date = test_date
    
    # Get market data for test date
    daily_market_data = engine._get_market_data_for_date(test_date)
    print(f"Market data available for {len(daily_market_data)} symbols on {test_date}")
    
    if daily_market_data:
        # Generate signals
        signals = strategy.execute_strategy(daily_market_data)
        print(f"Generated {len(signals)} signals")
        
        for signal in signals:
            print(f"Signal: {signal.symbol} - {signal.action} (confidence: {signal.confidence:.3f})")
        
        # Test signal processing
        if signals:
            print("\nTesting signal processing...")
            
            # Get portfolio value
            portfolio_value = engine.position_manager.get_portfolio_value()
            print(f"Portfolio value: ₹{portfolio_value:,.2f}")
            
            # Get current positions
            current_positions = engine.position_manager.get_open_positions()
            print(f"Current positions: {len(current_positions)}")
            
            # Test each signal
            for signal in signals:
                if signal.symbol in daily_market_data:
                    current_price = daily_market_data[signal.symbol]['Close']
                    print(f"\nTesting signal for {signal.symbol}:")
                    print(f"  Action: {signal.action}")
                    print(f"  Current price: ₹{current_price:.2f}")
                    print(f"  Confidence: {signal.confidence:.3f}")
                    
                    # Test risk manager validation
                    is_valid, reason = engine.risk_manager.validate_trade(
                        signal, portfolio_value, current_positions
                    )
                    print(f"  Risk validation: {is_valid} - {reason}")
                    
                    if is_valid and signal.action == 'buy':
                        # Test position sizing
                        position_size = engine.risk_manager.calculat