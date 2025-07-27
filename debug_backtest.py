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
from src.trading.simple_strategy import SimpleTradingStrategy
from src.config.settings import config

def main():
    """Test backtest signal processing"""
    
    # Enable short selling for testing
    config.backtest.allow_short_selling = True
    
    # Create backtest engine (it will use the global config)
    engine = BacktestEngine()
    
    # Initialize backtest
    if not engine.initialize_backtest(1000000.0, "2024-01-01", "2024-12-31"):
        print("Failed to initialize backtest")
        return
    
    # Load market data from processed files
    import os
    processed_dir = "data/processed"
    market_data = {}
    
    if os.path.exists(processed_dir):
        processed_files = [f for f in os.listdir(processed_dir) if f.endswith('_processed.csv')]
        print(f"Found {len(processed_files)} processed files")
        
        for file in processed_files:
            symbol = file.replace('_processed.csv', '') + '.NS'
            file_path = os.path.join(processed_dir, file)
            
            try:
                data = pd.read_csv(file_path)
                
                # Convert Date column to datetime index if it exists
                if 'Date' in data.columns:
                    data['Date'] = pd.to_datetime(data['Date'])
                    # Convert to timezone-naive to avoid comparison issues
                    if data['Date'].dt.tz is not None:
                        data['Date'] = data['Date'].dt.tz_convert(None)
                    data.set_index('Date', inplace=True)
                elif 'date' in data.columns:
                    data['date'] = pd.to_datetime(data['date'])
                    # Convert to timezone-naive to avoid comparison issues
                    if data['date'].dt.tz is not None:
                        data['date'] = data['date'].dt.tz_convert(None)
                    data.set_index('date', inplace=True)
                
                # Ensure required columns exist (map lowercase to uppercase if needed)
                required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                for col in required_columns:
                    if col not in data.columns and col.lower() in data.columns:
                        data[col] = data[col.lower()]
                
                # Check if all required columns are present
                if all(col in data.columns for col in required_columns):
                    market_data[symbol] = data
                    print(f"Loaded {symbol}: {len(data)} rows")
                else:
                    print(f"Skipping {symbol}: missing required columns")
                    
            except Exception as e:
                print(f"Error loading {symbol}: {e}")
    
    if not market_data:
        print("No market data could be loaded")
        return
    
    # Load market data into engine
    if not engine.load_market_data(market_data):
        print("Failed to load market data into engine")
        return
    
    print(f"Market data loaded for {len(market_data)} symbols")
    
    # Create signal generator and strategy
    from src.trading.signal_generator import SignalGenerator
    signal_generator = SignalGenerator()
    
    # Use the position manager from the backtest engine
    strategy = SimpleTradingStrategy(signal_generator, engine.position_manager, 
                                   allow_short_selling=config.backtest.allow_short_selling)
    
    # Test signal generation for a specific date
    test_date = datetime(2025, 7, 15)  # Use a recent date with available data
    engine.current_date = test_date
    
    # Get market data for test date
    daily_market_data = engine._get_market_data_for_date(test_date)
    print(f"Market data available for {len(daily_market_data)} symbols on {test_date}")
    
    if daily_market_data:
        # Generate signals
        print("Calling strategy.execute_strategy...")
        signals = strategy.execute_strategy(daily_market_data)
        print(f"Generated {len(signals)} signals")
        
        # Also test direct signal generation
        print("\nTesting direct signal generation...")
        direct_signals = signal_generator.generate_signals_batch(daily_market_data)
        print(f"Direct signal generation returned {len(direct_signals)} signals")
        
        for signal in signals:
            print(f"Signal: {signal.symbol} - {signal.action} (confidence: {signal.confidence:.3f})")
        
        # Also show direct signals for comparison
        print("\nDirect signals from generator:")
        for symbol, signal in direct_signals.items():
            print(f"Direct Signal: {symbol} - {signal.action} (confidence: {signal.confidence:.3f})")
            
            # Test if the strategy would execute this signal
            should_execute = strategy._should_execute_signal(signal)
            print(f"Strategy would execute: {should_execute}")
        
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
                        position_size = engine.risk_manager.calculate_position_size(signal, portfolio_value)
                        print(f"  Position size: {position_size:.4f} ({position_size*100:.2f}%)")
                        
                        # Calculate actual investment amount
                        investment_amount = position_size * portfolio_value
                        shares = int(investment_amount / current_price)
                        print(f"  Investment amount: ₹{investment_amount:,.2f}")
                        print(f"  Shares to buy: {shares}")
    
    else:
        print("No market data available for test date")

if __name__ == "__main__":
    main()