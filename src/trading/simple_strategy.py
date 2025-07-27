"""
Simple Trading Strategy implementation for the quantitative trading system.
Combines LSTM predictions with DQN actions to make trading decisions.
"""

import logging
from typing import Dict, List
from datetime import datetime

from src.interfaces.trading_interfaces import ITradingStrategy, TradingSignal, Position
from src.trading.signal_generator import SignalGenerator
from src.trading.position_manager import PositionManager


class SimpleTradingStrategy(ITradingStrategy):
    """
    Simple trading strategy that uses LSTM predictions and DQN actions.
    
    This strategy:
    1. Generates signals using the SignalGenerator
    2. Manages positions using the PositionManager
    3. Implements basic risk management rules
    """
    
    def __init__(self, signal_generator: SignalGenerator, position_manager: PositionManager):
        """
        Initialize the trading strategy.
        
        Args:
            signal_generator: Signal generator instance
            position_manager: Position manager instance
        """
        self.signal_generator = signal_generator
        self.position_manager = position_manager
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("SimpleTradingStrategy initialized")
    
    def execute_strategy(self, market_data: Dict) -> List[TradingSignal]:
        """
        Execute trading strategy and generate signals.
        
        Args:
            market_data: Dictionary containing market data for all symbols
            
        Returns:
            List of trading signals
        """
        try:
            self.logger.debug(f"Executing strategy for {len(market_data)} symbols")
            
            # Generate signals for all symbols
            signals_dict = self.signal_generator.generate_signals_batch(market_data)
            
            # Convert to list and filter based on strategy rules
            signals = []
            for symbol, signal in signals_dict.items():
                # Apply strategy-specific filtering
                if self._should_execute_signal(signal):
                    signals.append(signal)
                else:
                    self.logger.debug(f"Signal for {symbol} filtered out by strategy rules")
            
            self.logger.info(f"Strategy generated {len(signals)} valid signals")
            return signals
            
        except Exception as e:
            self.logger.error(f"Error executing strategy: {str(e)}")
            return []
    
    def process_signals(self, signals: List[TradingSignal]) -> List[Position]:
        """
        Process trading signals and manage positions.
        
        Args:
            signals: List of trading signals to process
            
        Returns:
            List of positions created or modified
        """
        try:
            positions = []
            
            for signal in signals:
                position = self._process_single_signal(signal)
                if position:
                    positions.append(position)
            
            self.logger.info(f"Processed {len(signals)} signals, created/modified {len(positions)} positions")
            return positions
            
        except Exception as e:
            self.logger.error(f"Error processing signals: {str(e)}")
            return []
    
    def _should_execute_signal(self, signal: TradingSignal) -> bool:
        """
        Apply strategy-specific rules to determine if signal should be executed.
        
        Args:
            signal: Trading signal to evaluate
            
        Returns:
            True if signal should be executed, False otherwise
        """
        try:
            # Don't execute hold signals
            if signal.action == 'hold':
                return False
            
            # Check if we already have a position in this symbol
            if self.position_manager.has_position(signal.symbol):
                existing_position = self.position_manager.get_position_by_symbol(signal.symbol)
                
                # If we have a position and signal is to buy more, skip (no pyramiding)
                if signal.action == 'buy':
                    self.logger.debug(f"Skipping buy signal for {signal.symbol} - already have position")
                    return False
                
                # If we have a position and signal is to sell, allow it (exit signal)
                if signal.action == 'sell':
                    return True
            
            # For new positions, check available capital
            if signal.action == 'buy':
                available_capital = self.position_manager.get_available_capital()
                required_capital = available_capital * signal.risk_adjusted_size
                
                if required_capital < 1000:  # Minimum position size
                    self.logger.debug(f"Skipping buy signal for {signal.symbol} - insufficient capital")
                    return False
            
            # Check maximum number of positions (risk management)
            if signal.action == 'buy' and self.position_manager.get_position_count() >= 10:
                self.logger.debug(f"Skipping buy signal for {signal.symbol} - max positions reached")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error evaluating signal for {signal.symbol}: {str(e)}")
            return False
    
    def _process_single_signal(self, signal: TradingSignal) -> Position:
        """
        Process a single trading signal.
        
        Args:
            signal: Trading signal to process
            
        Returns:
            Position object if created/modified, None otherwise
        """
        try:
            if signal.action == 'buy':
                # Open new position
                position = self.position_manager.open_position(signal, signal.risk_adjusted_size)
                self.logger.info(f"Opened new position for {signal.symbol}")
                return position
                
            elif signal.action == 'sell':
                # Close existing position
                existing_position = self.position_manager.get_position_by_symbol(signal.symbol)
                if existing_position:
                    # Use a placeholder close price - in real implementation this would be current market price
                    close_price = existing_position.entry_price * 1.02  # Assume 2% gain for demo
                    closed_position = self.position_manager.close_position(existing_position, close_price)
                    self.logger.info(f"Closed position for {signal.symbol}")
                    return closed_position
                else:
                    self.logger.warning(f"Sell signal for {signal.symbol} but no existing position")
                    return None
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error processing signal for {signal.symbol}: {str(e)}")
            return None
    
    def get_strategy_statistics(self) -> Dict:
        """
        Get strategy performance statistics.
        
        Returns:
            Dictionary with strategy statistics
        """
        try:
            position_stats = self.position_manager.get_position_statistics()
            signal_stats = self.signal_generator.get_signal_statistics()
            
            return {
                'strategy_name': 'SimpleTradingStrategy',
                'position_stats': position_stats,
                'signal_stats': signal_stats,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting strategy statistics: {str(e)}")
            return {}