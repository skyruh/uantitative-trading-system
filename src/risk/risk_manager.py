"""
Main risk management module that integrates position sizing and stop-loss management.
Implements the IRiskManager interface for comprehensive risk control.
"""

import logging
from typing import List, Optional, Dict, Tuple
from datetime import datetime

from ..interfaces.trading_interfaces import IRiskManager, TradingSignal, Position
from .position_sizer import PositionSizer, PositionSizingConfig
from .stop_loss_manager import StopLossManager, StopLossConfig
from .portfolio_monitor import PortfolioMonitor, PortfolioConfig


class RiskManager(IRiskManager):
    """
    Main risk management class that coordinates position sizing and stop-loss management.
    
    Implements comprehensive risk controls:
    - Position sizing (1-2% of capital)
    - Stop-loss management (5% below entry)
    - Automatic stop-loss execution
    """
    
    def __init__(self, 
                 position_config: Optional[PositionSizingConfig] = None,
                 stop_loss_config: Optional[StopLossConfig] = None,
                 portfolio_config: Optional[PortfolioConfig] = None):
        """
        Initialize risk manager with configuration.
        
        Args:
            position_config: Position sizing configuration
            stop_loss_config: Stop-loss configuration
            portfolio_config: Portfolio diversification configuration
        """
        self.position_sizer = PositionSizer(position_config)
        self.stop_loss_manager = StopLossManager(stop_loss_config)
        self.portfolio_monitor = PortfolioMonitor(portfolio_config)
        self.logger = logging.getLogger(__name__)
        
    def calculate_position_size(self, signal: TradingSignal, portfolio_value: float) -> float:
        """
        Calculate appropriate position size based on risk rules.
        
        Args:
            signal: Trading signal containing confidence and sentiment
            portfolio_value: Current total portfolio value
            
        Returns:
            Position size as percentage of portfolio value
            
        Requirements: 7.1, 7.2 - Position sizing 1-2% of capital with sentiment adjustments
        """
        try:
            # We need current price - for now use a placeholder
            # In real implementation, this would come from market data
            current_price = 100.0  # This should be passed as parameter or fetched
            
            position_size = self.position_sizer.calculate_position_size(
                signal, portfolio_value, current_price
            )
            
            self.logger.info(f"Calculated position size for {signal.symbol}: {position_size:.4f}")
            
            return position_size
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return 0.01  # Default to 1% if calculation fails
    
    def set_stop_loss(self, position: Position) -> float:
        """
        Set stop-loss price for a position.
        
        Args:
            position: Position to set stop-loss for
            
        Returns:
            Stop-loss price
            
        Requirements: 7.1 - Set stop-loss at 5% below entry price
        """
        return self.stop_loss_manager.set_stop_loss(position)
    
    def check_portfolio_limits(self, new_position: Position, 
                              current_positions: List[Position]) -> bool:
        """
        Check if new position violates portfolio diversification limits.
        
        Args:
            new_position: New position to validate
            current_positions: List of current open positions
            
        Returns:
            True if position is within limits, False otherwise
            
        Requirements: 7.3, 7.4 - Portfolio diversification limits
        """
        try:
            # Create a trading signal from the position for portfolio monitor
            temp_signal = TradingSignal(
                symbol=new_position.symbol,
                timestamp=new_position.entry_date,
                action="buy",
                confidence=0.8,  # Default confidence
                lstm_prediction=0.0,
                dqn_q_values={"buy": 0.5, "sell": 0.3, "hold": 0.2},
                sentiment_score=0.0,  # Neutral sentiment
                risk_adjusted_size=new_position.current_value / 1000000.0  # Assume 10L portfolio
            )
            
            # Use portfolio monitor to check limits
            is_allowed, reason = self.portfolio_monitor.check_position_limits(
                temp_signal, current_positions, 1000000.0  # Placeholder portfolio value
            )
            
            if not is_allowed:
                self.logger.warning(f"Portfolio limit check failed: {reason}")
            
            return is_allowed
            
        except Exception as e:
            self.logger.error(f"Error checking portfolio limits: {e}")
            return False
    
    def should_close_position(self, position: Position, current_price: float) -> bool:
        """
        Determine if position should be closed based on risk rules.
        
        Args:
            position: Position to evaluate
            current_price: Current market price
            
        Returns:
            True if position should be closed, False otherwise
            
        Requirements: 7.5 - Automatic stop-loss execution when triggered
        """
        return self.stop_loss_manager.should_execute_stop_loss(position, current_price)
    
    def execute_stop_loss(self, position: Position, current_price: float) -> Tuple[bool, str]:
        """
        Execute stop-loss for a position.
        
        Args:
            position: Position to close
            current_price: Current market price
            
        Returns:
            Tuple of (success, message)
        """
        return self.stop_loss_manager.execute_stop_loss(position, current_price)
    
    def validate_trade(self, signal: TradingSignal, portfolio_value: float, 
                      current_positions: List[Position]) -> Tuple[bool, str]:
        """
        Comprehensive trade validation before execution.
        
        Args:
            signal: Trading signal to validate
            portfolio_value: Current portfolio value
            current_positions: Current open positions
            
        Returns:
            Tuple of (is_valid, reason)
        """
        try:
            # Check if signal action is valid
            if signal.action not in ['buy', 'sell', 'hold']:
                return False, f"Invalid signal action: {signal.action}"
            
            # Skip validation for hold signals
            if signal.action == 'hold':
                return True, "Hold signal - no validation needed"
            
            # For buy signals, check position sizing
            if signal.action == 'buy':
                position_size = self.calculate_position_size(signal, portfolio_value)
                
                if not self.position_sizer.validate_position_size(position_size, portfolio_value):
                    return False, "Position size validation failed"
                
                # Create temporary position for portfolio limit check
                temp_position = Position(
                    symbol=signal.symbol,
                    entry_date=signal.timestamp,
                    entry_price=100.0,  # Placeholder - should be current price
                    quantity=int((position_size * portfolio_value) / 100.0),
                    stop_loss_price=95.0,  # Placeholder
                    current_value=position_size * portfolio_value,
                    unrealized_pnl=0.0,
                    status='open'
                )
                
                if not self.check_portfolio_limits(temp_position, current_positions):
                    return False, "Portfolio diversification limits exceeded"
            
            return True, "Trade validation passed"
            
        except Exception as e:
            self.logger.error(f"Error validating trade: {e}")
            return False, f"Validation error: {e}"
    
    def get_risk_metrics(self, positions: List[Position], portfolio_value: float) -> Dict[str, float]:
        """
        Calculate current risk metrics for the portfolio.
        
        Args:
            positions: List of current positions
            portfolio_value: Current portfolio value
            
        Returns:
            Dictionary of risk metrics
        """
        try:
            if not positions:
                return {
                    'total_exposure': 0.0,
                    'number_of_positions': 0,
                    'average_position_size': 0.0,
                    'largest_position_pct': 0.0,
                    'stop_loss_coverage': 0.0
                }
            
            # Calculate exposure metrics
            total_position_value = sum(pos.current_value for pos in positions)
            total_exposure = total_position_value / portfolio_value
            
            position_sizes = [pos.current_value / portfolio_value for pos in positions]
            average_position_size = sum(position_sizes) / len(position_sizes)
            largest_position_pct = max(position_sizes) if position_sizes else 0.0
            
            # Calculate stop-loss coverage
            positions_with_stop_loss = sum(1 for pos in positions 
                                         if self.stop_loss_manager.get_stop_loss_price(pos) is not None)
            stop_loss_coverage = positions_with_stop_loss / len(positions) if positions else 0.0
            
            return {
                'total_exposure': total_exposure,
                'number_of_positions': len(positions),
                'average_position_size': average_position_size,
                'largest_position_pct': largest_position_pct,
                'stop_loss_coverage': stop_loss_coverage
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating risk metrics: {e}")
            return {}
    
    def get_position_sizer(self) -> PositionSizer:
        """Get the position sizer instance."""
        return self.position_sizer
    
    def get_stop_loss_manager(self) -> StopLossManager:
        """Get the stop-loss manager instance."""
        return self.stop_loss_manager
    
    def get_portfolio_monitor(self) -> PortfolioMonitor:
        """Get the portfolio monitor instance."""
        return self.portfolio_monitor
    
    def get_portfolio_metrics(self, positions: List[Position], portfolio_value: float):
        """Get comprehensive portfolio diversification metrics."""
        return self.portfolio_monitor.get_portfolio_metrics(positions, portfolio_value)
    
    def suggest_rebalancing(self, positions: List[Position], portfolio_value: float) -> List[str]:
        """Get portfolio rebalancing suggestions."""
        return self.portfolio_monitor.suggest_rebalancing(positions, portfolio_value)