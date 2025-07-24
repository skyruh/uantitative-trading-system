"""
Stop-loss management module for risk management.
Implements stop-loss logic to set stop-loss at 5% below entry price and automatic execution.
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from ..interfaces.trading_interfaces import Position


@dataclass
class StopLossConfig:
    """Configuration for stop-loss parameters."""
    stop_loss_pct: float = 0.05  # 5% stop-loss below entry price
    trailing_stop_enabled: bool = False  # Enable trailing stop-loss
    trailing_stop_pct: float = 0.03  # 3% trailing stop distance
    min_profit_for_trailing: float = 0.02  # 2% minimum profit before trailing starts


@dataclass
class StopLossOrder:
    """Stop-loss order data model."""
    position_id: str
    symbol: str
    stop_price: float
    original_stop_price: float
    is_trailing: bool
    created_at: datetime
    last_updated: datetime


class StopLossManager:
    """
    Handles stop-loss management and automatic execution.
    
    Implements:
    - 5% stop-loss below entry price
    - Automatic stop-loss execution when triggered
    - Optional trailing stop-loss functionality
    """
    
    def __init__(self, config: Optional[StopLossConfig] = None):
        """
        Initialize stop-loss manager with configuration.
        
        Args:
            config: Stop-loss configuration parameters
        """
        self.config = config or StopLossConfig()
        self.logger = logging.getLogger(__name__)
        self.stop_loss_orders: Dict[str, StopLossOrder] = {}
        
    def set_stop_loss(self, position: Position) -> float:
        """
        Set stop-loss price for a position.
        
        Args:
            position: Position to set stop-loss for
            
        Returns:
            Stop-loss price
            
        Requirements: 7.1 - Set stop-loss at 5% below entry price
        """
        try:
            # Calculate stop-loss price (5% below entry price)
            stop_price = position.entry_price * (1 - self.config.stop_loss_pct)
            
            # Create stop-loss order
            order_id = f"{position.symbol}_{position.entry_date.strftime('%Y%m%d_%H%M%S')}"
            stop_order = StopLossOrder(
                position_id=order_id,
                symbol=position.symbol,
                stop_price=stop_price,
                original_stop_price=stop_price,
                is_trailing=False,
                created_at=datetime.now(),
                last_updated=datetime.now()
            )
            
            self.stop_loss_orders[order_id] = stop_order
            
            self.logger.info(f"Set stop-loss for {position.symbol} at {stop_price:.2f} "
                           f"(entry: {position.entry_price:.2f}, -{self.config.stop_loss_pct*100:.1f}%)")
            
            return stop_price
            
        except Exception as e:
            self.logger.error(f"Error setting stop-loss for {position.symbol}: {e}")
            # Return a default stop-loss price if entry_price is invalid
            if position.entry_price is not None:
                return position.entry_price * (1 - self.config.stop_loss_pct)
            else:
                return 0.0  # Default fallback value
    
    def should_execute_stop_loss(self, position: Position, current_price: float) -> bool:
        """
        Check if stop-loss should be executed for a position.
        
        Args:
            position: Position to check
            current_price: Current market price
            
        Returns:
            True if stop-loss should be executed, False otherwise
            
        Requirements: 7.5 - Automatic stop-loss execution when triggered
        """
        try:
            order_id = f"{position.symbol}_{position.entry_date.strftime('%Y%m%d_%H%M%S')}"
            
            if order_id not in self.stop_loss_orders:
                self.logger.warning(f"No stop-loss order found for position {position.symbol}")
                return False
            
            stop_order = self.stop_loss_orders[order_id]
            
            # Update trailing stop if enabled and conditions are met
            if self.config.trailing_stop_enabled:
                self._update_trailing_stop(position, current_price, stop_order)
            
            # Check if current price has hit stop-loss
            should_execute = current_price <= stop_order.stop_price
            
            if should_execute:
                self.logger.warning(f"Stop-loss triggered for {position.symbol}: "
                                  f"current_price={current_price:.2f}, stop_price={stop_order.stop_price:.2f}")
            
            return should_execute
            
        except Exception as e:
            self.logger.error(f"Error checking stop-loss for {position.symbol}: {e}")
            return False
    
    def execute_stop_loss(self, position: Position, current_price: float) -> Tuple[bool, str]:
        """
        Execute stop-loss for a position.
        
        Args:
            position: Position to close
            current_price: Current market price
            
        Returns:
            Tuple of (success, message)
        """
        try:
            order_id = f"{position.symbol}_{position.entry_date.strftime('%Y%m%d_%H%M%S')}"
            
            if order_id not in self.stop_loss_orders:
                return False, f"No stop-loss order found for {position.symbol}"
            
            stop_order = self.stop_loss_orders[order_id]
            
            # Calculate loss
            loss_amount = (position.entry_price - current_price) * position.quantity
            loss_pct = (position.entry_price - current_price) / position.entry_price * 100
            
            # Remove stop-loss order
            del self.stop_loss_orders[order_id]
            
            message = (f"Stop-loss executed for {position.symbol}: "
                      f"entry={position.entry_price:.2f}, exit={current_price:.2f}, "
                      f"loss={loss_amount:.2f} INR ({loss_pct:.2f}%)")
            
            self.logger.info(message)
            
            return True, message
            
        except Exception as e:
            error_msg = f"Error executing stop-loss for {position.symbol}: {e}"
            self.logger.error(error_msg)
            return False, error_msg
    
    def _update_trailing_stop(self, position: Position, current_price: float, 
                            stop_order: StopLossOrder) -> None:
        """
        Update trailing stop-loss if conditions are met.
        
        Args:
            position: Current position
            current_price: Current market price
            stop_order: Stop-loss order to update
        """
        try:
            # Check if position has minimum profit for trailing
            current_profit_pct = (current_price - position.entry_price) / position.entry_price
            
            if current_profit_pct < self.config.min_profit_for_trailing:
                return
            
            # Calculate new trailing stop price
            new_stop_price = current_price * (1 - self.config.trailing_stop_pct)
            
            # Only update if new stop price is higher than current
            if new_stop_price > stop_order.stop_price:
                old_stop_price = stop_order.stop_price
                stop_order.stop_price = new_stop_price
                stop_order.is_trailing = True
                stop_order.last_updated = datetime.now()
                
                self.logger.info(f"Updated trailing stop for {position.symbol}: "
                               f"{old_stop_price:.2f} -> {new_stop_price:.2f}")
                
        except Exception as e:
            self.logger.error(f"Error updating trailing stop for {position.symbol}: {e}")
    
    def get_stop_loss_price(self, position: Position) -> Optional[float]:
        """
        Get current stop-loss price for a position.
        
        Args:
            position: Position to get stop-loss price for
            
        Returns:
            Current stop-loss price or None if not found
        """
        try:
            order_id = f"{position.symbol}_{position.entry_date.strftime('%Y%m%d_%H%M%S')}"
            
            if order_id in self.stop_loss_orders:
                return self.stop_loss_orders[order_id].stop_price
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting stop-loss price for {position.symbol}: {e}")
            return None
    
    def remove_stop_loss(self, position: Position) -> bool:
        """
        Remove stop-loss order for a position.
        
        Args:
            position: Position to remove stop-loss for
            
        Returns:
            True if removed successfully, False otherwise
        """
        try:
            order_id = f"{position.symbol}_{position.entry_date.strftime('%Y%m%d_%H%M%S')}"
            
            if order_id in self.stop_loss_orders:
                del self.stop_loss_orders[order_id]
                self.logger.info(f"Removed stop-loss order for {position.symbol}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error removing stop-loss for {position.symbol}: {e}")
            return False
    
    def get_all_stop_losses(self) -> Dict[str, StopLossOrder]:
        """
        Get all active stop-loss orders.
        
        Returns:
            Dictionary of all stop-loss orders
        """
        return self.stop_loss_orders.copy()
    
    def get_stop_loss_summary(self) -> Dict[str, int]:
        """
        Get summary of stop-loss orders.
        
        Returns:
            Summary statistics of stop-loss orders
        """
        total_orders = len(self.stop_loss_orders)
        trailing_orders = sum(1 for order in self.stop_loss_orders.values() if order.is_trailing)
        
        return {
            'total_orders': total_orders,
            'trailing_orders': trailing_orders,
            'fixed_orders': total_orders - trailing_orders
        }