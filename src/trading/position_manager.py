"""
Position Manager for the quantitative trading system.
Manages opening, monitoring, and closing of trading positions.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional, Any
import logging

from src.interfaces.trading_interfaces import IPositionManager, Position, TradingSignal


class PositionManager(IPositionManager):
    """
    Manages trading positions throughout their lifecycle.
    
    Features:
    - Opens new positions based on trading signals
    - Tracks position values and P&L in real-time
    - Closes positions based on stop-loss or exit signals
    - Maintains position history and statistics
    """
    
    def __init__(self, initial_capital: float = 1000000.0):
        """
        Initialize position manager.
        
        Args:
            initial_capital: Initial capital amount in rupees
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.open_positions: Dict[str, Position] = {}
        self.closed_positions: List[Position] = []
        self.position_history: List[Dict] = []
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"PositionManager initialized with capital ₹{initial_capital:,.2f}")
    
    def open_position(self, signal: TradingSignal, position_size: float) -> Position:
        """
        Open a new trading position.
        
        Args:
            signal: Trading signal containing entry information
            position_size: Position size as fraction of capital
            
        Returns:
            Created position object
        """
        try:
            # Calculate position value and quantity
            position_value = self.current_capital * position_size
            
            # Get current price from signal (assuming we have market data)
            # In practice, this would come from real-time market data
            entry_price = 100.0  # Placeholder - would be actual market price
            
            # Calculate quantity (assuming we can buy fractional shares)
            quantity = int(position_value / entry_price)
            actual_position_value = quantity * entry_price
            
            # Calculate stop-loss price (5% below entry as per requirements)
            stop_loss_price = entry_price * 0.95
            
            # Create position
            position = Position(
                symbol=signal.symbol,
                entry_date=signal.timestamp,
                entry_price=entry_price,
                quantity=quantity,
                stop_loss_price=stop_loss_price,
                current_value=actual_position_value,
                unrealized_pnl=0.0,
                status='open'
            )
            
            # Update capital and add to open positions
            self.current_capital -= actual_position_value
            self.open_positions[signal.symbol] = position
            
            # Log position opening
            self._log_position_event('OPEN', position, signal)
            
            self.logger.info(f"Opened position: {signal.symbol} - {quantity} shares at ₹{entry_price:.2f}")
            
            return position
            
        except Exception as e:
            self.logger.error(f"Error opening position for {signal.symbol}: {str(e)}")
            # Return a dummy position to maintain interface contract
            return Position(
                symbol=signal.symbol,
                entry_date=datetime.now(),
                entry_price=0.0,
                quantity=0,
                stop_loss_price=0.0,
                current_value=0.0,
                unrealized_pnl=0.0,
                status='error'
            )
    
    def close_position(self, position: Position, close_price: float) -> Position:
        """
        Close an existing position.
        
        Args:
            position: Position to close
            close_price: Current market price for closing
            
        Returns:
            Updated position object with final P&L
        """
        try:
            if position.status != 'open':
                self.logger.warning(f"Attempting to close non-open position: {position.symbol}")
                return position
            
            # Calculate final values
            final_value = position.quantity * close_price
            realized_pnl = final_value - (position.quantity * position.entry_price)
            
            # Update position
            position.current_value = final_value
            position.unrealized_pnl = realized_pnl
            position.status = 'closed'
            
            # Update capital
            self.current_capital += final_value
            
            # Move from open to closed positions
            if position.symbol in self.open_positions:
                del self.open_positions[position.symbol]
            self.closed_positions.append(position)
            
            # Log position closing
            self._log_position_event('CLOSE', position, None, close_price, realized_pnl)
            
            self.logger.info(f"Closed position: {position.symbol} - "
                           f"P&L: ₹{realized_pnl:.2f} ({(realized_pnl/(position.quantity * position.entry_price))*100:.2f}%)")
            
            return position
            
        except Exception as e:
            self.logger.error(f"Error closing position for {position.symbol}: {str(e)}")
            return position
    
    def update_position_value(self, position: Position, current_price: float) -> Position:
        """
        Update position current value and P&L.
        
        Args:
            position: Position to update
            current_price: Current market price
            
        Returns:
            Updated position object
        """
        try:
            if position.status != 'open':
                return position
            
            # Calculate current values
            current_value = position.quantity * current_price
            unrealized_pnl = current_value - (position.quantity * position.entry_price)
            
            # Update position
            position.current_value = current_value
            position.unrealized_pnl = unrealized_pnl
            
            # Update in open positions dictionary
            if position.symbol in self.open_positions:
                self.open_positions[position.symbol] = position
            
            self.logger.debug(f"Updated position {position.symbol}: "
                            f"value=₹{current_value:.2f}, P&L=₹{unrealized_pnl:.2f}")
            
            return position
            
        except Exception as e:
            self.logger.error(f"Error updating position value for {position.symbol}: {str(e)}")
            return position
    
    def get_open_positions(self) -> List[Position]:
        """
        Get all currently open positions.
        
        Returns:
            List of open positions
        """
        return list(self.open_positions.values())
    
    def get_position_by_symbol(self, symbol: str) -> Optional[Position]:
        """
        Get open position for a specific symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Position object if found, None otherwise
        """
        return self.open_positions.get(symbol)
    
    def has_position(self, symbol: str) -> bool:
        """
        Check if there's an open position for a symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            True if position exists, False otherwise
        """
        return symbol in self.open_positions
    
    def get_portfolio_value(self) -> float:
        """
        Calculate total portfolio value (cash + positions).
        
        Returns:
            Total portfolio value
        """
        try:
            positions_value = sum(pos.current_value for pos in self.open_positions.values())
            total_value = self.current_capital + positions_value
            
            self.logger.debug(f"Portfolio value: cash=₹{self.current_capital:.2f}, "
                            f"positions=₹{positions_value:.2f}, total=₹{total_value:.2f}")
            
            return total_value
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio value: {str(e)}")
            return self.current_capital
    
    def get_total_pnl(self) -> float:
        """
        Calculate total P&L (realized + unrealized).
        
        Returns:
            Total P&L amount
        """
        try:
            # Unrealized P&L from open positions
            unrealized_pnl = sum(pos.unrealized_pnl for pos in self.open_positions.values())
            
            # Realized P&L from closed positions
            realized_pnl = sum(pos.unrealized_pnl for pos in self.closed_positions)
            
            total_pnl = unrealized_pnl + realized_pnl
            
            self.logger.debug(f"Total P&L: unrealized=₹{unrealized_pnl:.2f}, "
                            f"realized=₹{realized_pnl:.2f}, total=₹{total_pnl:.2f}")
            
            return total_pnl
            
        except Exception as e:
            self.logger.error(f"Error calculating total P&L: {str(e)}")
            return 0.0
    
    def get_position_count(self) -> int:
        """
        Get number of open positions.
        
        Returns:
            Number of open positions
        """
        return len(self.open_positions)
    
    def get_available_capital(self) -> float:
        """
        Get available capital for new positions.
        
        Returns:
            Available capital amount
        """
        return self.current_capital
    
    def update_positions_batch(self, price_data: Dict[str, float]) -> Dict[str, Position]:
        """
        Update multiple positions with current prices.
        
        Args:
            price_data: Dictionary mapping symbols to current prices
            
        Returns:
            Dictionary of updated positions
        """
        updated_positions = {}
        
        for symbol, position in self.open_positions.items():
            if symbol in price_data:
                updated_position = self.update_position_value(position, price_data[symbol])
                updated_positions[symbol] = updated_position
        
        self.logger.debug(f"Updated {len(updated_positions)} positions with current prices")
        return updated_positions
    
    def check_stop_losses(self, price_data: Dict[str, float]) -> List[Position]:
        """
        Check for stop-loss triggers and return positions that should be closed.
        
        Args:
            price_data: Dictionary mapping symbols to current prices
            
        Returns:
            List of positions that hit stop-loss
        """
        stop_loss_positions = []
        
        for symbol, position in self.open_positions.items():
            if symbol in price_data:
                current_price = price_data[symbol]
                
                # Check if current price is at or below stop-loss
                if current_price <= position.stop_loss_price:
                    stop_loss_positions.append(position)
                    self.logger.warning(f"Stop-loss triggered for {symbol}: "
                                      f"current=₹{current_price:.2f}, stop=₹{position.stop_loss_price:.2f}")
        
        return stop_loss_positions
    
    def close_positions_batch(self, positions: List[Position], price_data: Dict[str, float]) -> List[Position]:
        """
        Close multiple positions in batch.
        
        Args:
            positions: List of positions to close
            price_data: Dictionary mapping symbols to current prices
            
        Returns:
            List of closed positions
        """
        closed_positions = []
        
        for position in positions:
            if position.symbol in price_data:
                closed_position = self.close_position(position, price_data[position.symbol])
                closed_positions.append(closed_position)
        
        self.logger.info(f"Closed {len(closed_positions)} positions in batch")
        return closed_positions
    
    def get_position_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive position statistics.
        
        Returns:
            Dictionary with position statistics
        """
        try:
            open_positions = list(self.open_positions.values())
            
            # Calculate statistics
            total_positions = len(open_positions) + len(self.closed_positions)
            win_count = sum(1 for pos in self.closed_positions if pos.unrealized_pnl > 0)
            loss_count = len(self.closed_positions) - win_count
            
            win_rate = (win_count / len(self.closed_positions)) * 100 if self.closed_positions else 0.0
            
            avg_win = np.mean([pos.unrealized_pnl for pos in self.closed_positions if pos.unrealized_pnl > 0]) if win_count > 0 else 0.0
            avg_loss = np.mean([pos.unrealized_pnl for pos in self.closed_positions if pos.unrealized_pnl <= 0]) if loss_count > 0 else 0.0
            
            portfolio_value = self.get_portfolio_value()
            total_return = ((portfolio_value - self.initial_capital) / self.initial_capital) * 100
            
            return {
                'open_positions': len(open_positions),
                'closed_positions': len(self.closed_positions),
                'total_positions': total_positions,
                'win_count': win_count,
                'loss_count': loss_count,
                'win_rate': win_rate,
                'average_win': avg_win,
                'average_loss': avg_loss,
                'portfolio_value': portfolio_value,
                'available_capital': self.current_capital,
                'total_pnl': self.get_total_pnl(),
                'total_return_pct': total_return,
                'capital_utilization': ((self.initial_capital - self.current_capital) / self.initial_capital) * 100
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating position statistics: {str(e)}")
            return {}
    
    def _log_position_event(self, event_type: str, position: Position, 
                           signal: Optional[TradingSignal] = None,
                           close_price: Optional[float] = None,
                           pnl: Optional[float] = None) -> None:
        """
        Log position events for audit trail.
        
        Args:
            event_type: Type of event ('OPEN', 'CLOSE', 'UPDATE')
            position: Position object
            signal: Trading signal (for open events)
            close_price: Closing price (for close events)
            pnl: Realized P&L (for close events)
        """
        try:
            event = {
                'timestamp': datetime.now(),
                'event_type': event_type,
                'symbol': position.symbol,
                'quantity': position.quantity,
                'entry_price': position.entry_price,
                'current_value': position.current_value,
                'unrealized_pnl': position.unrealized_pnl,
                'status': position.status
            }
            
            if signal:
                event.update({
                    'signal_action': signal.action,
                    'signal_confidence': signal.confidence,
                    'lstm_prediction': signal.lstm_prediction,
                    'sentiment_score': signal.sentiment_score
                })
            
            if close_price is not None:
                event['close_price'] = close_price
            
            if pnl is not None:
                event['realized_pnl'] = pnl
            
            self.position_history.append(event)
            
        except Exception as e:
            self.logger.error(f"Error logging position event: {str(e)}")
    
    def get_position_history(self) -> List[Dict]:
        """
        Get complete position history.
        
        Returns:
            List of position events
        """
        return self.position_history.copy()
    
    def reset_portfolio(self, new_capital: Optional[float] = None) -> None:
        """
        Reset portfolio to initial state.
        
        Args:
            new_capital: New initial capital (optional)
        """
        if new_capital is not None:
            self.initial_capital = new_capital
        
        self.current_capital = self.initial_capital
        self.open_positions.clear()
        self.closed_positions.clear()
        self.position_history.clear()
        
        self.logger.info(f"Portfolio reset with capital ₹{self.initial_capital:,.2f}")
    
    def export_positions_to_dataframe(self) -> pd.DataFrame:
        """
        Export position history to pandas DataFrame.
        
        Returns:
            DataFrame with position history
        """
        try:
            if not self.position_history:
                return pd.DataFrame()
            
            df = pd.DataFrame(self.position_history)
            return df
            
        except Exception as e:
            self.logger.error(f"Error exporting positions to DataFrame: {str(e)}")
            return pd.DataFrame()