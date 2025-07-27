#!/usr/bin/env python3
"""
Paper Trading Engine for simulating live trading without real money.
Handles order execution, position management, and portfolio tracking.
"""

import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass, field
import logging
import json


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"


class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass
class PaperOrder:
    """Represents a paper trading order."""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: int
    price: Optional[float] = None  # None for market orders
    stop_price: Optional[float] = None  # For stop loss orders
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: int = 0
    filled_price: Optional[float] = None
    created_at: datetime = field(default_factory=datetime.now)
    filled_at: Optional[datetime] = None
    commission: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert order to dictionary for logging/storage."""
        return {
            'order_id': self.order_id,
            'symbol': self.symbol,
            'side': self.side.value,
            'order_type': self.order_type.value,
            'quantity': self.quantity,
            'price': self.price,
            'stop_price': self.stop_price,
            'status': self.status.value,
            'filled_quantity': self.filled_quantity,
            'filled_price': self.filled_price,
            'created_at': self.created_at.isoformat(),
            'filled_at': self.filled_at.isoformat() if self.filled_at else None,
            'commission': self.commission
        }


@dataclass
class PaperPosition:
    """Represents a paper trading position."""
    symbol: str
    quantity: int  # Positive for long, negative for short
    avg_price: float
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    @property
    def is_long(self) -> bool:
        return self.quantity > 0
    
    @property
    def is_short(self) -> bool:
        return self.quantity < 0
    
    def update_market_value(self, current_price: float):
        """Update market value and unrealized P&L."""
        self.market_value = abs(self.quantity) * current_price
        
        if self.is_long:
            self.unrealized_pnl = self.quantity * (current_price - self.avg_price)
        else:  # Short position
            self.unrealized_pnl = abs(self.quantity) * (self.avg_price - current_price)
        
        self.updated_at = datetime.now()


class PaperTradingEngine:
    """
    Paper trading engine that simulates real trading without actual money.
    Handles order execution, position management, and portfolio tracking.
    """
    
    def __init__(self, initial_capital: float = 1000000.0, commission_rate: float = 0.001):
        """
        Initialize paper trading engine.
        
        Args:
            initial_capital: Starting capital for paper trading
            commission_rate: Commission rate per trade (0.001 = 0.1%)
        """
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.logger = logging.getLogger(__name__)
        
        # Portfolio tracking
        self.cash = initial_capital
        self.positions: Dict[str, PaperPosition] = {}
        self.orders: Dict[str, PaperOrder] = {}
        self.trade_history: List[Dict] = []
        
        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.total_commission = 0.0
        self.max_drawdown = 0.0
        self.peak_portfolio_value = initial_capital
        
        self.logger.info(f"Paper trading engine initialized with ₹{initial_capital:,.2f}")
    
    def get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """Calculate total portfolio value including cash and positions."""
        portfolio_value = self.cash
        
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                position.update_market_value(current_prices[symbol])
                portfolio_value += position.market_value
        
        return portfolio_value
    
    def get_buying_power(self) -> float:
        """Get available buying power (cash for long positions)."""
        return self.cash
    
    def place_order(self, symbol: str, side: OrderSide, quantity: int, 
                   order_type: OrderType = OrderType.MARKET, 
                   price: Optional[float] = None,
                   stop_price: Optional[float] = None) -> str:
        """
        Place a paper trading order.
        
        Args:
            symbol: Stock symbol
            side: Buy or sell
            quantity: Number of shares
            order_type: Market, limit, or stop loss
            price: Limit price (for limit orders)
            stop_price: Stop price (for stop loss orders)
            
        Returns:
            Order ID
        """
        order_id = str(uuid.uuid4())[:8]
        
        # Calculate commission
        estimated_value = quantity * (price or 100)  # Rough estimate for validation
        commission = estimated_value * self.commission_rate
        
        # Validate order
        if side == OrderSide.BUY:
            required_cash = estimated_value + commission
            if required_cash > self.cash:
                self.logger.error(f"Insufficient funds for order {order_id}")
                return None
        
        # Create order
        order = PaperOrder(
            order_id=order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            stop_price=stop_price,
            commission=commission
        )
        
        self.orders[order_id] = order
        
        self.logger.info(f"Order placed: {order_id} - {side.value.upper()} {quantity} {symbol}")
        
        return order_id
    
    def execute_market_order(self, order_id: str, current_price: float) -> bool:
        """
        Execute a market order at current price.
        
        Args:
            order_id: Order ID to execute
            current_price: Current market price
            
        Returns:
            True if executed successfully
        """
        if order_id not in self.orders:
            self.logger.error(f"Order {order_id} not found")
            return False
        
        order = self.orders[order_id]
        
        if order.status != OrderStatus.PENDING:
            self.logger.warning(f"Order {order_id} is not pending")
            return False
        
        # Calculate trade value and commission
        trade_value = order.quantity * current_price
        commission = trade_value * self.commission_rate
        
        # Execute the trade
        if order.side == OrderSide.BUY:
            # Buy order
            total_cost = trade_value + commission
            
            if total_cost > self.cash:
                order.status = OrderStatus.REJECTED
                self.logger.error(f"Order {order_id} rejected: insufficient funds")
                return False
            
            # Update cash
            self.cash -= total_cost
            
            # Update position
            self._update_position(order.symbol, order.quantity, current_price)
            
        else:  # SELL
            # Sell order
            net_proceeds = trade_value - commission
            
            # Update cash
            self.cash += net_proceeds
            
            # Update position
            self._update_position(order.symbol, -order.quantity, current_price)
        
        # Update order status
        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.filled_price = current_price
        order.filled_at = datetime.now()
        order.commission = commission
        
        # Update statistics
        self.total_trades += 1
        self.total_commission += commission
        
        # Log trade
        trade_record = {
            'timestamp': datetime.now().isoformat(),
            'order_id': order_id,
            'symbol': order.symbol,
            'side': order.side.value,
            'quantity': order.quantity,
            'price': current_price,
            'value': trade_value,
            'commission': commission,
            'cash_after': self.cash
        }
        
        self.trade_history.append(trade_record)
        
        self.logger.info(f"Order executed: {order_id} - {order.side.value.upper()} "
                        f"{order.quantity} {order.symbol} @ ₹{current_price:.2f}")
        
        return True
    
    def _update_position(self, symbol: str, quantity_change: int, price: float):
        """Update position after a trade."""
        if symbol not in self.positions:
            # New position
            self.positions[symbol] = PaperPosition(
                symbol=symbol,
                quantity=quantity_change,
                avg_price=price
            )
        else:
            # Existing position
            position = self.positions[symbol]
            
            if (position.quantity > 0 and quantity_change > 0) or \
               (position.quantity < 0 and quantity_change < 0):
                # Adding to existing position (same direction)
                total_value = (position.quantity * position.avg_price) + (quantity_change * price)
                position.quantity += quantity_change
                position.avg_price = total_value / position.quantity if position.quantity != 0 else price
            else:
                # Reducing or reversing position
                if abs(quantity_change) >= abs(position.quantity):
                    # Closing or reversing position
                    realized_pnl = self._calculate_realized_pnl(position, quantity_change, price)
                    position.realized_pnl += realized_pnl
                    
                    remaining_quantity = quantity_change + position.quantity
                    if remaining_quantity == 0:
                        # Position closed
                        del self.positions[symbol]
                        return
                    else:
                        # Position reversed
                        position.quantity = remaining_quantity
                        position.avg_price = price
                else:
                    # Partially closing position
                    realized_pnl = self._calculate_realized_pnl(position, quantity_change, price)
                    position.realized_pnl += realized_pnl
                    position.quantity += quantity_change
            
            position.updated_at = datetime.now()
    
    def _calculate_realized_pnl(self, position: PaperPosition, quantity_change: int, price: float) -> float:
        """Calculate realized P&L for a position change."""
        if position.is_long and quantity_change < 0:
            # Selling long position
            return abs(quantity_change) * (price - position.avg_price)
        elif position.is_short and quantity_change > 0:
            # Covering short position
            return abs(quantity_change) * (position.avg_price - price)
        return 0.0
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order."""
        if order_id not in self.orders:
            return False
        
        order = self.orders[order_id]
        if order.status == OrderStatus.PENDING:
            order.status = OrderStatus.CANCELLED
            self.logger.info(f"Order cancelled: {order_id}")
            return True
        
        return False
    
    def get_open_orders(self) -> List[PaperOrder]:
        """Get all open (pending) orders."""
        return [order for order in self.orders.values() 
                if order.status == OrderStatus.PENDING]
    
    def get_positions(self) -> Dict[str, PaperPosition]:
        """Get all current positions."""
        return self.positions.copy()
    
    def get_position(self, symbol: str) -> Optional[PaperPosition]:
        """Get position for a specific symbol."""
        return self.positions.get(symbol)
    
    def get_performance_summary(self, current_prices: Dict[str, float]) -> Dict:
        """Get performance summary."""
        portfolio_value = self.get_portfolio_value(current_prices)
        total_return = (portfolio_value - self.initial_capital) / self.initial_capital
        
        # Update drawdown
        if portfolio_value > self.peak_portfolio_value:
            self.peak_portfolio_value = portfolio_value
        
        current_drawdown = (self.peak_portfolio_value - portfolio_value) / self.peak_portfolio_value
        self.max_drawdown = max(self.max_drawdown, current_drawdown)
        
        return {
            'initial_capital': self.initial_capital,
            'current_cash': self.cash,
            'portfolio_value': portfolio_value,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'total_trades': self.total_trades,
            'total_commission': self.total_commission,
            'max_drawdown': self.max_drawdown,
            'max_drawdown_pct': self.max_drawdown * 100,
            'open_positions': len(self.positions),
            'open_orders': len(self.get_open_orders())
        }
    
    def save_state(self, filepath: str):
        """Save current state to file."""
        state = {
            'initial_capital': self.initial_capital,
            'cash': self.cash,
            'positions': {symbol: {
                'symbol': pos.symbol,
                'quantity': pos.quantity,
                'avg_price': pos.avg_price,
                'realized_pnl': pos.realized_pnl
            } for symbol, pos in self.positions.items()},
            'trade_history': self.trade_history,
            'total_trades': self.total_trades,
            'total_commission': self.total_commission,
            'max_drawdown': self.max_drawdown
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        
        self.logger.info(f"Paper trading state saved to {filepath}")


# Example usage
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Create paper trading engine
    engine = PaperTradingEngine(initial_capital=100000)
    
    # Place some test orders
    order1 = engine.place_order("RELIANCE.NS", OrderSide.BUY, 10, OrderType.MARKET)
    order2 = engine.place_order("TCS.NS", OrderSide.BUY, 5, OrderType.MARKET)
    
    # Simulate order execution
    current_prices = {"RELIANCE.NS": 1500.0, "TCS.NS": 3200.0}
    
    if order1:
        engine.execute_market_order(order1, current_prices["RELIANCE.NS"])
    if order2:
        engine.execute_market_order(order2, current_prices["TCS.NS"])
    
    # Check performance
    performance = engine.get_performance_summary(current_prices)
    print("Performance Summary:")
    for key, value in performance.items():
        print(f"  {key}: {value}")