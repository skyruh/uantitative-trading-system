"""
Trading interfaces for the quantitative trading system.
Defines contracts for trading strategy, risk management, and backtesting.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass


@dataclass
class TradingSignal:
    """Trading signal data model."""
    symbol: str
    timestamp: datetime
    action: str  # 'buy', 'sell', 'hold'
    confidence: float
    lstm_prediction: float
    dqn_q_values: Dict[str, float]
    sentiment_score: float
    risk_adjusted_size: float


@dataclass
class Position:
    """Position data model."""
    symbol: str
    entry_date: datetime
    entry_price: float
    quantity: int
    stop_loss_price: float
    current_value: float
    unrealized_pnl: float
    status: str  # 'open', 'closed'


@dataclass
class PerformanceMetrics:
    """Performance metrics data model."""
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    benchmark_return: float
    alpha: float
    beta: float


class ISignalGenerator(ABC):
    """Interface for trading signal generation."""
    
    @abstractmethod
    def generate_signal(self, symbol: str, market_data: Dict, 
                       lstm_prediction: float, dqn_q_values: Dict[str, float],
                       sentiment_score: float) -> TradingSignal:
        """Generate trading signal based on model outputs."""
        pass
    
    @abstractmethod
    def validate_signal(self, signal: TradingSignal) -> bool:
        """Validate trading signal before execution."""
        pass


class IRiskManager(ABC):
    """Interface for risk management operations."""
    
    @abstractmethod
    def calculate_position_size(self, signal: TradingSignal, portfolio_value: float) -> float:
        """Calculate appropriate position size based on risk rules."""
        pass
    
    @abstractmethod
    def set_stop_loss(self, position: Position) -> float:
        """Set stop-loss price for a position."""
        pass
    
    @abstractmethod
    def check_portfolio_limits(self, new_position: Position, 
                              current_positions: List[Position]) -> bool:
        """Check if new position violates portfolio diversification limits."""
        pass
    
    @abstractmethod
    def should_close_position(self, position: Position, current_price: float) -> bool:
        """Determine if position should be closed based on risk rules."""
        pass


class IPositionManager(ABC):
    """Interface for position management."""
    
    @abstractmethod
    def open_position(self, signal: TradingSignal, position_size: float) -> Position:
        """Open a new trading position."""
        pass
    
    @abstractmethod
    def close_position(self, position: Position, close_price: float) -> Position:
        """Close an existing position."""
        pass
    
    @abstractmethod
    def update_position_value(self, position: Position, current_price: float) -> Position:
        """Update position current value and P&L."""
        pass
    
    @abstractmethod
    def get_open_positions(self) -> List[Position]:
        """Get all currently open positions."""
        pass


class ITradingStrategy(ABC):
    """Interface for trading strategy execution."""
    
    @abstractmethod
    def execute_strategy(self, market_data: Dict) -> List[TradingSignal]:
        """Execute trading strategy and generate signals."""
        pass
    
    @abstractmethod
    def process_signals(self, signals: List[TradingSignal]) -> List[Position]:
        """Process trading signals and manage positions."""
        pass


class IBacktestEngine(ABC):
    """Interface for backtesting operations."""
    
    @abstractmethod
    def initialize_backtest(self, initial_capital: float, start_date: str, end_date: str) -> bool:
        """Initialize backtesting environment."""
        pass
    
    @abstractmethod
    def run_backtest(self, strategy: ITradingStrategy) -> PerformanceMetrics:
        """Run complete backtesting simulation."""
        pass
    
    @abstractmethod
    def calculate_performance_metrics(self, trades: List[Position]) -> PerformanceMetrics:
        """Calculate performance metrics from trading history."""
        pass


class IPerformanceMonitor(ABC):
    """Interface for performance monitoring."""
    
    @abstractmethod
    def track_performance(self, positions: List[Position]) -> PerformanceMetrics:
        """Track real-time performance metrics."""
        pass
    
    @abstractmethod
    def compare_to_benchmark(self, portfolio_returns: List[float], 
                           benchmark_returns: List[float]) -> Dict[str, float]:
        """Compare portfolio performance to benchmark."""
        pass
    
    @abstractmethod
    def generate_performance_report(self, metrics: PerformanceMetrics) -> Dict:
        """Generate comprehensive performance report."""
        pass