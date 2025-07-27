"""
Backtesting Engine for the quantitative trading system.
Simulates trading strategy execution on historical data with realistic constraints.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass

from ..interfaces.trading_interfaces import (
    IBacktestEngine, ITradingStrategy, PerformanceMetrics, 
    Position, TradingSignal
)
from ..trading.position_manager import PositionManager
from ..risk.risk_manager import RiskManager
from .performance_calculator import PerformanceCalculator


@dataclass
class BacktestConfig:
    """Configuration for backtesting parameters."""
    initial_capital: float = 1000000.0  # ₹10,00,000 as per requirements
    transaction_cost_pct: float = 0.001  # 0.1% transaction cost
    start_date: str = "1995-01-01"
    end_date: str = "2025-01-01"
    benchmark_symbol: str = "^NSEI"  # NIFTY 50 index
    rebalance_frequency: str = "daily"  # daily, weekly, monthly
    allow_short_selling: bool = False  # Enable short selling for intraday trading
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.initial_capital <= 0:
            raise ValueError("Initial capital must be positive")
        if not (0 <= self.transaction_cost_pct <= 1):
            raise ValueError("Transaction cost must be between 0 and 1")


class BacktestEngine(IBacktestEngine):
    """
    Backtesting engine that simulates trading strategy execution on historical data.
    
    Features:
    - Realistic transaction cost modeling (0.1%)
    - No look-ahead bias prevention
    - Daily rebalancing and position management
    - Comprehensive performance tracking
    - Benchmark comparison capabilities
    
    Requirements: 8.1, 8.2, 8.4, 8.6
    """
    
    def __init__(self, config: Optional[BacktestConfig] = None):
        """
        Initialize backtesting engine.
        
        Args:
            config: Backtesting configuration parameters
        """
        self.config = config or BacktestConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.position_manager: Optional[PositionManager] = None
        self.risk_manager: Optional[RiskManager] = None
        
        # Backtesting state
        self.current_date: Optional[datetime] = None
        self.market_data: Dict[str, pd.DataFrame] = {}
        self.benchmark_data: Optional[pd.DataFrame] = None
        self.trade_history: List[Dict] = []
        self.daily_portfolio_values: List[Dict] = []
        self.is_initialized = False
        
        self.logger.info(f"BacktestEngine initialized with capital ₹{self.config.initial_capital:,.2f}")
    
    def initialize_backtest(self, initial_capital: float, start_date: str, end_date: str) -> bool:
        """
        Initialize backtesting environment.
        
        Args:
            initial_capital: Starting capital amount
            start_date: Backtest start date (YYYY-MM-DD)
            end_date: Backtest end date (YYYY-MM-DD)
            
        Returns:
            True if initialization successful, False otherwise
            
        Requirements: 8.1 - Initialize with ₹10,00,000 virtual capital
        """
        try:
            # Update configuration
            self.config.initial_capital = initial_capital
            self.config.start_date = start_date
            self.config.end_date = end_date
            
            # Initialize position and risk managers
            self.position_manager = PositionManager(initial_capital)
            self.risk_manager = RiskManager()
            
            # Reset state
            self.current_date = datetime.strptime(start_date, "%Y-%m-%d")
            self.trade_history.clear()
            self.daily_portfolio_values.clear()
            self.market_data.clear()
            
            # Initialize daily portfolio tracking
            self.daily_portfolio_values.append({
                'date': self.current_date,
                'portfolio_value': initial_capital,
                'cash': initial_capital,
                'positions_value': 0.0,
                'daily_return': 0.0,
                'cumulative_return': 0.0
            })
            
            self.is_initialized = True
            self.logger.info(f"Backtest initialized: {start_date} to {end_date}, "
                           f"capital ₹{initial_capital:,.2f}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing backtest: {e}")
            self.is_initialized = False
            return False
    
    def load_market_data(self, market_data: Dict[str, pd.DataFrame]) -> bool:
        """
        Load historical market data for backtesting.
        
        Args:
            market_data: Dictionary mapping symbols to price DataFrames
            
        Returns:
            True if data loaded successfully, False otherwise
        """
        try:
            if not market_data:
                self.logger.error("No market data provided")
                return False
            
            valid_symbols = 0
            
            # Validate and store market data
            for symbol, data in market_data.items():
                if not isinstance(data, pd.DataFrame):
                    self.logger.error(f"Invalid data format for {symbol}")
                    continue
                
                # Ensure required columns exist
                required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                if not all(col in data.columns for col in required_columns):
                    self.logger.error(f"Missing required columns for {symbol}")
                    continue
                
                # Ensure date index
                if not isinstance(data.index, pd.DatetimeIndex):
                    if 'Date' in data.columns:
                        data = data.set_index('Date')
                    else:
                        self.logger.error(f"No date information for {symbol}")
                        continue
                
                # Sort by date to prevent look-ahead bias
                data = data.sort_index()
                self.market_data[symbol] = data
                valid_symbols += 1
            
            if valid_symbols == 0:
                self.logger.error("No valid market data loaded")
                return False
            
            self.logger.info(f"Loaded market data for {valid_symbols} symbols")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading market data: {e}")
            return False
    
    def load_benchmark_data(self, benchmark_data: pd.DataFrame) -> bool:
        """
        Load benchmark data for performance comparison.
        
        Args:
            benchmark_data: Benchmark price DataFrame (e.g., NIFTY 50)
            
        Returns:
            True if benchmark data loaded successfully, False otherwise
        """
        try:
            if not isinstance(benchmark_data, pd.DataFrame):
                self.logger.error("Invalid benchmark data format")
                return False
            
            # Ensure date index
            if not isinstance(benchmark_data.index, pd.DatetimeIndex):
                if 'Date' in benchmark_data.columns:
                    benchmark_data = benchmark_data.set_index('Date')
                else:
                    self.logger.error("No date information in benchmark data")
                    return False
            
            # Sort by date and store
            self.benchmark_data = benchmark_data.sort_index()
            self.logger.info("Benchmark data loaded successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading benchmark data: {e}")
            return False
    
    def run_backtest(self, strategy: ITradingStrategy) -> PerformanceMetrics:
        """
        Run complete backtesting simulation.
        
        Args:
            strategy: Trading strategy to backtest
            
        Returns:
            Performance metrics from the backtest
            
        Requirements: 8.2, 8.4 - No look-ahead bias, transaction costs
        """
        try:
            if not self.is_initialized:
                raise ValueError("Backtest not initialized. Call initialize_backtest() first.")
            
            if not self.market_data:
                raise ValueError("No market data loaded. Call load_market_data() first.")
            
            self.logger.info("Starting backtesting simulation...")
            
            # Get date range for backtesting
            start_date = datetime.strptime(self.config.start_date, "%Y-%m-%d")
            end_date = datetime.strptime(self.config.end_date, "%Y-%m-%d")
            
            # Create date range (business days only)
            date_range = pd.bdate_range(start=start_date, end=end_date)
            
            # Run simulation day by day
            for current_date in date_range:
                self.current_date = current_date
                
                # Get available market data for current date
                daily_market_data = self._get_market_data_for_date(current_date)
                

                
                if not daily_market_data:
                    continue  # Skip days with no market data
                
                # Update position values with current prices
                self._update_position_values(daily_market_data)
                
                # Check for stop-loss triggers
                self._process_stop_losses(daily_market_data)
                
                # Generate trading signals using strategy
                signals = strategy.execute_strategy(daily_market_data)
                
                # Process signals and execute trades
                if signals:
                    self._process_trading_signals(signals, daily_market_data)
                
                # Record daily portfolio performance
                self._record_daily_performance(current_date, daily_market_data)
                
                # Log progress periodically
                if current_date.day == 1:  # Monthly logging
                    portfolio_value = self.position_manager.get_portfolio_value()
                    self.logger.info(f"Backtest progress: {current_date.strftime('%Y-%m')}, "
                                   f"Portfolio: ₹{portfolio_value:,.2f}")
            
            # Calculate final performance metrics
            performance_metrics = self.calculate_performance_metrics(
                self.position_manager.closed_positions
            )
            
            self.logger.info("Backtesting simulation completed successfully")
            return performance_metrics
            
        except Exception as e:
            self.logger.error(f"Error running backtest: {e}")
            # Return empty metrics on error
            return PerformanceMetrics(
                total_return=0.0, annualized_return=0.0, sharpe_ratio=0.0,
                max_drawdown=0.0, win_rate=0.0, total_trades=0,
                benchmark_return=0.0, alpha=0.0, beta=0.0
            )
    
    def calculate_performance_metrics(self, trades: List[Position]) -> PerformanceMetrics:
        """
        Calculate performance metrics from trading history using PerformanceCalculator.
        
        Args:
            trades: List of completed trades
            
        Returns:
            Comprehensive performance metrics
            
        Requirements: 8.5 - Performance metric calculations
        """
        try:
            # Use PerformanceCalculator for comprehensive metrics
            calculator = PerformanceCalculator()
            
            metrics = calculator.calculate_comprehensive_metrics(
                daily_portfolio_values=self.daily_portfolio_values,
                trades=trades,
                benchmark_data=self.benchmark_data,
                initial_capital=self.config.initial_capital
            )
            
            self.logger.info(f"Performance metrics calculated: "
                           f"Return={metrics.total_return:.2f}%, Sharpe={metrics.sharpe_ratio:.2f}, "
                           f"Drawdown={metrics.max_drawdown:.2f}%, Win Rate={metrics.win_rate:.1f}%")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {e}")
            return PerformanceMetrics(
                total_return=0.0, annualized_return=0.0, sharpe_ratio=0.0,
                max_drawdown=0.0, win_rate=0.0, total_trades=len(trades) if trades else 0,
                benchmark_return=0.0, alpha=0.0, beta=0.0
            )
    
    def _get_market_data_for_date(self, date: datetime) -> Dict[str, Dict[str, float]]:
        """
        Get market data for a specific date, ensuring no look-ahead bias.
        
        Args:
            date: Date to get data for
            
        Returns:
            Dictionary mapping symbols to OHLCV data
        """
        daily_data = {}
        
        for symbol, data in self.market_data.items():
            try:
                # Get data up to and including current date (no look-ahead)
                available_data = data[data.index <= date]
                
                if len(available_data) == 0:
                    continue
                
                # Get the most recent data point
                latest_data = available_data.iloc[-1]
                
                daily_data[symbol] = {
                    'Open': float(latest_data['Open']),
                    'High': float(latest_data['High']),
                    'Low': float(latest_data['Low']),
                    'Close': float(latest_data['Close']),
                    'Volume': float(latest_data['Volume'])
                }
                
            except Exception as e:
                self.logger.debug(f"Error getting data for {symbol} on {date}: {e}")
                continue
        
        return daily_data
    
    def _update_position_values(self, market_data: Dict[str, Dict[str, float]]) -> None:
        """
        Update position values with current market prices.
        
        Args:
            market_data: Current market data
        """
        if not self.position_manager:
            return
        
        # Extract current prices
        current_prices = {symbol: data['Close'] for symbol, data in market_data.items()}
        
        # Update all positions
        self.position_manager.update_positions_batch(current_prices)
    
    def _process_stop_losses(self, market_data: Dict[str, Dict[str, float]]) -> None:
        """
        Process stop-loss triggers and close positions.
        
        Args:
            market_data: Current market data
        """
        if not self.position_manager or not self.risk_manager:
            return
        
        # Extract current prices
        current_prices = {symbol: data['Close'] for symbol, data in market_data.items()}
        
        # Check for stop-loss triggers
        stop_loss_positions = self.position_manager.check_stop_losses(current_prices)
        
        # Close triggered positions
        for position in stop_loss_positions:
            if position.symbol in current_prices:
                close_price = current_prices[position.symbol]
                
                # Apply transaction costs
                transaction_cost = close_price * position.quantity * self.config.transaction_cost_pct
                adjusted_close_price = close_price - (transaction_cost / position.quantity)
                
                # Close position
                closed_position = self.position_manager.close_position(position, adjusted_close_price)
                
                # Record trade
                self._record_trade(closed_position, 'STOP_LOSS', close_price, transaction_cost)
    
    def _process_trading_signals(self, signals: List[TradingSignal], 
                               market_data: Dict[str, Dict[str, float]]) -> None:
        """
        Process trading signals and execute trades.
        
        Args:
            signals: List of trading signals
            market_data: Current market data
        """
        if not self.position_manager or not self.risk_manager:
            return
        
        for signal in signals:
            try:
                # Skip if no market data for this symbol
                if signal.symbol not in market_data:
                    continue
                
                current_price = market_data[signal.symbol]['Close']
                portfolio_value = self.position_manager.get_portfolio_value()
                current_positions = self.position_manager.get_open_positions()
                
                # Validate trade
                is_valid, reason = self.risk_manager.validate_trade(
                    signal, portfolio_value, current_positions
                )
                
                if not is_valid:
                    print(f"DEBUG: Trade rejected for {signal.symbol}: {reason}")
                    self.logger.debug(f"Trade rejected for {signal.symbol}: {reason}")
                    continue
                else:
                    print(f"DEBUG: Trade validated for {signal.symbol}: {signal.action}")
                
                # Process based on signal action
                if signal.action == 'buy':
                    self._execute_buy_signal(signal, current_price, portfolio_value)
                elif signal.action == 'sell':
                    self._execute_sell_signal(signal, current_price)
                # 'hold' signals don't require action
                
            except Exception as e:
                self.logger.error(f"Error processing signal for {signal.symbol}: {e}")
    
    def _execute_buy_signal(self, signal: TradingSignal, current_price: float, 
                          portfolio_value: float) -> None:
        """
        Execute a buy signal.
        
        Args:
            signal: Buy signal to execute
            current_price: Current market price
            portfolio_value: Current portfolio value
        """
        try:
            # Check if we already have a position in this symbol
            if self.position_manager.has_position(signal.symbol):
                self.logger.debug(f"Already have position in {signal.symbol}, skipping buy")
                return
            
            # Calculate position size
            position_size = self.risk_manager.calculate_position_size(signal, portfolio_value)
            
            # Calculate transaction costs
            position_value = portfolio_value * position_size
            quantity = int(position_value / current_price)
            actual_position_value = quantity * current_price
            transaction_cost = actual_position_value * self.config.transaction_cost_pct
            
            # Adjust for transaction costs
            total_cost = actual_position_value + transaction_cost
            
            # Check if we have enough capital
            if total_cost > self.position_manager.get_available_capital():
                self.logger.debug(f"Insufficient capital for {signal.symbol}")
                return
            
            # Create modified signal with adjusted size
            adjusted_signal = TradingSignal(
                symbol=signal.symbol,
                timestamp=self.current_date,
                action=signal.action,
                confidence=signal.confidence,
                lstm_prediction=signal.lstm_prediction,
                dqn_q_values=signal.dqn_q_values,
                sentiment_score=signal.sentiment_score,
                risk_adjusted_size=position_size
            )
            
            # Open position
            position = self.position_manager.open_position(adjusted_signal, position_size)
            
            # Update position with actual market price and transaction costs
            position.entry_price = current_price + (transaction_cost / quantity)
            position.current_value = actual_position_value
            position.stop_loss_price = self.risk_manager.set_stop_loss(position)
            
            # Record trade
            self._record_trade(position, 'BUY', current_price, transaction_cost)
            
        except Exception as e:
            self.logger.error(f"Error executing buy signal for {signal.symbol}: {e}")
    
    def _execute_sell_signal(self, signal: TradingSignal, current_price: float) -> None:
        """
        Execute a sell signal.
        
        Args:
            signal: Sell signal to execute
            current_price: Current market price
        """
        try:
            # Check if we have a position to sell
            position = self.position_manager.get_position_by_symbol(signal.symbol)
            if not position:
                self.logger.debug(f"No position to sell for {signal.symbol}")
                return
            
            # Calculate transaction costs
            transaction_cost = current_price * position.quantity * self.config.transaction_cost_pct
            adjusted_close_price = current_price - (transaction_cost / position.quantity)
            
            # Close position
            closed_position = self.position_manager.close_position(position, adjusted_close_price)
            
            # Record trade
            self._record_trade(closed_position, 'SELL', current_price, transaction_cost)
            
        except Exception as e:
            self.logger.error(f"Error executing sell signal for {signal.symbol}: {e}")
    
    def _record_trade(self, position: Position, trade_type: str, 
                     market_price: float, transaction_cost: float) -> None:
        """
        Record trade details for analysis.
        
        Args:
            position: Position involved in trade
            trade_type: Type of trade ('BUY', 'SELL', 'STOP_LOSS')
            market_price: Market price at execution
            transaction_cost: Transaction cost incurred
        """
        trade_record = {
            'date': self.current_date,
            'symbol': position.symbol,
            'trade_type': trade_type,
            'quantity': position.quantity,
            'market_price': market_price,
            'execution_price': position.entry_price if trade_type == 'BUY' else market_price,
            'transaction_cost': transaction_cost,
            'position_value': position.current_value,
            'pnl': position.unrealized_pnl if trade_type in ['SELL', 'STOP_LOSS'] else 0.0,
            'portfolio_value': self.position_manager.get_portfolio_value()
        }
        
        self.trade_history.append(trade_record)
    
    def _record_daily_performance(self, date: datetime, 
                                market_data: Dict[str, Dict[str, float]]) -> None:
        """
        Record daily portfolio performance.
        
        Args:
            date: Current date
            market_data: Current market data
        """
        try:
            portfolio_value = self.position_manager.get_portfolio_value()
            cash = self.position_manager.get_available_capital()
            positions_value = portfolio_value - cash
            
            # Calculate daily return
            if len(self.daily_portfolio_values) > 0:
                previous_value = self.daily_portfolio_values[-1]['portfolio_value']
                daily_return = (portfolio_value / previous_value - 1) * 100 if previous_value > 0 else 0.0
            else:
                daily_return = 0.0
            
            # Calculate cumulative return
            cumulative_return = (portfolio_value / self.config.initial_capital - 1) * 100
            
            daily_record = {
                'date': date,
                'portfolio_value': portfolio_value,
                'cash': cash,
                'positions_value': positions_value,
                'daily_return': daily_return,
                'cumulative_return': cumulative_return,
                'num_positions': self.position_manager.get_position_count()
            }
            
            self.daily_portfolio_values.append(daily_record)
            
        except Exception as e:
            self.logger.error(f"Error recording daily performance: {e}")
    
    def _calculate_benchmark_metrics(self, portfolio_df: pd.DataFrame) -> Tuple[float, float, float]:
        """
        Calculate benchmark comparison metrics.
        
        Args:
            portfolio_df: Portfolio performance DataFrame
            
        Returns:
            Tuple of (benchmark_return, alpha, beta)
        """
        try:
            if self.benchmark_data is None:
                return 0.0, 0.0, 0.0
            
            # Align benchmark data with portfolio dates
            benchmark_aligned = self.benchmark_data.reindex(portfolio_df.index, method='ffill')
            
            if len(benchmark_aligned) == 0:
                return 0.0, 0.0, 0.0
            
            # Calculate benchmark returns
            benchmark_returns = benchmark_aligned['Close'].pct_change().dropna()
            
            # Calculate benchmark total return
            benchmark_total_return = (benchmark_aligned['Close'].iloc[-1] / 
                                    benchmark_aligned['Close'].iloc[0] - 1) * 100
            
            # Calculate alpha and beta
            portfolio_returns = portfolio_df['daily_return'].dropna()
            
            # Align returns
            common_dates = portfolio_returns.index.intersection(benchmark_returns.index)
            if len(common_dates) > 1:
                port_returns_aligned = portfolio_returns.loc[common_dates]
                bench_returns_aligned = benchmark_returns.loc[common_dates]
                
                # Calculate beta (covariance / variance)
                covariance = np.cov(port_returns_aligned, bench_returns_aligned)[0, 1]
                benchmark_variance = np.var(bench_returns_aligned)
                beta = covariance / benchmark_variance if benchmark_variance > 0 else 0.0
                
                # Calculate alpha (portfolio return - risk-free rate - beta * (benchmark return - risk-free rate))
                risk_free_rate = 0.06  # 6% annual risk-free rate for India
                portfolio_excess_return = port_returns_aligned.mean() * 252 - risk_free_rate
                benchmark_excess_return = bench_returns_aligned.mean() * 252 - risk_free_rate
                alpha = portfolio_excess_return - beta * benchmark_excess_return
            else:
                alpha, beta = 0.0, 0.0
            
            return benchmark_total_return, alpha, beta
            
        except Exception as e:
            self.logger.error(f"Error calculating benchmark metrics: {e}")
            return 0.0, 0.0, 0.0
    
    def get_trade_history(self) -> List[Dict]:
        """Get complete trade history."""
        return self.trade_history.copy()
    
    def get_daily_performance(self) -> List[Dict]:
        """Get daily portfolio performance history."""
        return self.daily_portfolio_values.copy()
    
    def export_results_to_dataframe(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Export backtest results to DataFrames.
        
        Returns:
            Tuple of (trades_df, daily_performance_df)
        """
        try:
            trades_df = pd.DataFrame(self.trade_history) if self.trade_history else pd.DataFrame()
            performance_df = pd.DataFrame(self.daily_portfolio_values) if self.daily_portfolio_values else pd.DataFrame()
            
            return trades_df, performance_df
            
        except Exception as e:
            self.logger.error(f"Error exporting results: {e}")
            return pd.DataFrame(), pd.DataFrame()
    
    def reset_backtest(self) -> None:
        """Reset backtesting engine to initial state."""
        self.position_manager = None
        self.risk_manager = None
        self.current_date = None
        self.trade_history.clear()
        self.daily_portfolio_values.clear()
        self.is_initialized = False
        
        self.logger.info("Backtest engine reset")