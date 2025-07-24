"""
Performance Tracker for real-time performance monitoring and logging.
Implements comprehensive performance tracking with trade logging and benchmark comparison.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import logging
import json
import os
from dataclasses import dataclass, asdict
from collections import defaultdict

from ..interfaces.trading_interfaces import PerformanceMetrics, Position, TradingSignal
from ..backtesting.performance_calculator import PerformanceCalculator
from ..utils.logging_utils import get_logger


@dataclass
class TradeLog:
    """Trade log entry with decision rationale."""
    timestamp: datetime
    symbol: str
    action: str
    quantity: float
    price: float
    signal_confidence: float
    lstm_prediction: float
    dqn_q_values: Dict[str, float]
    sentiment_score: float
    risk_adjusted_size: float
    decision_rationale: str
    portfolio_value_before: float
    portfolio_value_after: float
    position_id: Optional[str] = None


@dataclass
class PerformanceSnapshot:
    """Performance snapshot at a point in time."""
    timestamp: datetime
    portfolio_value: float
    total_return: float
    daily_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    open_positions: int
    benchmark_return: float
    alpha: float
    beta: float


class PerformanceTracker:
    """
    Real-time performance tracking system with comprehensive logging and monitoring.
    
    Features:
    - Real-time performance metric calculation
    - Trade logging with decision rationale
    - Benchmark comparison against NIFTY 50
    - Performance snapshots and historical tracking
    - Alert system for performance degradation
    
    Requirements: 9.5, 10.1, 10.4
    """
    
    def __init__(self, initial_capital: float = 1000000.0, 
                 benchmark_symbol: str = "^NSEI"):
        """
        Initialize performance tracker.
        
        Args:
            initial_capital: Initial portfolio capital
            benchmark_symbol: Benchmark symbol for comparison (NIFTY 50)
        """
        self.initial_capital = initial_capital
        self.benchmark_symbol = benchmark_symbol
        self.logger = get_logger("PerformanceTracker")
        
        # Performance calculator
        self.performance_calculator = PerformanceCalculator()
        
        # Tracking data
        self.trade_logs: List[TradeLog] = []
        self.performance_snapshots: List[PerformanceSnapshot] = []
        self.daily_portfolio_values: List[Dict] = []
        self.benchmark_data: Optional[pd.DataFrame] = None
        
        # Current state
        self.current_portfolio_value = initial_capital
        self.current_positions: List[Position] = []
        self.last_snapshot_time: Optional[datetime] = None
        
        # Performance alerts
        self.alert_thresholds = {
            'max_drawdown_warning': 5.0,  # 5% drawdown warning
            'max_drawdown_critical': 8.0,  # 8% drawdown critical
            'sharpe_ratio_warning': 1.5,   # Below 1.5 Sharpe warning
            'win_rate_warning': 55.0       # Below 55% win rate warning
        }
        
        # Data storage paths
        self.data_dir = "data/performance"
        os.makedirs(self.data_dir, exist_ok=True)
        
        self.logger.info(f"Performance tracker initialized with ${initial_capital:,.2f} capital")
    
    def log_trade(self, signal: TradingSignal, position: Position, 
                  decision_rationale: str, portfolio_value_before: float,
                  portfolio_value_after: float) -> None:
        """
        Log trade execution with comprehensive details.
        
        Args:
            signal: Trading signal that triggered the trade
            position: Position created/modified by the trade
            decision_rationale: Explanation of the trading decision
            portfolio_value_before: Portfolio value before trade
            portfolio_value_after: Portfolio value after trade
            
        Requirements: 10.1 - Trade logging with timestamps and decision rationale
        """
        try:
            trade_log = TradeLog(
                timestamp=signal.timestamp,
                symbol=signal.symbol,
                action=signal.action,
                quantity=position.quantity,
                price=position.entry_price,
                signal_confidence=signal.confidence,
                lstm_prediction=signal.lstm_prediction,
                dqn_q_values=signal.dqn_q_values,
                sentiment_score=signal.sentiment_score,
                risk_adjusted_size=signal.risk_adjusted_size,
                decision_rationale=decision_rationale,
                portfolio_value_before=portfolio_value_before,
                portfolio_value_after=portfolio_value_after,
                position_id=f"{signal.symbol}_{signal.timestamp.strftime('%Y%m%d_%H%M%S')}"
            )
            
            self.trade_logs.append(trade_log)
            
            # Log to file
            self._log_trade_to_file(trade_log)
            
            # Log summary to console
            self.logger.info(
                f"Trade logged: {signal.action.upper()} {position.quantity} {signal.symbol} "
                f"@ ${position.entry_price:.2f} | Confidence: {signal.confidence:.3f} | "
                f"Rationale: {decision_rationale}"
            )
            
        except Exception as e:
            self.logger.error(f"Error logging trade: {e}")
    
    def update_portfolio_value(self, new_value: float, 
                             positions: List[Position]) -> None:
        """
        Update current portfolio value and positions.
        
        Args:
            new_value: New portfolio value
            positions: Current positions list
        """
        try:
            self.current_portfolio_value = new_value
            self.current_positions = positions.copy()
            
            # Add daily portfolio value record
            portfolio_record = {
                'date': datetime.now().date(),
                'portfolio_value': new_value,
                'daily_return': self._calculate_daily_return(new_value),
                'positions_count': len([p for p in positions if p.status == 'open'])
            }
            
            # Update or append daily record
            if (self.daily_portfolio_values and 
                self.daily_portfolio_values[-1]['date'] == portfolio_record['date']):
                self.daily_portfolio_values[-1] = portfolio_record
            else:
                self.daily_portfolio_values.append(portfolio_record)
            
            self.logger.debug(f"Portfolio value updated: ${new_value:,.2f}")
            
        except Exception as e:
            self.logger.error(f"Error updating portfolio value: {e}")
    
    def calculate_real_time_metrics(self) -> PerformanceMetrics:
        """
        Calculate real-time performance metrics.
        
        Returns:
            Current performance metrics
            
        Requirements: 9.5 - Real-time performance metric calculation
        """
        try:
            if not self.daily_portfolio_values:
                return self._empty_metrics()
            
            # Get completed trades for trade-based metrics
            completed_trades = [p for p in self.current_positions if p.status == 'closed']
            
            # Calculate comprehensive metrics
            metrics = self.performance_calculator.calculate_comprehensive_metrics(
                daily_portfolio_values=self.daily_portfolio_values,
                trades=completed_trades,
                benchmark_data=self.benchmark_data,
                initial_capital=self.initial_capital
            )
            
            self.logger.debug(f"Real-time metrics calculated: Return={metrics.total_return:.2f}%, "
                            f"Sharpe={metrics.sharpe_ratio:.2f}")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating real-time metrics: {e}")
            return self._empty_metrics()
    
    def create_performance_snapshot(self) -> PerformanceSnapshot:
        """
        Create performance snapshot at current time.
        
        Returns:
            Performance snapshot
        """
        try:
            metrics = self.calculate_real_time_metrics()
            
            # Calculate daily return
            daily_return = 0.0
            if len(self.daily_portfolio_values) >= 2:
                today_value = self.daily_portfolio_values[-1]['portfolio_value']
                yesterday_value = self.daily_portfolio_values[-2]['portfolio_value']
                daily_return = (today_value / yesterday_value - 1) * 100
            
            snapshot = PerformanceSnapshot(
                timestamp=datetime.now(),
                portfolio_value=self.current_portfolio_value,
                total_return=metrics.total_return,
                daily_return=daily_return,
                sharpe_ratio=metrics.sharpe_ratio,
                max_drawdown=metrics.max_drawdown,
                win_rate=metrics.win_rate,
                total_trades=metrics.total_trades,
                open_positions=len([p for p in self.current_positions if p.status == 'open']),
                benchmark_return=metrics.benchmark_return,
                alpha=metrics.alpha,
                beta=metrics.beta
            )
            
            self.performance_snapshots.append(snapshot)
            self.last_snapshot_time = snapshot.timestamp
            
            # Check for performance alerts
            self._check_performance_alerts(snapshot)
            
            return snapshot
            
        except Exception as e:
            self.logger.error(f"Error creating performance snapshot: {e}")
            return self._empty_snapshot()
    
    def load_benchmark_data(self, benchmark_data: pd.DataFrame) -> None:
        """
        Load benchmark data for comparison.
        
        Args:
            benchmark_data: Benchmark price data (NIFTY 50)
            
        Requirements: 10.4 - Performance comparison against NIFTY 50
        """
        try:
            self.benchmark_data = benchmark_data.copy()
            self.logger.info(f"Benchmark data loaded: {len(benchmark_data)} records")
            
        except Exception as e:
            self.logger.error(f"Error loading benchmark data: {e}")
    
    def compare_to_benchmark(self) -> Dict[str, Any]:
        """
        Compare current performance to benchmark.
        
        Returns:
            Benchmark comparison results
            
        Requirements: 10.4 - Performance comparison dashboard against NIFTY 50
        """
        try:
            if self.benchmark_data is None or len(self.daily_portfolio_values) == 0:
                return {'error': 'Insufficient data for benchmark comparison'}
            
            # Calculate portfolio returns
            portfolio_df = pd.DataFrame(self.daily_portfolio_values)
            portfolio_df['date'] = pd.to_datetime(portfolio_df['date'])
            portfolio_df = portfolio_df.set_index('date')
            
            # Calculate benchmark comparison using performance calculator
            benchmark_metrics = self.performance_calculator._calculate_benchmark_comparison(
                portfolio_df, self.benchmark_data
            )
            
            # Additional comparison metrics
            current_metrics = self.calculate_real_time_metrics()
            
            comparison = {
                'portfolio_return': current_metrics.total_return,
                'benchmark_return': benchmark_metrics['benchmark_return'],
                'excess_return': current_metrics.total_return - benchmark_metrics['benchmark_return'],
                'alpha': benchmark_metrics['alpha'],
                'beta': benchmark_metrics['beta'],
                'information_ratio': benchmark_metrics.get('information_ratio', 0.0),
                'tracking_error': benchmark_metrics.get('tracking_error', 0.0),
                'correlation': benchmark_metrics.get('correlation', 0.0),
                'outperformance': current_metrics.total_return > benchmark_metrics['benchmark_return'],
                'risk_adjusted_outperformance': current_metrics.sharpe_ratio > 1.8  # Target Sharpe
            }
            
            self.logger.info(f"Benchmark comparison: Portfolio={comparison['portfolio_return']:.2f}%, "
                           f"Benchmark={comparison['benchmark_return']:.2f}%, "
                           f"Excess={comparison['excess_return']:.2f}%")
            
            return comparison
            
        except Exception as e:
            self.logger.error(f"Error comparing to benchmark: {e}")
            return {'error': str(e)}
    
    def get_performance_dashboard_data(self) -> Dict[str, Any]:
        """
        Get comprehensive data for performance dashboard.
        
        Returns:
            Dashboard data dictionary
        """
        try:
            current_metrics = self.calculate_real_time_metrics()
            latest_snapshot = self.create_performance_snapshot()
            benchmark_comparison = self.compare_to_benchmark()
            
            dashboard_data = {
                'current_metrics': {
                    'portfolio_value': f"${self.current_portfolio_value:,.2f}",
                    'total_return': f"{current_metrics.total_return:.2f}%",
                    'daily_return': f"{latest_snapshot.daily_return:.2f}%",
                    'sharpe_ratio': f"{current_metrics.sharpe_ratio:.2f}",
                    'max_drawdown': f"{current_metrics.max_drawdown:.2f}%",
                    'win_rate': f"{current_metrics.win_rate:.1f}%",
                    'total_trades': current_metrics.total_trades,
                    'open_positions': latest_snapshot.open_positions
                },
                'benchmark_comparison': benchmark_comparison,
                'recent_trades': self._get_recent_trades(limit=10),
                'performance_alerts': self._get_active_alerts(),
                'historical_snapshots': [asdict(s) for s in self.performance_snapshots[-30:]],  # Last 30 snapshots
                'timestamp': datetime.now().isoformat()
            }
            
            return dashboard_data
            
        except Exception as e:
            self.logger.error(f"Error generating dashboard data: {e}")
            return {'error': str(e)}
    
    def export_performance_data(self, filepath: Optional[str] = None) -> str:
        """
        Export performance data to JSON file.
        
        Args:
            filepath: Optional custom filepath
            
        Returns:
            Path to exported file
        """
        try:
            if filepath is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filepath = os.path.join(self.data_dir, f"performance_data_{timestamp}.json")
            
            export_data = {
                'metadata': {
                    'export_timestamp': datetime.now().isoformat(),
                    'initial_capital': self.initial_capital,
                    'benchmark_symbol': self.benchmark_symbol
                },
                'trade_logs': [asdict(log) for log in self.trade_logs],
                'performance_snapshots': [asdict(snapshot) for snapshot in self.performance_snapshots],
                'daily_portfolio_values': self.daily_portfolio_values,
                'current_metrics': asdict(self.calculate_real_time_metrics()),
                'benchmark_comparison': self.compare_to_benchmark()
            }
            
            # Convert datetime objects to strings for JSON serialization
            export_data = self._serialize_datetime_objects(export_data)
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            self.logger.info(f"Performance data exported to: {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error(f"Error exporting performance data: {e}")
            return ""
    
    def _calculate_daily_return(self, current_value: float) -> float:
        """Calculate daily return percentage."""
        if len(self.daily_portfolio_values) == 0:
            return 0.0
        
        previous_value = self.daily_portfolio_values[-1]['portfolio_value']
        return (current_value / previous_value - 1) * 100 if previous_value > 0 else 0.0
    
    def _log_trade_to_file(self, trade_log: TradeLog) -> None:
        """Log trade to file."""
        try:
            log_file = os.path.join(self.data_dir, "trade_log.jsonl")
            
            with open(log_file, 'a') as f:
                trade_dict = asdict(trade_log)
                trade_dict = self._serialize_datetime_objects(trade_dict)
                f.write(json.dumps(trade_dict, default=str) + '\n')
                
        except Exception as e:
            self.logger.error(f"Error writing trade log to file: {e}")
    
    def _check_performance_alerts(self, snapshot: PerformanceSnapshot) -> None:
        """Check for performance alerts and log warnings."""
        try:
            alerts = []
            
            # Drawdown alerts
            if snapshot.max_drawdown >= self.alert_thresholds['max_drawdown_critical']:
                alerts.append(f"CRITICAL: Maximum drawdown {snapshot.max_drawdown:.2f}% exceeds critical threshold")
            elif snapshot.max_drawdown >= self.alert_thresholds['max_drawdown_warning']:
                alerts.append(f"WARNING: Maximum drawdown {snapshot.max_drawdown:.2f}% exceeds warning threshold")
            
            # Sharpe ratio alerts
            if snapshot.sharpe_ratio < self.alert_thresholds['sharpe_ratio_warning']:
                alerts.append(f"WARNING: Sharpe ratio {snapshot.sharpe_ratio:.2f} below target threshold")
            
            # Win rate alerts
            if snapshot.win_rate < self.alert_thresholds['win_rate_warning']:
                alerts.append(f"WARNING: Win rate {snapshot.win_rate:.1f}% below target threshold")
            
            # Log alerts
            for alert in alerts:
                self.logger.warning(alert)
                
        except Exception as e:
            self.logger.error(f"Error checking performance alerts: {e}")
    
    def _get_recent_trades(self, limit: int = 10) -> List[Dict]:
        """Get recent trades for dashboard."""
        try:
            recent_trades = self.trade_logs[-limit:] if self.trade_logs else []
            return [asdict(trade) for trade in recent_trades]
        except Exception as e:
            self.logger.error(f"Error getting recent trades: {e}")
            return []
    
    def _get_active_alerts(self) -> List[str]:
        """Get currently active performance alerts."""
        try:
            if not self.performance_snapshots:
                return []
            
            latest_snapshot = self.performance_snapshots[-1]
            alerts = []
            
            if latest_snapshot.max_drawdown >= self.alert_thresholds['max_drawdown_warning']:
                alerts.append(f"High drawdown: {latest_snapshot.max_drawdown:.2f}%")
            
            if latest_snapshot.sharpe_ratio < self.alert_thresholds['sharpe_ratio_warning']:
                alerts.append(f"Low Sharpe ratio: {latest_snapshot.sharpe_ratio:.2f}")
            
            if latest_snapshot.win_rate < self.alert_thresholds['win_rate_warning']:
                alerts.append(f"Low win rate: {latest_snapshot.win_rate:.1f}%")
            
            return alerts
            
        except Exception as e:
            self.logger.error(f"Error getting active alerts: {e}")
            return []
    
    def _serialize_datetime_objects(self, obj: Any) -> Any:
        """Recursively serialize datetime objects to strings."""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {key: self._serialize_datetime_objects(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._serialize_datetime_objects(item) for item in obj]
        else:
            return obj
    
    def _empty_metrics(self) -> PerformanceMetrics:
        """Return empty performance metrics."""
        return PerformanceMetrics(
            total_return=0.0,
            annualized_return=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            win_rate=0.0,
            total_trades=0,
            benchmark_return=0.0,
            alpha=0.0,
            beta=0.0
        )
    
    def _empty_snapshot(self) -> PerformanceSnapshot:
        """Return empty performance snapshot."""
        return PerformanceSnapshot(
            timestamp=datetime.now(),
            portfolio_value=self.initial_capital,
            total_return=0.0,
            daily_return=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            win_rate=0.0,
            total_trades=0,
            open_positions=0,
            benchmark_return=0.0,
            alpha=0.0,
            beta=0.0
        )