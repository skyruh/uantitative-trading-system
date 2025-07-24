"""
Backtesting module for the quantitative trading system.
Provides backtesting engine and performance calculation capabilities.
"""

from .backtest_engine import BacktestEngine, BacktestConfig
from .performance_calculator import PerformanceCalculator, PerformanceTargets

__all__ = ['BacktestEngine', 'BacktestConfig', 'PerformanceCalculator', 'PerformanceTargets']