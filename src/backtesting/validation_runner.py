"""
Backtesting validation runner for the quantitative trading system.
Runs comprehensive backtesting validation on historical data (1995-2025).
"""

import os
import sys
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple

from src.config.settings import config
from src.utils.logging_utils import get_logger, log_performance_metrics
from src.backtesting.backtest_engine import BacktestEngine
from src.monitoring.performance_tracker import PerformanceTracker
from src.monitoring.visualization import TradingVisualizer


class ValidationRunner:
    """
    Runs comprehensive backtesting validation on historical data.
    Validates performance targets and benchmark comparisons.
    """
    
    def __init__(self):
        """Initialize validation runner."""
        self.logger = get_logger("ValidationRunner")
        self.backtest_engine = BacktestEngine()
        self.performance_tracker = PerformanceTracker()
        self.visualizer = TradingVisualizer()
    
    def run_historical_validation(self, symbols: List[str], 
                                 start_date: str = "1995-01-01",
                                 end_date: str = None) -> Tuple[bool, Dict[str, Any]]:
        """
        Run complete backtesting validation on historical data.
        
        Args:
            symbols: List of stock symbols to validate
            start_date: Start date for validation (default: 1995-01-01)
            end_date: End date for validation (default: current date)
            
        Returns:
            Tuple of (success_flag, performance_metrics)
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        self.logger.info(f"Starting historical validation from {start_date} to {end_date}")
        self.logger.info(f"Validating {len(symbols)} symbols")
        
        start_time = time.time()
        
        try:
            # Initialize backtest
            self.logger.info("Initializing backtest engine...")
            init_success = self.backtest_engine.initialize_backtest(
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                initial_capital=config.backtest.initial_capital
            )
            
            if not init_success:
                self.logger.error("Failed to initialize backtest engine")
                return False, {}
            
            # Run backtest
            self.logger.info("Running backtest simulation...")
            backtest_results = self.backtest_engine.run_backtest()
            
            if backtest_results is None:
                self.logger.error("Backtest simulation failed")
                return False, {}
            
            # Calculate performance metrics
            self.logger.info("Calculating performance metrics...")
            metrics = self.performance_tracker.calculate_comprehensive_metrics(backtest_results)
            
            # Log performance metrics
            log_performance_metrics(metrics)
            
            # Generate visualizations
            self.logger.info("Generating performance visualizations...")
            self.visualizer.create_backtest_visualizations(backtest_results, metrics)
            
            # Validate performance targets
            validation_result = self._validate_performance_targets(metrics)
            
            # Generate validation report
            self._generate_validation_report(metrics, backtest_results, validation_result)
            
            end_time = time.time()
            duration = end_time - start_time
            
            self.logger.info(f"Historical validation completed in {duration:.2f} seconds")
            self.logger.info(f"Performance target validation: {'PASSED' if validation_result else 'FAILED'}")
            
            return validation_result, metrics
            
        except Exception as e:
            self.logger.error(f"Historical validation failed: {str(e)}")
            self.logger.exception("Validation error details:")
            return False, {}
    
    def run_period_comparison(self, symbols: List[str], periods: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Run backtesting comparison across different time periods.
        
        Args:
            symbols: List of stock symbols to validate
            periods: List of period dictionaries with 'name', 'start_date', 'end_date'
            
        Returns:
            Dictionary of period names mapped to performance metrics
        """
        self.logger.info(f"Starting period comparison for {len(periods)} periods")
        
        results = {}
        
        for period in periods:
            name = period.get('name', f"{period['start_date']} to {period['end_date']}")
            self.logger.info(f"Running backtest for period: {name}")
            
            success, metrics = self.run_historical_validation(
                symbols=symbols,
                start_date=period['start_date'],
                end_date=period['end_date']
            )
            
            if success:
                results[name] = metrics
            else:
                self.logger.warning(f"Validation failed for period: {name}")
        
        # Compare period results
        if len(results) > 1:
            self._generate_period_comparison_report(results)
        
        return results
    
    def optimize_system_parameters(self, symbols: List[str], 
                                  parameter_sets: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Optimize system parameters through backtesting.
        
        Args:
            symbols: List of stock symbols to validate
            parameter_sets: List of parameter dictionaries to test
            
        Returns:
            Dictionary with best parameters and performance metrics
        """
        self.logger.info(f"Starting parameter optimization with {len(parameter_sets)} parameter sets")
        
        best_metrics = None
        best_params = None
        best_sharpe = -float('inf')
        
        for i, params in enumerate(parameter_sets):
            self.logger.info(f"Testing parameter set {i+1}/{len(parameter_sets)}")
            
            # Apply parameters temporarily
            self._apply_parameters(params)
            
            # Run validation
            success, metrics = self.run_historical_validation(
                symbols=symbols,
                start_date=config.backtest.train_start_date,
                end_date=config.backtest.train_end_date
            )
            
            if success and metrics.get('sharpe_ratio', 0) > best_sharpe:
                best_sharpe = metrics.get('sharpe_ratio', 0)
                best_metrics = metrics
                best_params = params
                self.logger.info(f"New best parameter set found: Sharpe = {best_sharpe:.4f}")
        
        # Restore default parameters
        self._restore_default_parameters()
        
        # Generate optimization report
        if best_params:
            self._generate_optimization_report(best_params, best_metrics)
        
        return {
            'best_parameters': best_params,
            'best_metrics': best_metrics
        }
    
    def _validate_performance_targets(self, metrics: Dict[str, Any]) -> bool:
        """
        Validate if performance meets target criteria.
        
        Args:
            metrics: Dictionary of performance metrics
            
        Returns:
            Boolean indicating if targets are met
        """
        targets_met = []
        targets_missed = []
        
        # Check annual return target
        target_return = config.performance.target_annual_return
        actual_return = metrics.get('annualized_return', 0)
        if actual_return >= target_return:
            targets_met.append(f"Annual Return: {actual_return:.2%} (Target: {target_return:.2%})")
        else:
            targets_missed.append(f"Annual Return: {actual_return:.2%} (Target: {target_return:.2%})")
        
        # Check Sharpe ratio target
        target_sharpe = config.performance.target_sharpe_ratio
        actual_sharpe = metrics.get('sharpe_ratio', 0)
        if actual_sharpe >= target_sharpe:
            targets_met.append(f"Sharpe Ratio: {actual_sharpe:.2f} (Target: {target_sharpe:.2f})")
        else:
            targets_missed.append(f"Sharpe Ratio: {actual_sharpe:.2f} (Target: {target_sharpe:.2f})")
        
        # Check max drawdown limit
        target_drawdown = config.performance.max_drawdown_limit
        actual_drawdown = metrics.get('max_drawdown', 1)
        if actual_drawdown <= target_drawdown:
            targets_met.append(f"Max Drawdown: {actual_drawdown:.2%} (Target: ≤{target_drawdown:.2%})")
        else:
            targets_missed.append(f"Max Drawdown: {actual_drawdown:.2%} (Target: ≤{target_drawdown:.2%})")
        
        # Check win rate target
        target_winrate = config.performance.target_win_rate
        actual_winrate = metrics.get('win_rate', 0)
        if actual_winrate >= target_winrate:
            targets_met.append(f"Win Rate: {actual_winrate:.2%} (Target: {target_winrate:.2%})")
        else:
            targets_missed.append(f"Win Rate: {actual_winrate:.2%} (Target: {target_winrate:.2%})")
        
        # Log results
        self.logger.info(f"Performance targets met ({len(targets_met)}/{len(targets_met) + len(targets_missed)}):")
        for target in targets_met:
            self.logger.info(f"  ✓ {target}")
        
        if targets_missed:
            self.logger.warning("Performance targets missed:")
            for target in targets_missed:
                self.logger.warning(f"  ✗ {target}")
        
        # Require at least 3 out of 4 targets to be met
        return len(targets_met) >= 3
    
    def _generate_validation_report(self, metrics: Dict[str, Any], 
                                   backtest_results: Dict[str, Any],
                                   validation_result: bool):
        """
        Generate comprehensive validation report.
        
        Args:
            metrics: Dictionary of performance metrics
            backtest_results: Dictionary of backtest results
            validation_result: Boolean indicating if validation passed
        """
        report_path = f"data/performance/validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        try:
            with open(report_path, 'w') as f:
                f.write("QUANTITATIVE TRADING SYSTEM - VALIDATION REPORT\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Validation Result: {'PASSED' if validation_result else 'FAILED'}\n\n")
                
                f.write("PERFORMANCE METRICS\n")
                f.write("-" * 30 + "\n")
                for key, value in metrics.items():
                    if isinstance(value, float):
                        if key.endswith('_rate') or key.endswith('_return') or key == 'max_drawdown':
                            f.write(f"{key}: {value:.2%}\n")
                        else:
                            f.write(f"{key}: {value:.4f}\n")
                    else:
                        f.write(f"{key}: {value}\n")
                
                f.write("\nPERFORMANCE TARGETS\n")
                f.write("-" * 30 + "\n")
                f.write(f"Annual Return: {metrics.get('annualized_return', 0):.2%} (Target: {config.performance.target_annual_return:.2%})\n")
                f.write(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f} (Target: {config.performance.target_sharpe_ratio:.2f})\n")
                f.write(f"Max Drawdown: {metrics.get('max_drawdown', 0):.2%} (Target: ≤{config.performance.max_drawdown_limit:.2%})\n")
                f.write(f"Win Rate: {metrics.get('win_rate', 0):.2%} (Target: {config.performance.target_win_rate:.2%})\n")
                
                f.write("\nBENCHMARK COMPARISON\n")
                f.write("-" * 30 + "\n")
                f.write(f"Strategy Return: {metrics.get('total_return', 0):.2%}\n")
                f.write(f"Benchmark Return: {metrics.get('benchmark_return', 0):.2%}\n")
                f.write(f"Alpha: {metrics.get('alpha', 0):.4f}\n")
                f.write(f"Beta: {metrics.get('beta', 0):.4f}\n")
                
                f.write("\nTRADING STATISTICS\n")
                f.write("-" * 30 + "\n")
                f.write(f"Total Trades: {metrics.get('total_trades', 0)}\n")
                f.write(f"Win Rate: {metrics.get('win_rate', 0):.2%}\n")
                f.write(f"Average Profit per Trade: {metrics.get('avg_profit_per_trade', 0):.2f}\n")
                f.write(f"Average Loss per Trade: {metrics.get('avg_loss_per_trade', 0):.2f}\n")
                f.write(f"Profit Factor: {metrics.get('profit_factor', 0):.2f}\n")
                
                f.write("\nSYSTEM CONFIGURATION\n")
                f.write("-" * 30 + "\n")
                config_dict = config.to_dict()
                for section, values in config_dict.items():
                    f.write(f"\n{section.upper()}:\n")
                    if isinstance(values, dict):
                        for k, v in values.items():
                            f.write(f"  {k}: {v}\n")
            
            self.logger.info(f"Validation report saved to {report_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to generate validation report: {str(e)}")
    
    def _generate_period_comparison_report(self, period_results: Dict[str, Dict[str, Any]]):
        """
        Generate report comparing performance across different time periods.
        
        Args:
            period_results: Dictionary mapping period names to performance metrics
        """
        report_path = f"data/performance/period_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        try:
            with open(report_path, 'w') as f:
                f.write("QUANTITATIVE TRADING SYSTEM - PERIOD COMPARISON REPORT\n")
                f.write("=" * 70 + "\n\n")
                f.write(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # Create comparison table
                metrics_to_compare = [
                    'annualized_return', 'sharpe_ratio', 'max_drawdown', 
                    'win_rate', 'alpha', 'beta'
                ]
                
                # Write header
                f.write(f"{'Metric':<20}")
                for period in period_results.keys():
                    f.write(f"{period:<15}")
                f.write("\n")
                f.write("-" * 70 + "\n")
                
                # Write metrics
                for metric in metrics_to_compare:
                    f.write(f"{metric:<20}")
                    for period, metrics in period_results.items():
                        value = metrics.get(metric, 0)
                        if metric in ['annualized_return', 'max_drawdown', 'win_rate']:
                            f.write(f"{value:.2%:<15}")
                        else:
                            f.write(f"{value:.2f}{'  ':<15}")
                    f.write("\n")
                
                # Write analysis
                f.write("\nANALYSIS\n")
                f.write("-" * 30 + "\n")
                
                # Find best and worst periods for each metric
                for metric in metrics_to_compare:
                    best_period = max(period_results.items(), key=lambda x: x[1].get(metric, 0) if metric != 'max_drawdown' else -x[1].get(metric, 0))
                    worst_period = min(period_results.items(), key=lambda x: x[1].get(metric, 0) if metric != 'max_drawdown' else -x[1].get(metric, 0))
                    
                    best_value = best_period[1].get(metric, 0)
                    worst_value = worst_period[1].get(metric, 0)
                    
                    if metric in ['annualized_return', 'max_drawdown', 'win_rate']:
                        f.write(f"Best {metric}: {best_period[0]} ({best_value:.2%})\n")
                        f.write(f"Worst {metric}: {worst_period[0]} ({worst_value:.2%})\n")
                    else:
                        f.write(f"Best {metric}: {best_period[0]} ({best_value:.2f})\n")
                        f.write(f"Worst {metric}: {worst_period[0]} ({worst_value:.2f})\n")
                    f.write("\n")
            
            self.logger.info(f"Period comparison report saved to {report_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to generate period comparison report: {str(e)}")
    
    def _generate_optimization_report(self, best_params: Dict[str, Any], 
                                     best_metrics: Dict[str, Any]):
        """
        Generate report for parameter optimization results.
        
        Args:
            best_params: Dictionary of best parameters
            best_metrics: Dictionary of performance metrics with best parameters
        """
        report_path = f"data/performance/optimization_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        try:
            with open(report_path, 'w') as f:
                f.write("QUANTITATIVE TRADING SYSTEM - OPTIMIZATION REPORT\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                f.write("OPTIMIZED PARAMETERS\n")
                f.write("-" * 30 + "\n")
                for param, value in best_params.items():
                    f.write(f"{param}: {value}\n")
                
                f.write("\nPERFORMANCE WITH OPTIMIZED PARAMETERS\n")
                f.write("-" * 30 + "\n")
                for key, value in best_metrics.items():
                    if isinstance(value, float):
                        if key.endswith('_rate') or key.endswith('_return') or key == 'max_drawdown':
                            f.write(f"{key}: {value:.2%}\n")
                        else:
                            f.write(f"{key}: {value:.4f}\n")
                    else:
                        f.write(f"{key}: {value}\n")
                
                f.write("\nRECOMMENDED CONFIGURATION\n")
                f.write("-" * 30 + "\n")
                f.write("To use these optimized parameters, update the following in src/config/settings.py:\n\n")
                
                for param, value in best_params.items():
                    if param.startswith('lstm_'):
                        f.write(f"config.model.{param} = {value}\n")
                    elif param.startswith('dqn_'):
                        f.write(f"config.model.{param} = {value}\n")
                    elif param.startswith('risk_'):
                        f.write(f"config.risk.{param.replace('risk_', '')} = {value}\n")
                    elif param.startswith('tech_'):
                        f.write(f"config.technical_indicators.{param.replace('tech_', '')} = {value}\n")
            
            self.logger.info(f"Optimization report saved to {report_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to generate optimization report: {str(e)}")
    
    def _apply_parameters(self, params: Dict[str, Any]):
        """
        Temporarily apply parameters for testing.
        
        Args:
            params: Dictionary of parameters to apply
        """
        # Store original parameters
        self._original_params = {}
        
        # Apply LSTM parameters
        for param, value in params.items():
            if param.startswith('lstm_'):
                self._original_params[param] = getattr(config.model, param)
                setattr(config.model, param, value)
            elif param.startswith('dqn_'):
                self._original_params[param] = getattr(config.model, param)
                setattr(config.model, param, value)
            elif param.startswith('risk_'):
                param_name = param.replace('risk_', '')
                self._original_params[param] = getattr(config.risk, param_name)
                setattr(config.risk, param_name, value)
            elif param.startswith('tech_'):
                param_name = param.replace('tech_', '')
                self._original_params[param] = getattr(config.technical_indicators, param_name)
                setattr(config.technical_indicators, param_name, value)
    
    def _restore_default_parameters(self):
        """Restore original parameters after testing."""
        if hasattr(self, '_original_params'):
            for param, value in self._original_params.items():
                if param.startswith('lstm_'):
                    setattr(config.model, param, value)
                elif param.startswith('dqn_'):
                    setattr(config.model, param, value)
                elif param.startswith('risk_'):
                    param_name = param.replace('risk_', '')
                    setattr(config.risk, param_name, value)
                elif param.startswith('tech_'):
                    param_name = param.replace('tech_', '')
                    setattr(config.technical_indicators, param_name, value)


def run_validation(symbols: List[str] = None, start_date: str = None, end_date: str = None) -> bool:
    """
    Run system validation with specified parameters.
    
    Args:
        symbols: List of stock symbols (default: from config)
        start_date: Start date (default: from config)
        end_date: End date (default: current date)
        
    Returns:
        Boolean indicating if validation passed
    """
    logger = get_logger("Validation")
    
    if symbols is None:
        symbols = config.get_stock_symbols()
    
    if start_date is None:
        start_date = config.backtest.train_start_date
    
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    
    logger.info(f"Running system validation for {len(symbols)} symbols")
    logger.info(f"Date range: {start_date} to {end_date}")
    
    runner = ValidationRunner()
    success, metrics = runner.run_historical_validation(symbols, start_date, end_date)
    
    return success


def run_optimization(symbols: List[str] = None) -> Dict[str, Any]:
    """
    Run parameter optimization.
    
    Args:
        symbols: List of stock symbols (default: from config)
        
    Returns:
        Dictionary with optimization results
    """
    logger = get_logger("Optimization")
    
    if symbols is None:
        symbols = config.get_stock_symbols()[:10]  # Use subset for faster optimization
    
    logger.info(f"Running parameter optimization for {len(symbols)} symbols")
    
    # Define parameter sets to test
    parameter_sets = [
        # Default parameters
        {},
        
        # LSTM variations
        {'lstm_units': 32, 'lstm_dropout': 0.1},
        {'lstm_units': 64, 'lstm_dropout': 0.2},
        {'lstm_units': 100, 'lstm_dropout': 0.3},
        
        # DQN variations
        {'dqn_learning_rate': 0.0005, 'dqn_epsilon_decay': 0.99},
        {'dqn_learning_rate': 0.002, 'dqn_epsilon_decay': 0.98},
        
        # Risk variations
        {'risk_stop_loss_percentage': 0.03, 'risk_position_size_percentage': 0.01},
        {'risk_stop_loss_percentage': 0.07, 'risk_position_size_percentage': 0.03},
        
        # Technical indicator variations
        {'tech_rsi_period': 10, 'tech_sma_period': 30},
        {'tech_rsi_period': 20, 'tech_sma_period': 100}
    ]
    
    runner = ValidationRunner()
    results = runner.optimize_system_parameters(symbols, parameter_sets)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="System Validation Runner")
    parser.add_argument("--mode", choices=["validate", "optimize"], default="validate",
                       help="Validation mode")
    parser.add_argument("--symbols", nargs="+", help="Stock symbols to use")
    parser.add_argument("--start", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", help="End date (YYYY-MM-DD)")
    
    args = parser.parse_args()
    
    if args.mode == "validate":
        success = run_validation(args.symbols, args.start, args.end)
        sys.exit(0 if success else 1)
    elif args.mode == "optimize":
        results = run_optimization(args.symbols)
        sys.exit(0 if results.get('best_params') else 1)