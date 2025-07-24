#!/usr/bin/env python3
"""
System validation script for the Quantitative Trading System.
Runs complete backtesting validation on historical data (1995-2025).
"""

import os
import sys
import time
import argparse
from datetime import datetime

# Add src to Python path
sys.path.append('src')

from src.config.settings import config
from src.utils.logging_utils import get_logger, log_system_startup, log_system_shutdown
from src.backtesting.validation_runner import ValidationRunner, run_validation
from src.system.performance_optimizer import PerformanceOptimizer, optimize_system


def run_complete_validation(symbols=None, optimize=True):
    """
    Run complete system validation and optimization.
    
    Args:
        symbols: List of stock symbols to validate (default: from config)
        optimize: Whether to run system optimization
        
    Returns:
        Boolean indicating if validation passed
    """
    logger = get_logger("SystemValidation")
    
    logger.info("=" * 60)
    logger.info("Starting Complete System Validation")
    logger.info("=" * 60)
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    start_time = time.time()
    
    # Log system startup
    log_system_startup()
    
    try:
        # Get stock symbols
        if symbols is None:
            symbols = config.get_stock_symbols()
        
        logger.info(f"Validating {len(symbols)} stock symbols")
        
        # Run historical validation
        logger.info("Running historical backtesting validation...")
        runner = ValidationRunner()
        
        # Define time periods for validation
        periods = [
            {
                'name': 'Full History',
                'start_date': config.backtest.train_start_date,
                'end_date': config.backtest.test_end_date
            },
            {
                'name': 'Training Period',
                'start_date': config.backtest.train_start_date,
                'end_date': config.backtest.train_end_date
            },
            {
                'name': 'Testing Period',
                'start_date': config.backtest.test_start_date,
                'end_date': config.backtest.test_end_date
            },
            {
                'name': 'Recent Period',
                'start_date': '2023-01-01',
                'end_date': config.backtest.test_end_date
            }
        ]
        
        # Run period comparison
        period_results = runner.run_period_comparison(symbols[:20], periods)  # Use subset for faster validation
        
        # Run system optimization if requested
        if optimize:
            logger.info("Running system performance optimization...")
            optimization_results = optimize_system()
            logger.info(f"Optimization completed with {len(optimization_results)} optimizations")
        
        # Run final validation on full dataset
        logger.info("Running final validation on complete dataset...")
        success, metrics = runner.run_historical_validation(
            symbols=symbols,
            start_date=config.backtest.train_start_date,
            end_date=config.backtest.test_end_date
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        logger.info("=" * 60)
        logger.info(f"System Validation {'PASSED' if success else 'FAILED'}")
        logger.info(f"Validation completed in {duration:.2f} seconds")
        logger.info("=" * 60)
        
        # Log key metrics
        logger.info("Key Performance Metrics:")
        logger.info(f"  Annual Return: {metrics.get('annualized_return', 0):.2%}")
        logger.info(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
        logger.info(f"  Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
        logger.info(f"  Win Rate: {metrics.get('win_rate', 0):.2%}")
        logger.info(f"  Alpha: {metrics.get('alpha', 0):.4f}")
        logger.info(f"  Beta: {metrics.get('beta', 0):.2f}")
        
        return success
        
    except Exception as e:
        logger.error(f"System validation failed: {str(e)}")
        logger.exception("Validation error details:")
        return False
    
    finally:
        # Log system shutdown
        log_system_shutdown()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="System Validation Script")
    parser.add_argument("--symbols", nargs="+", help="Stock symbols to validate")
    parser.add_argument("--no-optimize", action="store_true", help="Skip system optimization")
    
    args = parser.parse_args()
    
    success = run_complete_validation(
        symbols=args.symbols,
        optimize=not args.no_optimize
    )
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())