"""
Tests for system validation and optimization.
Validates performance targets and resource optimization.
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add src to Python path
sys.path.append('src')

from src.config.settings import config
from src.system.trading_system_orchestrator import TradingSystemOrchestrator
from src.monitoring.performance_tracker import PerformanceTracker
from src.monitoring.system_health import SystemHealthMonitor


class TestSystemValidation(unittest.TestCase):
    """Test system validation and optimization."""
    
    def setUp(self):
        """Set up test environment."""
        # Use testing environment
        self.orchestrator = TradingSystemOrchestrator(environment="testing")
        
        # Create test directories if they don't exist
        os.makedirs("data/performance", exist_ok=True)
        
        # Sample performance metrics
        self.sample_metrics = {
            'total_return': 0.35,  # 35% total return
            'annualized_return': 0.18,  # 18% annualized
            'sharpe_ratio': 1.9,
            'max_drawdown': 0.05,  # 5%
            'win_rate': 0.65,  # 65%
            'total_trades': 120,
            'benchmark_return': 0.10,  # 10%
            'alpha': 0.08,
            'beta': 0.85
        }
        
        # Sample backtest results
        dates = pd.date_range(start='2020-01-01', end='2020-12-31')
        self.sample_backtest_results = {
            'portfolio_value': pd.Series(
                np.linspace(10000, 13500, len(dates)), 
                index=dates
            ),
            'benchmark_value': pd.Series(
                np.linspace(10000, 11000, len(dates)),
                index=dates
            ),
            'trades': [
                {'symbol': 'RELIANCE.NS', 'action': 'buy', 'price': 100, 'quantity': 10, 
                 'timestamp': dates[0], 'pnl': 0},
                {'symbol': 'RELIANCE.NS', 'action': 'sell', 'price': 110, 'quantity': 10,
                 'timestamp': dates[10], 'pnl': 100}
            ]
        }
    
    def tearDown(self):
        """Clean up after tests."""
        self.orchestrator.shutdown_system()
    
    def test_performance_target_validation(self):
        """Test validation of performance targets."""
        # Test with all targets met
        result = self.orchestrator._validate_performance_targets(self.sample_metrics)
        self.assertTrue(result)
        
        # Test with annual return below target
        metrics = self.sample_metrics.copy()
        metrics['annualized_return'] = 0.10  # Below 15% target
        result = self.orchestrator._validate_performance_targets(metrics)
        self.assertTrue(result)  # Should still pass with 3/4 targets met
        
        # Test with multiple targets missed
        metrics = self.sample_metrics.copy()
        metrics['annualized_return'] = 0.10  # Below target
        metrics['sharpe_ratio'] = 1.5  # Below target
        result = self.orchestrator._validate_performance_targets(metrics)
        self.assertFalse(result)  # Should fail with only 2/4 targets met
    
    @patch('src.monitoring.performance_tracker.PerformanceTracker.calculate_comprehensive_metrics')
    def test_performance_report_generation(self, mock_metrics):
        """Test performance report generation."""
        # Configure mock
        mock_metrics.return_value = self.sample_metrics
        
        # Generate report
        result = self.orchestrator._generate_performance_report(
            self.sample_metrics, self.sample_backtest_results
        )
        
        # Verify results
        self.assertTrue(result)
        
        # Check if report file was created
        report_files = [f for f in os.listdir("data/performance") if f.startswith("performance_report_")]
        self.assertTrue(len(report_files) > 0)
    
    @patch('src.monitoring.system_health.SystemHealthMonitor.check_system_health')
    def test_final_system_validation(self, mock_health_check):
        """Test final system validation."""
        # Configure mock
        mock_health_check.return_value = True
        
        # Set all system states to True
        self.orchestrator.system_state = {
            "initialized": True,
            "data_collected": True,
            "models_trained": True,
            "backtest_completed": True,
            "ready_for_trading": True
        }
        
        # Run validation
        result = self.orchestrator._perform_final_system_validation()
        
        # Verify results
        self.assertTrue(result)
        
        # Test with missing state
        self.orchestrator.system_state["models_trained"] = False
        result = self.orchestrator._perform_final_system_validation()
        self.assertFalse(result)
        
        # Test with health check failure
        self.orchestrator.system_state["models_trained"] = True
        mock_health_check.return_value = False
        result = self.orchestrator._perform_final_system_validation()
        self.assertFalse(result)
    
    @patch('src.monitoring.system_health.SystemHealthMonitor.get_performance_metrics')
    def test_system_resource_optimization(self, mock_metrics):
        """Test system resource optimization."""
        # Configure mock
        mock_metrics.return_value = MagicMock(
            cpu_usage=30.0,
            memory_usage=40.0,
            disk_usage=50.0,
            network_io={'bytes_sent': 1000, 'bytes_recv': 2000},
            process_count=10,
            uptime_seconds=3600
        )
        
        # Get system health
        health_monitor = SystemHealthMonitor()
        health_summary = health_monitor.get_health_summary()
        
        # Verify health summary
        self.assertEqual(health_summary['overall_status'], 'healthy')
        self.assertIn('performance_metrics', health_summary)
        self.assertIn('cpu_usage', health_summary['performance_metrics'])
        self.assertIn('memory_usage', health_summary['performance_metrics'])
    
    def test_system_state_persistence(self):
        """Test system state persistence."""
        # Set system state
        self.orchestrator.system_state = {
            "initialized": True,
            "data_collected": True,
            "models_trained": True,
            "backtest_completed": True,
            "ready_for_trading": True
        }
        
        # Save state
        self.orchestrator._save_system_state()
        
        # Check if state file was created
        self.assertTrue(os.path.exists("data/system_state.json"))
        
        # Load state in a new orchestrator
        new_orchestrator = TradingSystemOrchestrator(environment="testing")
        new_orchestrator._load_existing_state()
        
        # Verify state was loaded
        self.assertTrue(new_orchestrator.system_state["initialized"])
        self.assertTrue(new_orchestrator.system_state["data_collected"])
        self.assertTrue(new_orchestrator.system_state["models_trained"])
        self.assertTrue(new_orchestrator.system_state["backtest_completed"])
        self.assertTrue(new_orchestrator.system_state["ready_for_trading"])


if __name__ == '__main__':
    unittest.main()