"""
Unit tests for the Performance Tracker.
Tests real-time performance tracking, trade logging, and benchmark comparison.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import os
import json

from src.monitoring.performance_tracker import PerformanceTracker, TradeLog, PerformanceSnapshot
from src.interfaces.trading_interfaces import TradingSignal, Position, PerformanceMetrics


class TestPerformanceTracker(unittest.TestCase):
    """Test cases for PerformanceTracker class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.initial_capital = 1000000.0
        self.tracker = PerformanceTracker(initial_capital=self.initial_capital)
        
        # Sample trading signal
        self.sample_signal = TradingSignal(
            symbol="RELIANCE",
            timestamp=datetime.now(),
            action="buy",
            confidence=0.85,
            lstm_prediction=0.75,
            dqn_q_values={"buy": 0.8, "sell": 0.1, "hold": 0.1},
            sentiment_score=0.6,
            risk_adjusted_size=0.02
        )
        
        # Sample position
        self.sample_position = Position(
            symbol="RELIANCE",
            entry_date=datetime.now(),
            entry_price=2500.0,
            quantity=100,
            stop_loss_price=2375.0,
            current_value=250000.0,
            unrealized_pnl=0.0,
            status="open"
        )
        
        # Sample benchmark data
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        self.benchmark_data = pd.DataFrame({
            'Close': np.random.normal(18000, 500, len(dates))
        }, index=dates)
    
    def test_initialization(self):
        """Test PerformanceTracker initialization."""
        self.assertEqual(self.tracker.initial_capital, self.initial_capital)
        self.assertEqual(self.tracker.current_portfolio_value, self.initial_capital)
        self.assertEqual(len(self.tracker.trade_logs), 0)
        self.assertEqual(len(self.tracker.performance_snapshots), 0)
        self.assertIsNotNone(self.tracker.performance_calculator)
    
    def test_log_trade(self):
        """Test trade logging functionality."""
        decision_rationale = "Strong LSTM signal with positive sentiment"
        portfolio_before = 1000000.0
        portfolio_after = 1002500.0
        
        self.tracker.log_trade(
            signal=self.sample_signal,
            position=self.sample_position,
            decision_rationale=decision_rationale,
            portfolio_value_before=portfolio_before,
            portfolio_value_after=portfolio_after
        )
        
        # Verify trade was logged
        self.assertEqual(len(self.tracker.trade_logs), 1)
        
        trade_log = self.tracker.trade_logs[0]
        self.assertEqual(trade_log.symbol, "RELIANCE")
        self.assertEqual(trade_log.action, "buy")
        self.assertEqual(trade_log.quantity, 100)
        self.assertEqual(trade_log.price, 2500.0)
        self.assertEqual(trade_log.decision_rationale, decision_rationale)
        self.assertEqual(trade_log.portfolio_value_before, portfolio_before)
        self.assertEqual(trade_log.portfolio_value_after, portfolio_after)
    
    def test_update_portfolio_value(self):
        """Test portfolio value updates."""
        new_value = 1050000.0
        positions = [self.sample_position]
        
        self.tracker.update_portfolio_value(new_value, positions)
        
        # Verify portfolio value updated
        self.assertEqual(self.tracker.current_portfolio_value, new_value)
        self.assertEqual(len(self.tracker.current_positions), 1)
        self.assertEqual(len(self.tracker.daily_portfolio_values), 1)
        
        # Verify daily portfolio record
        record = self.tracker.daily_portfolio_values[0]
        self.assertEqual(record['portfolio_value'], new_value)
        self.assertEqual(record['positions_count'], 1)
    
    def test_calculate_real_time_metrics(self):
        """Test real-time metrics calculation."""
        # Add some portfolio data
        self.tracker.daily_portfolio_values = [
            {'date': datetime.now().date() - timedelta(days=1), 'portfolio_value': 1000000.0},
            {'date': datetime.now().date(), 'portfolio_value': 1050000.0}
        ]
        
        metrics = self.tracker.calculate_real_time_metrics()
        
        # Verify metrics structure
        self.assertIsInstance(metrics, PerformanceMetrics)
        self.assertIsInstance(metrics.total_return, float)
        self.assertIsInstance(metrics.sharpe_ratio, float)
        self.assertIsInstance(metrics.max_drawdown, float)
    
    def test_create_performance_snapshot(self):
        """Test performance snapshot creation."""
        # Add portfolio data
        self.tracker.daily_portfolio_values = [
            {'date': datetime.now().date() - timedelta(days=1), 'portfolio_value': 1000000.0},
            {'date': datetime.now().date(), 'portfolio_value': 1050000.0}
        ]
        self.tracker.current_portfolio_value = 1050000.0
        
        snapshot = self.tracker.create_performance_snapshot()
        
        # Verify snapshot
        self.assertIsInstance(snapshot, PerformanceSnapshot)
        self.assertEqual(snapshot.portfolio_value, 1050000.0)
        self.assertIsInstance(snapshot.timestamp, datetime)
        self.assertEqual(len(self.tracker.performance_snapshots), 1)
    
    def test_load_benchmark_data(self):
        """Test benchmark data loading."""
        self.tracker.load_benchmark_data(self.benchmark_data)
        
        # Verify benchmark data loaded
        self.assertIsNotNone(self.tracker.benchmark_data)
        self.assertEqual(len(self.tracker.benchmark_data), len(self.benchmark_data))
    
    def test_compare_to_benchmark(self):
        """Test benchmark comparison."""
        # Set up data
        self.tracker.load_benchmark_data(self.benchmark_data)
        self.tracker.daily_portfolio_values = [
            {'date': datetime(2023, 1, 1).date(), 'portfolio_value': 1000000.0},
            {'date': datetime(2023, 6, 1).date(), 'portfolio_value': 1100000.0},
            {'date': datetime(2023, 12, 31).date(), 'portfolio_value': 1200000.0}
        ]
        
        comparison = self.tracker.compare_to_benchmark()
        
        # Verify comparison results
        self.assertIn('portfolio_return', comparison)
        self.assertIn('benchmark_return', comparison)
        self.assertIn('excess_return', comparison)
        self.assertIn('alpha', comparison)
        self.assertIn('beta', comparison)
        self.assertIn('outperformance', comparison)
    
    def test_get_performance_dashboard_data(self):
        """Test dashboard data generation."""
        # Set up some data
        self.tracker.daily_portfolio_values = [
            {'date': datetime.now().date(), 'portfolio_value': 1050000.0}
        ]
        self.tracker.current_portfolio_value = 1050000.0
        
        dashboard_data = self.tracker.get_performance_dashboard_data()
        
        # Verify dashboard structure
        self.assertIn('current_metrics', dashboard_data)
        self.assertIn('benchmark_comparison', dashboard_data)
        self.assertIn('recent_trades', dashboard_data)
        self.assertIn('performance_alerts', dashboard_data)
        self.assertIn('timestamp', dashboard_data)
        
        # Verify current metrics format
        current_metrics = dashboard_data['current_metrics']
        self.assertIn('portfolio_value', current_metrics)
        self.assertIn('total_return', current_metrics)
        self.assertIn('sharpe_ratio', current_metrics)
    
    def test_export_performance_data(self):
        """Test performance data export."""
        # Add some test data
        self.tracker.log_trade(
            self.sample_signal, self.sample_position,
            "Test trade", 1000000.0, 1002500.0
        )
        self.tracker.daily_portfolio_values = [
            {'date': datetime.now().date(), 'portfolio_value': 1050000.0}
        ]
        
        # Export to temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as tmp_file:
            filepath = self.tracker.export_performance_data(tmp_file.name)
        
        try:
            # Verify file was created
            self.assertTrue(os.path.exists(filepath))
            
            # Verify file content
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            self.assertIn('metadata', data)
            self.assertIn('trade_logs', data)
            self.assertIn('performance_snapshots', data)
            self.assertIn('daily_portfolio_values', data)
            
        finally:
            # Clean up
            if os.path.exists(filepath):
                os.unlink(filepath)
    
    def test_performance_alerts(self):
        """Test performance alert system."""
        # Create snapshot with high drawdown
        snapshot = PerformanceSnapshot(
            timestamp=datetime.now(),
            portfolio_value=900000.0,  # 10% drawdown
            total_return=-10.0,
            daily_return=-2.0,
            sharpe_ratio=0.5,  # Low Sharpe ratio
            max_drawdown=10.0,  # High drawdown
            win_rate=45.0,  # Low win rate
            total_trades=10,
            open_positions=3,
            benchmark_return=-5.0,
            alpha=-2.0,
            beta=1.2
        )
        
        self.tracker.performance_snapshots.append(snapshot)
        
        # Get active alerts
        alerts = self.tracker._get_active_alerts()
        
        # Should have multiple alerts
        self.assertGreater(len(alerts), 0)
        self.assertTrue(any("drawdown" in alert.lower() for alert in alerts))
        self.assertTrue(any("sharpe" in alert.lower() for alert in alerts))
        self.assertTrue(any("win rate" in alert.lower() for alert in alerts))
    
    def test_daily_return_calculation(self):
        """Test daily return calculation."""
        # Test with no previous data
        daily_return = self.tracker._calculate_daily_return(1050000.0)
        self.assertEqual(daily_return, 0.0)
        
        # Add previous day data
        self.tracker.daily_portfolio_values = [
            {'date': datetime.now().date() - timedelta(days=1), 'portfolio_value': 1000000.0}
        ]
        
        # Test with previous data
        daily_return = self.tracker._calculate_daily_return(1050000.0)
        self.assertAlmostEqual(daily_return, 5.0, places=6)  # 5% return
    
    def test_recent_trades_retrieval(self):
        """Test recent trades retrieval."""
        # Add multiple trades
        for i in range(15):
            signal = TradingSignal(
                symbol=f"STOCK{i}",
                timestamp=datetime.now() - timedelta(minutes=i),
                action="buy",
                confidence=0.8,
                lstm_prediction=0.7,
                dqn_q_values={"buy": 0.8, "sell": 0.1, "hold": 0.1},
                sentiment_score=0.5,
                risk_adjusted_size=0.02
            )
            
            position = Position(
                symbol=f"STOCK{i}",
                entry_date=datetime.now(),
                entry_price=1000.0 + i,
                quantity=100,
                stop_loss_price=950.0 + i,
                current_value=100000.0,
                unrealized_pnl=0.0,
                status="open"
            )
            
            self.tracker.log_trade(signal, position, f"Trade {i}", 1000000.0, 1000000.0)
        
        # Get recent trades (limit 10)
        recent_trades = self.tracker._get_recent_trades(limit=10)
        
        # Should return only 10 most recent trades
        self.assertEqual(len(recent_trades), 10)
        
        # Should be in chronological order (most recent first due to how we added them)
        symbols = [trade['symbol'] for trade in recent_trades]
        expected_symbols = [f"STOCK{i}" for i in range(5, 15)]  # Last 10 trades
        self.assertEqual(symbols, expected_symbols)
    
    @patch('src.monitoring.performance_tracker.get_logger')
    def test_error_handling(self, mock_logger):
        """Test error handling in various methods."""
        mock_logger.return_value = Mock()
        
        # Test with invalid data
        tracker = PerformanceTracker()
        
        # Test calculate_real_time_metrics with no data
        metrics = tracker.calculate_real_time_metrics()
        self.assertEqual(metrics.total_return, 0.0)
        
        # Test compare_to_benchmark with no data
        comparison = tracker.compare_to_benchmark()
        self.assertIn('error', comparison)
    
    def test_datetime_serialization(self):
        """Test datetime object serialization."""
        test_obj = {
            'timestamp': datetime.now(),
            'nested': {
                'date': datetime.now().date(),
                'list': [datetime.now(), 'string', 123]
            }
        }
        
        serialized = self.tracker._serialize_datetime_objects(test_obj)
        
        # Verify datetime objects are converted to strings
        self.assertIsInstance(serialized['timestamp'], str)
        self.assertIsInstance(serialized['nested']['list'][0], str)
        self.assertEqual(serialized['nested']['list'][1], 'string')  # Non-datetime unchanged
        self.assertEqual(serialized['nested']['list'][2], 123)  # Non-datetime unchanged


class TestTradeLog(unittest.TestCase):
    """Test cases for TradeLog dataclass."""
    
    def test_trade_log_creation(self):
        """Test TradeLog creation and attributes."""
        trade_log = TradeLog(
            timestamp=datetime.now(),
            symbol="RELIANCE",
            action="buy",
            quantity=100.0,
            price=2500.0,
            signal_confidence=0.85,
            lstm_prediction=0.75,
            dqn_q_values={"buy": 0.8, "sell": 0.1, "hold": 0.1},
            sentiment_score=0.6,
            risk_adjusted_size=0.02,
            decision_rationale="Strong technical signal",
            portfolio_value_before=1000000.0,
            portfolio_value_after=1002500.0,
            position_id="RELIANCE_20231201_143000"
        )
        
        # Verify all attributes
        self.assertEqual(trade_log.symbol, "RELIANCE")
        self.assertEqual(trade_log.action, "buy")
        self.assertEqual(trade_log.quantity, 100.0)
        self.assertEqual(trade_log.price, 2500.0)
        self.assertEqual(trade_log.signal_confidence, 0.85)
        self.assertEqual(trade_log.decision_rationale, "Strong technical signal")


class TestPerformanceSnapshot(unittest.TestCase):
    """Test cases for PerformanceSnapshot dataclass."""
    
    def test_performance_snapshot_creation(self):
        """Test PerformanceSnapshot creation and attributes."""
        snapshot = PerformanceSnapshot(
            timestamp=datetime.now(),
            portfolio_value=1050000.0,
            total_return=5.0,
            daily_return=1.2,
            sharpe_ratio=1.8,
            max_drawdown=3.5,
            win_rate=62.5,
            total_trades=25,
            open_positions=8,
            benchmark_return=4.2,
            alpha=0.8,
            beta=1.1
        )
        
        # Verify all attributes
        self.assertEqual(snapshot.portfolio_value, 1050000.0)
        self.assertEqual(snapshot.total_return, 5.0)
        self.assertEqual(snapshot.daily_return, 1.2)
        self.assertEqual(snapshot.sharpe_ratio, 1.8)
        self.assertEqual(snapshot.max_drawdown, 3.5)
        self.assertEqual(snapshot.win_rate, 62.5)
        self.assertEqual(snapshot.total_trades, 25)
        self.assertEqual(snapshot.open_positions, 8)


if __name__ == '__main__':
    unittest.main()