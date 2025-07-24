"""
Integration tests for the Visualization system.
Tests complete visualization pipeline with various chart types and data scenarios.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import os
import shutil

from src.monitoring.visualization import PerformanceVisualizer
from src.monitoring.performance_tracker import PerformanceSnapshot, TradeLog
from src.interfaces.trading_interfaces import TradingSignal, Position


class TestPerformanceVisualizer(unittest.TestCase):
    """Test cases for PerformanceVisualizer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for test outputs
        self.temp_dir = tempfile.mkdtemp()
        self.visualizer = PerformanceVisualizer(output_dir=self.temp_dir)
        
        # Sample portfolio values
        self.portfolio_values = []
        base_date = datetime(2023, 1, 1)
        base_value = 1000000.0
        
        for i in range(100):  # 100 days of data
            date = base_date + timedelta(days=i)
            # Simulate some volatility with trend
            daily_return = np.random.normal(0.001, 0.02)  # 0.1% daily return with 2% volatility
            base_value *= (1 + daily_return)
            
            self.portfolio_values.append({
                'date': date.date(),
                'portfolio_value': base_value,
                'daily_return': daily_return * 100,
                'positions_count': np.random.randint(5, 15)
            })
        
        # Sample benchmark data
        dates = pd.date_range(start='2023-01-01', end='2023-04-10', freq='D')
        self.benchmark_data = pd.DataFrame({
            'Close': 18000 + np.cumsum(np.random.normal(10, 50, len(dates)))
        }, index=dates)
        
        # Sample performance snapshots
        self.performance_snapshots = []
        for i in range(0, 100, 10):  # Every 10 days
            snapshot = PerformanceSnapshot(
                timestamp=base_date + timedelta(days=i),
                portfolio_value=self.portfolio_values[i]['portfolio_value'],
                total_return=(self.portfolio_values[i]['portfolio_value'] / 1000000.0 - 1) * 100,
                daily_return=self.portfolio_values[i]['daily_return'],
                sharpe_ratio=1.5 + np.random.normal(0, 0.3),
                max_drawdown=abs(np.random.normal(3, 2)),
                win_rate=60 + np.random.normal(0, 5),
                total_trades=i + 10,
                open_positions=np.random.randint(5, 15),
                benchmark_return=5.0 + np.random.normal(0, 2),
                alpha=1.0 + np.random.normal(0, 0.5),
                beta=1.0 + np.random.normal(0, 0.2)
            )
            self.performance_snapshots.append(snapshot)
        
        # Sample trade logs
        self.trade_logs = []
        symbols = ['RELIANCE', 'TCS', 'INFY', 'HDFC', 'ICICI']
        
        for i in range(50):  # 50 trades
            symbol = np.random.choice(symbols)
            action = np.random.choice(['buy', 'sell'], p=[0.6, 0.4])
            
            trade_log = TradeLog(
                timestamp=base_date + timedelta(days=np.random.randint(0, 100)),
                symbol=symbol,
                action=action,
                quantity=100.0,
                price=2000 + np.random.normal(0, 200),
                signal_confidence=0.5 + np.random.random() * 0.5,
                lstm_prediction=0.3 + np.random.random() * 0.4,
                dqn_q_values={'buy': np.random.random(), 'sell': np.random.random(), 'hold': np.random.random()},
                sentiment_score=np.random.uniform(-1, 1),
                risk_adjusted_size=0.01 + np.random.random() * 0.02,
                decision_rationale=f"Test trade for {symbol}",
                portfolio_value_before=1000000.0,
                portfolio_value_after=1002000.0
            )
            self.trade_logs.append(trade_log)
        
        # Sample price data for trade signals
        self.price_data = pd.DataFrame({
            'Close': 2500 + np.cumsum(np.random.normal(1, 10, 100)),
            'SMA_50': 2500 + np.cumsum(np.random.normal(0.5, 5, 100)),
            'BB_Upper': 2600 + np.cumsum(np.random.normal(1, 8, 100)),
            'BB_Lower': 2400 + np.cumsum(np.random.normal(1, 8, 100))
        }, index=pd.date_range(start='2023-01-01', periods=100, freq='D'))
    
    def tearDown(self):
        """Clean up test fixtures."""
        # Remove temporary directory
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test PerformanceVisualizer initialization."""
        self.assertTrue(os.path.exists(self.temp_dir))
        self.assertEqual(self.visualizer.output_dir, self.temp_dir)
        self.assertIsNotNone(self.visualizer.logger)
    
    def test_plot_cumulative_returns(self):
        """Test cumulative returns plotting."""
        # Test with portfolio data only
        plot_path = self.visualizer.plot_cumulative_returns(self.portfolio_values)
        
        # Verify plot was created
        self.assertTrue(os.path.exists(plot_path))
        self.assertTrue(plot_path.endswith('.png'))
        
        # Test with benchmark data
        plot_path_with_benchmark = self.visualizer.plot_cumulative_returns(
            self.portfolio_values, self.benchmark_data
        )
        
        self.assertTrue(os.path.exists(plot_path_with_benchmark))
    
    def test_plot_cumulative_returns_empty_data(self):
        """Test cumulative returns plotting with empty data."""
        plot_path = self.visualizer.plot_cumulative_returns([])
        self.assertEqual(plot_path, "")
    
    def test_plot_trade_signals_with_sentiment(self):
        """Test trade signals plotting with sentiment scores."""
        # Filter trades for RELIANCE
        reliance_trades = [trade for trade in self.trade_logs if trade.symbol == 'RELIANCE']
        
        if reliance_trades:  # Only test if we have RELIANCE trades
            plot_path = self.visualizer.plot_trade_signals_with_sentiment(
                self.price_data, reliance_trades, 'RELIANCE'
            )
            
            # Verify plot was created
            self.assertTrue(os.path.exists(plot_path))
            self.assertTrue(plot_path.endswith('.png'))
            self.assertIn('RELIANCE', plot_path)
    
    def test_plot_trade_signals_empty_data(self):
        """Test trade signals plotting with empty data."""
        plot_path = self.visualizer.plot_trade_signals_with_sentiment(
            pd.DataFrame(), [], 'TEST'
        )
        self.assertEqual(plot_path, "")
    
    def test_plot_drawdown_analysis(self):
        """Test drawdown analysis plotting."""
        plot_path = self.visualizer.plot_drawdown_analysis(self.portfolio_values)
        
        # Verify plot was created
        self.assertTrue(os.path.exists(plot_path))
        self.assertTrue(plot_path.endswith('.png'))
        self.assertIn('drawdown', plot_path.lower())
    
    def test_plot_drawdown_analysis_empty_data(self):
        """Test drawdown analysis plotting with empty data."""
        plot_path = self.visualizer.plot_drawdown_analysis([])
        self.assertEqual(plot_path, "")
    
    def test_plot_sharpe_ratio_evolution(self):
        """Test Sharpe ratio evolution plotting."""
        plot_path = self.visualizer.plot_sharpe_ratio_evolution(self.performance_snapshots)
        
        # Verify plot was created
        self.assertTrue(os.path.exists(plot_path))
        self.assertTrue(plot_path.endswith('.png'))
        self.assertIn('sharpe', plot_path.lower())
    
    def test_plot_sharpe_ratio_evolution_empty_data(self):
        """Test Sharpe ratio evolution plotting with empty data."""
        plot_path = self.visualizer.plot_sharpe_ratio_evolution([])
        self.assertEqual(plot_path, "")
    
    def test_create_performance_dashboard(self):
        """Test comprehensive performance dashboard creation."""
        dashboard_path = self.visualizer.create_performance_dashboard(
            self.portfolio_values,
            self.performance_snapshots,
            self.trade_logs,
            self.benchmark_data
        )
        
        # Verify dashboard was created
        self.assertTrue(os.path.exists(dashboard_path))
        self.assertTrue(dashboard_path.endswith('.png'))
        self.assertIn('dashboard', dashboard_path.lower())
    
    def test_create_performance_dashboard_minimal_data(self):
        """Test dashboard creation with minimal data."""
        # Test with minimal data
        minimal_portfolio = [self.portfolio_values[0]]
        minimal_snapshots = [self.performance_snapshots[0]]
        minimal_trades = [self.trade_logs[0]]
        
        dashboard_path = self.visualizer.create_performance_dashboard(
            minimal_portfolio,
            minimal_snapshots,
            minimal_trades
        )
        
        # Should still create dashboard even with minimal data
        self.assertTrue(os.path.exists(dashboard_path))
    
    def test_export_all_visualizations(self):
        """Test exporting all visualization types."""
        export_paths = self.visualizer.export_all_visualizations(
            self.portfolio_values,
            self.performance_snapshots,
            self.trade_logs,
            self.benchmark_data
        )
        
        # Verify multiple visualizations were exported
        self.assertGreater(len(export_paths), 0)
        
        # Check that files exist
        for viz_type, path in export_paths.items():
            self.assertTrue(os.path.exists(path), f"File not found for {viz_type}: {path}")
            self.assertTrue(path.endswith('.png'))
    
    def test_subplot_methods(self):
        """Test individual subplot methods."""
        import matplotlib.pyplot as plt
        
        # Test cumulative returns subplot
        fig, ax = plt.subplots()
        self.visualizer._plot_cumulative_returns_subplot(ax, self.portfolio_values, self.benchmark_data)
        plt.close()
        
        # Test metrics summary subplot
        fig, ax = plt.subplots()
        self.visualizer._plot_metrics_summary_subplot(ax, self.performance_snapshots)
        plt.close()
        
        # Test drawdown subplot
        fig, ax = plt.subplots()
        self.visualizer._plot_drawdown_subplot(ax, self.portfolio_values)
        plt.close()
        
        # Test Sharpe subplot
        fig, ax = plt.subplots()
        self.visualizer._plot_sharpe_subplot(ax, self.performance_snapshots)
        plt.close()
        
        # Test trade distribution subplot
        fig, ax = plt.subplots()
        self.visualizer._plot_trade_distribution_subplot(ax, self.trade_logs)
        plt.close()
        
        # Test monthly returns heatmap subplot
        fig, ax = plt.subplots()
        self.visualizer._plot_monthly_returns_heatmap_subplot(ax, self.portfolio_values)
        plt.close()
    
    def test_subplot_methods_empty_data(self):
        """Test subplot methods with empty data."""
        import matplotlib.pyplot as plt
        
        # All subplot methods should handle empty data gracefully
        fig, ax = plt.subplots()
        self.visualizer._plot_cumulative_returns_subplot(ax, [], None)
        plt.close()
        
        fig, ax = plt.subplots()
        self.visualizer._plot_metrics_summary_subplot(ax, [])
        plt.close()
        
        fig, ax = plt.subplots()
        self.visualizer._plot_drawdown_subplot(ax, [])
        plt.close()
        
        fig, ax = plt.subplots()
        self.visualizer._plot_sharpe_subplot(ax, [])
        plt.close()
        
        fig, ax = plt.subplots()
        self.visualizer._plot_trade_distribution_subplot(ax, [])
        plt.close()
        
        fig, ax = plt.subplots()
        self.visualizer._plot_monthly_returns_heatmap_subplot(ax, [])
        plt.close()
    
    def test_custom_save_paths(self):
        """Test using custom save paths."""
        custom_path = os.path.join(self.temp_dir, "custom_cumulative_returns.png")
        
        plot_path = self.visualizer.plot_cumulative_returns(
            self.portfolio_values, save_path=custom_path
        )
        
        # Verify custom path was used
        self.assertEqual(plot_path, custom_path)
        self.assertTrue(os.path.exists(custom_path))
    
    def test_file_naming_conventions(self):
        """Test file naming conventions."""
        # Test that files have proper timestamps and naming
        plot_path = self.visualizer.plot_cumulative_returns(self.portfolio_values)
        
        filename = os.path.basename(plot_path)
        self.assertTrue(filename.startswith('cumulative_returns_'))
        self.assertTrue(filename.endswith('.png'))
        
        # Check timestamp format (YYYYMMDD_HHMMSS_ffffff)
        timestamp_part = filename.replace('cumulative_returns_', '').replace('.png', '')
        self.assertEqual(len(timestamp_part), 22)  # YYYYMMDD_HHMMSS_ffffff format
    
    @patch('src.monitoring.visualization.get_logger')
    def test_error_handling(self, mock_logger):
        """Test error handling in visualization methods."""
        mock_logger.return_value = Mock()
        
        # Test with corrupted data
        corrupted_data = [{'invalid': 'data'}]
        
        plot_path = self.visualizer.plot_cumulative_returns(corrupted_data)
        self.assertEqual(plot_path, "")
        
        plot_path = self.visualizer.plot_drawdown_analysis(corrupted_data)
        self.assertEqual(plot_path, "")
    
    def test_large_dataset_handling(self):
        """Test handling of large datasets."""
        # Create larger dataset
        large_portfolio_values = []
        base_date = datetime(2020, 1, 1)
        base_value = 1000000.0
        
        for i in range(1000):  # 1000 days of data
            date = base_date + timedelta(days=i)
            daily_return = np.random.normal(0.001, 0.02)
            base_value *= (1 + daily_return)
            
            large_portfolio_values.append({
                'date': date.date(),
                'portfolio_value': base_value,
                'daily_return': daily_return * 100,
                'positions_count': np.random.randint(5, 15)
            })
        
        # Should handle large dataset without errors
        plot_path = self.visualizer.plot_cumulative_returns(large_portfolio_values)
        self.assertTrue(os.path.exists(plot_path))
        
        plot_path = self.visualizer.plot_drawdown_analysis(large_portfolio_values)
        self.assertTrue(os.path.exists(plot_path))
    
    def test_different_date_ranges(self):
        """Test visualization with different date ranges."""
        # Test with very short date range (1 week)
        short_portfolio = self.portfolio_values[:7]
        
        plot_path = self.visualizer.plot_cumulative_returns(short_portfolio)
        self.assertTrue(os.path.exists(plot_path))
        
        # Test with longer date range (already tested in other methods)
        # This ensures the visualizer works across different time scales
    
    def test_extreme_values(self):
        """Test visualization with extreme values."""
        # Create data with extreme values
        extreme_portfolio = [
            {'date': datetime(2023, 1, 1).date(), 'portfolio_value': 1000000.0, 'daily_return': 0.0},
            {'date': datetime(2023, 1, 2).date(), 'portfolio_value': 2000000.0, 'daily_return': 100.0},  # 100% gain
            {'date': datetime(2023, 1, 3).date(), 'portfolio_value': 500000.0, 'daily_return': -75.0},   # 75% loss
            {'date': datetime(2023, 1, 4).date(), 'portfolio_value': 1500000.0, 'daily_return': 200.0},  # 200% gain
        ]
        
        # Should handle extreme values without crashing
        plot_path = self.visualizer.plot_cumulative_returns(extreme_portfolio)
        self.assertTrue(os.path.exists(plot_path))
        
        plot_path = self.visualizer.plot_drawdown_analysis(extreme_portfolio)
        self.assertTrue(os.path.exists(plot_path))


class TestVisualizationIntegration(unittest.TestCase):
    """Integration tests for complete visualization pipeline."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.visualizer = PerformanceVisualizer(output_dir=self.temp_dir)
        
        # Create comprehensive test data
        self._create_comprehensive_test_data()
    
    def tearDown(self):
        """Clean up integration test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def _create_comprehensive_test_data(self):
        """Create comprehensive test data for integration tests."""
        # Create 1 year of daily data
        self.portfolio_values = []
        self.performance_snapshots = []
        self.trade_logs = []
        
        base_date = datetime(2023, 1, 1)
        base_value = 1000000.0
        
        # Generate daily portfolio data
        for i in range(365):
            date = base_date + timedelta(days=i)
            daily_return = np.random.normal(0.0008, 0.015)  # Slightly positive expected return
            base_value *= (1 + daily_return)
            
            self.portfolio_values.append({
                'date': date.date(),
                'portfolio_value': base_value,
                'daily_return': daily_return * 100,
                'positions_count': np.random.randint(8, 20)
            })
            
            # Create weekly performance snapshots
            if i % 7 == 0:
                snapshot = PerformanceSnapshot(
                    timestamp=date,
                    portfolio_value=base_value,
                    total_return=(base_value / 1000000.0 - 1) * 100,
                    daily_return=daily_return * 100,
                    sharpe_ratio=max(0.5, 1.8 + np.random.normal(0, 0.4)),
                    max_drawdown=abs(np.random.normal(4, 3)),
                    win_rate=max(45, min(75, 62 + np.random.normal(0, 8))),
                    total_trades=i // 3 + 10,
                    open_positions=np.random.randint(8, 20),
                    benchmark_return=8.0 + np.random.normal(0, 3),
                    alpha=np.random.normal(2, 1),
                    beta=0.8 + np.random.normal(0, 0.3)
                )
                self.performance_snapshots.append(snapshot)
        
        # Generate realistic trade logs
        symbols = ['RELIANCE', 'TCS', 'INFY', 'HDFC', 'ICICI', 'WIPRO', 'LT', 'MARUTI', 'BAJFINANCE', 'HDFCBANK']
        
        for i in range(200):  # 200 trades over the year
            symbol = np.random.choice(symbols)
            action = np.random.choice(['buy', 'sell'], p=[0.55, 0.45])  # Slightly more buys
            
            trade_date = base_date + timedelta(days=np.random.randint(0, 365))
            
            trade_log = TradeLog(
                timestamp=trade_date,
                symbol=symbol,
                action=action,
                quantity=np.random.randint(50, 200),
                price=1500 + np.random.normal(0, 500),
                signal_confidence=0.4 + np.random.random() * 0.6,
                lstm_prediction=0.2 + np.random.random() * 0.6,
                dqn_q_values={
                    'buy': np.random.random(),
                    'sell': np.random.random(),
                    'hold': np.random.random()
                },
                sentiment_score=np.random.uniform(-0.8, 0.8),
                risk_adjusted_size=0.005 + np.random.random() * 0.025,
                decision_rationale=f"Algorithmic decision for {symbol} based on technical and sentiment analysis",
                portfolio_value_before=base_value * 0.98,
                portfolio_value_after=base_value
            )
            self.trade_logs.append(trade_log)
        
        # Create benchmark data
        benchmark_dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        benchmark_values = 18000 + np.cumsum(np.random.normal(8, 40, len(benchmark_dates)))
        self.benchmark_data = pd.DataFrame({
            'Close': benchmark_values
        }, index=benchmark_dates)
    
    def test_complete_visualization_pipeline(self):
        """Test complete visualization pipeline with comprehensive data."""
        # Test individual visualizations
        cumulative_path = self.visualizer.plot_cumulative_returns(
            self.portfolio_values, self.benchmark_data
        )
        self.assertTrue(os.path.exists(cumulative_path))
        
        drawdown_path = self.visualizer.plot_drawdown_analysis(self.portfolio_values)
        self.assertTrue(os.path.exists(drawdown_path))
        
        sharpe_path = self.visualizer.plot_sharpe_ratio_evolution(self.performance_snapshots)
        self.assertTrue(os.path.exists(sharpe_path))
        
        # Test trade signals for multiple symbols
        symbols_with_trades = {}
        for trade in self.trade_logs:
            if trade.symbol not in symbols_with_trades:
                symbols_with_trades[trade.symbol] = []
            symbols_with_trades[trade.symbol].append(trade)
        
        # Test trade signals for top 3 symbols
        top_symbols = sorted(symbols_with_trades.items(), key=lambda x: len(x[1]), reverse=True)[:3]
        
        for symbol, trades in top_symbols:
            # Create mock price data for the symbol
            price_data = pd.DataFrame({
                'Close': 2000 + np.cumsum(np.random.normal(2, 15, 365)),
                'SMA_50': 2000 + np.cumsum(np.random.normal(1, 10, 365)),
                'BB_Upper': 2100 + np.cumsum(np.random.normal(2, 12, 365)),
                'BB_Lower': 1900 + np.cumsum(np.random.normal(2, 12, 365))
            }, index=pd.date_range(start='2023-01-01', periods=365, freq='D'))
            
            signals_path = self.visualizer.plot_trade_signals_with_sentiment(
                price_data, trades, symbol
            )
            self.assertTrue(os.path.exists(signals_path))
        
        # Test comprehensive dashboard
        dashboard_path = self.visualizer.create_performance_dashboard(
            self.portfolio_values,
            self.performance_snapshots,
            self.trade_logs,
            self.benchmark_data
        )
        self.assertTrue(os.path.exists(dashboard_path))
        
        # Test export all visualizations
        export_paths = self.visualizer.export_all_visualizations(
            self.portfolio_values,
            self.performance_snapshots,
            self.trade_logs,
            self.benchmark_data
        )
        
        # Verify all exports
        self.assertGreaterEqual(len(export_paths), 4)  # At least 4 visualization types
        for viz_type, path in export_paths.items():
            self.assertTrue(os.path.exists(path))
    
    def test_visualization_consistency(self):
        """Test that visualizations are consistent across multiple runs."""
        # Run visualization twice with same data
        path1 = self.visualizer.plot_cumulative_returns(self.portfolio_values)
        path2 = self.visualizer.plot_cumulative_returns(self.portfolio_values)
        
        # Both should be created successfully
        self.assertTrue(os.path.exists(path1))
        self.assertTrue(os.path.exists(path2))
        
        # Files should have different names (due to timestamps)
        self.assertNotEqual(path1, path2)
    
    def test_performance_with_realistic_data_volumes(self):
        """Test performance with realistic data volumes."""
        import time
        
        # Measure time for dashboard creation
        start_time = time.time()
        
        dashboard_path = self.visualizer.create_performance_dashboard(
            self.portfolio_values,
            self.performance_snapshots,
            self.trade_logs,
            self.benchmark_data
        )
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Should complete within reasonable time (less than 30 seconds)
        self.assertLess(execution_time, 30.0)
        self.assertTrue(os.path.exists(dashboard_path))
    
    def test_memory_usage_with_large_datasets(self):
        """Test memory usage doesn't grow excessively with large datasets."""
        import psutil
        import os as os_module
        
        # Get initial memory usage
        process = psutil.Process(os_module.getpid())
        initial_memory = process.memory_info().rss
        
        # Create and process large dataset
        large_portfolio = self.portfolio_values * 10  # 10x the data
        large_snapshots = self.performance_snapshots * 10
        large_trades = self.trade_logs * 10
        
        # Create multiple visualizations
        self.visualizer.plot_cumulative_returns(large_portfolio, self.benchmark_data)
        self.visualizer.plot_drawdown_analysis(large_portfolio)
        self.visualizer.plot_sharpe_ratio_evolution(large_snapshots)
        
        # Check memory usage hasn't grown excessively
        final_memory = process.memory_info().rss
        memory_growth = final_memory - initial_memory
        
        # Memory growth should be reasonable (less than 500MB)
        self.assertLess(memory_growth, 500 * 1024 * 1024)


if __name__ == '__main__':
    unittest.main()