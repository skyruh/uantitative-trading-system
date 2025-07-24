"""
Integration tests for the complete trading system workflow.
Tests end-to-end functionality from data collection to performance reporting.
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


class TestSystemIntegration(unittest.TestCase):
    """Test complete system integration and workflow."""
    
    def setUp(self):
        """Set up test environment."""
        # Use testing environment
        self.orchestrator = TradingSystemOrchestrator(environment="testing")
        
        # Create test directories if they don't exist
        os.makedirs("data/stocks", exist_ok=True)
        os.makedirs("data/news", exist_ok=True)
        os.makedirs("data/indicators", exist_ok=True)
        os.makedirs("models/checkpoints", exist_ok=True)
        
        # Mock stock symbols for testing
        self.test_symbols = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS"]
        
        # Create sample data for testing
        self._create_sample_data()
    
    def _create_sample_data(self):
        """Create sample data for testing."""
        # Sample stock data
        for symbol in self.test_symbols:
            dates = pd.date_range(start='2020-01-01', end='2020-01-10')
            data = {
                'Open': np.random.uniform(100, 200, len(dates)),
                'High': np.random.uniform(110, 210, len(dates)),
                'Low': np.random.uniform(90, 190, len(dates)),
                'Close': np.random.uniform(100, 200, len(dates)),
                'Volume': np.random.randint(1000, 10000, len(dates))
            }
            df = pd.DataFrame(data, index=dates)
            
            # Ensure High is highest and Low is lowest
            for i in range(len(df)):
                values = [df.iloc[i]['Open'], df.iloc[i]['Close']]
                df.iloc[i, df.columns.get_loc('High')] = max(values) + 5
                df.iloc[i, df.columns.get_loc('Low')] = min(values) - 5
            
            # Save to CSV
            os.makedirs(f"data/stocks", exist_ok=True)
            df.to_csv(f"data/stocks/{symbol}.csv")
    
    def tearDown(self):
        """Clean up after tests."""
        self.orchestrator.shutdown_system()
    
    @patch('src.data.stock_data_fetcher.StockDataFetcher.fetch_all_stocks')
    @patch('src.data.news_data_fetcher.NewsDataFetcher.fetch_news_for_stocks')
    @patch('src.data.data_processor.DataProcessor.process_all_data')
    @patch('src.data.feature_builder.FeatureBuilder.build_features_for_all_stocks')
    @patch('src.data.sentiment_integration.SentimentIntegration.integrate_sentiment_for_all_stocks')
    def test_data_collection_workflow(self, mock_sentiment, mock_features, mock_process, 
                                     mock_news, mock_stocks):
        """Test the data collection workflow."""
        # Configure mocks
        mock_stocks.return_value = True
        mock_news.return_value = True
        mock_process.return_value = True
        mock_features.return_value = True
        mock_sentiment.return_value = True
        
        # Run the workflow
        with patch('src.config.settings.config.get_stock_symbols', return_value=self.test_symbols):
            result = self.orchestrator.run_data_collection_workflow()
        
        # Verify results
        self.assertTrue(result)
        self.assertTrue(self.orchestrator.system_state["data_collected"])
        
        # Verify all components were called
        mock_stocks.assert_called_once()
        mock_news.assert_called_once()
        mock_process.assert_called_once()
        mock_features.assert_called_once()
        mock_sentiment.assert_called_once()
    
    @patch('src.models.lstm_trainer.LSTMTrainer.train_models_for_all_stocks')
    @patch('src.models.dqn_trainer.DQNTrainer.train_agent')
    def test_model_training_workflow(self, mock_dqn, mock_lstm):
        """Test the model training workflow."""
        # Configure mocks
        mock_lstm.return_value = True
        mock_dqn.return_value = True
        
        # Set prerequisite state
        self.orchestrator.system_state["data_collected"] = True
        
        # Run the workflow
        with patch('src.config.settings.config.get_stock_symbols', return_value=self.test_symbols):
            result = self.orchestrator.run_model_training_workflow()
        
        # Verify results
        self.assertTrue(result)
        self.assertTrue(self.orchestrator.system_state["models_trained"])
        
        # Verify all components were called
        mock_lstm.assert_called_once_with(self.test_symbols)
        mock_dqn.assert_called_once_with(self.test_symbols)
    
    @patch('src.backtesting.backtest_engine.BacktestEngine.initialize_backtest')
    @patch('src.backtesting.backtest_engine.BacktestEngine.run_backtest')
    @patch('src.monitoring.performance_tracker.PerformanceTracker.calculate_comprehensive_metrics')
    def test_backtesting_workflow(self, mock_metrics, mock_run, mock_init):
        """Test the backtesting workflow."""
        # Configure mocks
        mock_init.return_value = True
        
        # Mock backtest results
        mock_backtest_results = {
            'portfolio_value': pd.Series([10000, 10100, 10200], 
                                        index=pd.date_range('2020-01-01', periods=3)),
            'trades': [
                {'symbol': 'RELIANCE.NS', 'action': 'buy', 'price': 100, 'quantity': 10},
                {'symbol': 'RELIANCE.NS', 'action': 'sell', 'price': 110, 'quantity': 10}
            ]
        }
        mock_run.return_value = mock_backtest_results
        
        # Mock performance metrics
        mock_metrics.return_value = {
            'annualized_return': 0.18,  # 18%
            'sharpe_ratio': 1.9,
            'max_drawdown': 0.05,  # 5%
            'win_rate': 0.65  # 65%
        }
        
        # Set prerequisite state
        self.orchestrator.system_state["data_collected"] = True
        self.orchestrator.system_state["models_trained"] = True
        
        # Run the workflow
        with patch('src.config.settings.config.get_stock_symbols', return_value=self.test_symbols):
            result = self.orchestrator.run_backtesting_workflow()
        
        # Verify results
        self.assertTrue(result)
        self.assertTrue(self.orchestrator.system_state["backtest_completed"])
        self.assertTrue(self.orchestrator.system_state["ready_for_trading"])
        
        # Verify all components were called
        mock_init.assert_called_once()
        mock_run.assert_called_once()
        mock_metrics.assert_called_once_with(mock_backtest_results)
    
    @patch('src.system.trading_system_orchestrator.TradingSystemOrchestrator.startup_system')
    @patch('src.system.trading_system_orchestrator.TradingSystemOrchestrator.run_data_collection_workflow')
    @patch('src.system.trading_system_orchestrator.TradingSystemOrchestrator.run_model_training_workflow')
    @patch('src.system.trading_system_orchestrator.TradingSystemOrchestrator.run_backtesting_workflow')
    def test_complete_system_workflow(self, mock_backtest, mock_train, mock_collect, mock_startup):
        """Test the complete end-to-end system workflow."""
        # Configure mocks
        mock_startup.return_value = True
        mock_collect.return_value = True
        mock_train.return_value = True
        mock_backtest.return_value = True
        
        # Run the workflow
        result = self.orchestrator.run_complete_system_workflow()
        
        # Verify results
        self.assertTrue(result)
        
        # Verify all components were called in sequence
        mock_startup.assert_called_once()
        mock_collect.assert_called_once()
        mock_train.assert_called_once()
        mock_backtest.assert_called_once()
    
    def test_system_startup(self):
        """Test system startup and validation."""
        # Run startup
        with patch('src.monitoring.system_health.SystemHealthMonitor.check_system_health', 
                  return_value=True):
            result = self.orchestrator.startup_system()
        
        # Verify results
        self.assertTrue(result)
        self.assertTrue(self.orchestrator.system_state["initialized"])
        
        # Verify directories were created
        self.assertTrue(os.path.exists(config.data.data_directory))
        self.assertTrue(os.path.exists(config.model.model_save_directory))
        self.assertTrue(os.path.exists(config.logging.log_directory))
    
    def test_get_system_status(self):
        """Test getting system status."""
        # Get status
        status = self.orchestrator.get_system_status()
        
        # Verify status structure
        self.assertIn("system_state", status)
        self.assertIn("environment", status)
        self.assertIn("health_status", status)
        self.assertIn("timestamp", status)
        
        # Verify environment
        self.assertEqual(status["environment"], "testing")
    
    def test_error_handling(self):
        """Test error handling in workflows."""
        # Test data collection error handling
        with patch('src.data.stock_data_fetcher.StockDataFetcher.fetch_all_stocks', 
                  side_effect=Exception("Test error")):
            result = self.orchestrator.run_data_collection_workflow()
            self.assertFalse(result)
            self.assertFalse(self.orchestrator.system_state["data_collected"])
        
        # Test model training error handling
        self.orchestrator.system_state["data_collected"] = True
        with patch('src.models.lstm_trainer.LSTMTrainer.train_models_for_all_stocks', 
                  side_effect=Exception("Test error")):
            result = self.orchestrator.run_model_training_workflow()
            self.assertFalse(result)
            self.assertFalse(self.orchestrator.system_state["models_trained"])


if __name__ == '__main__':
    unittest.main()