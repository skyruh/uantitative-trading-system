"""
Unit tests for logging utilities.
Tests comprehensive logging functionality including structured logging,
error handling, and log rotation.
"""

import unittest
import tempfile
import shutil
import json
import logging
import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
from datetime import datetime

from src.utils.logging_utils import (
    TradingSystemLogger,
    StructuredFormatter,
    get_logger,
    log_system_startup,
    log_system_shutdown,
    log_error_with_context,
    log_performance_metrics,
    log_trading_decision,
    log_model_prediction,
    log_risk_management_action,
    log_position_update,
    log_system_health,
    log_performance_alert,
    log_data_quality_issue,
    create_log_archiver
)


class TestStructuredFormatter(unittest.TestCase):
    """Test structured JSON formatter."""
    
    def setUp(self):
        self.formatter = StructuredFormatter()
    
    def test_basic_formatting(self):
        """Test basic log record formatting."""
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        formatted = self.formatter.format(record)
        log_data = json.loads(formatted)
        
        self.assertEqual(log_data['level'], 'INFO')
        self.assertEqual(log_data['logger'], 'test_logger')
        self.assertEqual(log_data['message'], 'Test message')
        self.assertEqual(log_data['line'], 10)
        self.assertIn('timestamp', log_data)
    
    def test_extra_data_formatting(self):
        """Test formatting with extra data."""
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None
        )
        record.extra_data = {'symbol': 'RELIANCE.NS', 'action': 'buy'}
        
        formatted = self.formatter.format(record)
        log_data = json.loads(formatted)
        
        self.assertIn('extra_data', log_data)
        self.assertEqual(log_data['extra_data']['symbol'], 'RELIANCE.NS')
        self.assertEqual(log_data['extra_data']['action'], 'buy')
    
    def test_exception_formatting(self):
        """Test formatting with exception information."""
        try:
            raise ValueError("Test exception")
        except ValueError:
            exc_info = sys.exc_info()
            record = logging.LogRecord(
                name="test_logger",
                level=logging.ERROR,
                pathname="test.py",
                lineno=10,
                msg="Test error",
                args=(),
                exc_info=exc_info
            )
        
        formatted = self.formatter.format(record)
        log_data = json.loads(formatted)
        
        self.assertIn('exception', log_data)
        self.assertEqual(log_data['exception']['type'], 'ValueError')
        self.assertEqual(log_data['exception']['message'], 'Test exception')
        self.assertIsInstance(log_data['exception']['traceback'], list)


class TestTradingSystemLogger(unittest.TestCase):
    """Test main trading system logger."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config_patcher = patch('src.utils.logging_utils.config')
        mock_config = self.config_patcher.start()
        mock_config.logging.log_directory = self.temp_dir
        mock_config.logging.log_level = "INFO"
        mock_config.logging.enable_console_logging = True
        mock_config.logging.enable_file_logging = False  # Disable file logging to avoid conflicts
        mock_config.logging.log_file_format = "test_{date}.log"
        mock_config.logging.log_rotation_size = "1MB"
        mock_config.logging.max_log_files = 5
        
        self.logger = TradingSystemLogger("TestLogger")
    
    def tearDown(self):
        # Close all handlers to release file locks
        for handler in self.logger.logger.handlers[:]:
            handler.close()
            self.logger.logger.removeHandler(handler)
        
        self.config_patcher.stop()
        try:
            shutil.rmtree(self.temp_dir)
        except PermissionError:
            # If files are still locked, try again after a brief delay
            import time
            time.sleep(0.1)
            try:
                shutil.rmtree(self.temp_dir)
            except PermissionError:
                pass  # Skip cleanup if still locked
    
    def test_logger_initialization(self):
        """Test logger is properly initialized."""
        self.assertIsNotNone(self.logger.logger)
        self.assertEqual(self.logger.name, "TestLogger")
        self.assertTrue(len(self.logger.logger.handlers) > 0)
    
    def test_size_parsing(self):
        """Test size string parsing."""
        self.assertEqual(self.logger._parse_size("1KB"), 1024)
        self.assertEqual(self.logger._parse_size("1MB"), 1024 * 1024)
        self.assertEqual(self.logger._parse_size("1GB"), 1024 * 1024 * 1024)
        self.assertEqual(self.logger._parse_size("1000"), 1000)
    
    def test_structured_logging(self):
        """Test structured logging functionality."""
        extra_data = {'symbol': 'TCS.NS', 'price': 3500.0}
        
        with patch.object(self.logger.logger, 'handle') as mock_handle:
            self.logger.log_structured("INFO", "Test message", extra_data)
            mock_handle.assert_called_once()
            
            # Check the record has extra data
            call_args = mock_handle.call_args[0][0]
            self.assertTrue(hasattr(call_args, 'extra_data'))
            self.assertEqual(call_args.extra_data, extra_data)


class TestLoggingFunctions(unittest.TestCase):
    """Test individual logging functions."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        
        # Mock config
        self.config_patcher = patch('src.utils.logging_utils.config')
        mock_config = self.config_patcher.start()
        mock_config.logging.log_directory = self.temp_dir
        mock_config.logging.log_level = "INFO"
        mock_config.logging.enable_console_logging = False
        mock_config.logging.enable_file_logging = False  # Disable file logging to avoid conflicts
        mock_config.logging.log_file_format = "test_{date}.log"
        mock_config.logging.log_rotation_size = "1MB"
        mock_config.logging.max_log_files = 5
        mock_config.environment = "test"
        mock_config.data.data_directory = "test_data"
        mock_config.model.model_save_directory = "test_models"
        mock_config.validate_config.return_value = True
    
    def tearDown(self):
        self.config_patcher.stop()
        try:
            shutil.rmtree(self.temp_dir)
        except PermissionError:
            # If files are still locked, try again after a brief delay
            import time
            time.sleep(0.1)
            try:
                shutil.rmtree(self.temp_dir)
            except PermissionError:
                pass  # Skip cleanup if still locked
    
    def test_get_logger(self):
        """Test logger retrieval."""
        logger = get_logger("TestComponent")
        self.assertIsInstance(logger, logging.Logger)
        self.assertIn("TestComponent", logger.name)
    
    def test_log_system_startup(self):
        """Test system startup logging."""
        with patch('src.utils.logging_utils.get_logger') as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger
            
            log_system_startup()
            
            # Verify startup messages were logged
            self.assertTrue(mock_logger.info.called)
            call_args = [call[0][0] for call in mock_logger.info.call_args_list]
            self.assertTrue(any("Starting Up" in arg for arg in call_args))
    
    def test_log_system_shutdown(self):
        """Test system shutdown logging."""
        with patch('src.utils.logging_utils.get_logger') as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger
            
            log_system_shutdown()
            
            # Verify shutdown messages were logged
            self.assertTrue(mock_logger.info.called)
            call_args = [call[0][0] for call in mock_logger.info.call_args_list]
            self.assertTrue(any("Shutting Down" in arg for arg in call_args))
    
    def test_log_error_with_context(self):
        """Test error logging with context."""
        mock_logger = MagicMock()
        test_error = ValueError("Test error message")
        context = "test_function"
        extra_data = {"symbol": "INFY.NS"}
        
        with patch('src.utils.logging_utils.TradingSystemLogger') as mock_logger_class:
            mock_structured_logger = MagicMock()
            mock_logger_class.return_value = mock_structured_logger
            
            log_error_with_context(mock_logger, test_error, context, extra_data)
            
            # Verify error was logged to regular logger
            mock_logger.error.assert_called()
            mock_logger.exception.assert_called_once()
            
            # Verify structured logging was called
            mock_structured_logger.log_structured.assert_called_once()
    
    def test_log_performance_metrics(self):
        """Test performance metrics logging."""
        metrics = {
            "total_return": 0.15,
            "sharpe_ratio": 1.8,
            "max_drawdown": 0.05,
            "win_rate": 0.62
        }
        
        with patch('src.utils.logging_utils.get_logger') as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger
            
            log_performance_metrics(metrics)
            
            # Verify metrics were logged
            self.assertTrue(mock_logger.info.called)
            call_args = [call[0][0] for call in mock_logger.info.call_args_list]
            self.assertTrue(any("Performance Metrics" in arg for arg in call_args))
    
    def test_log_trading_decision(self):
        """Test trading decision logging."""
        signal = {
            'symbol': 'RELIANCE.NS',
            'action': 'buy',
            'confidence': 0.85,
            'lstm_prediction': 0.7,
            'sentiment_score': 0.3,
            'risk_adjusted_size': 0.02,
            'dqn_q_values': {'buy': 0.8, 'sell': 0.1, 'hold': 0.1}
        }
        
        with patch('src.utils.logging_utils.get_logger') as mock_get_logger, \
             patch('src.utils.logging_utils.TradingSystemLogger') as mock_logger_class:
            
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger
            mock_structured_logger = MagicMock()
            mock_logger_class.return_value = mock_structured_logger
            
            log_trading_decision(signal)
            
            # Verify standard logging
            self.assertTrue(mock_logger.info.called)
            
            # Verify structured logging
            mock_structured_logger.log_structured.assert_called_once()
            call_args = mock_structured_logger.log_structured.call_args
            self.assertEqual(call_args[0][0], "INFO")
            # Check the structured data passed as third argument
            structured_data = call_args[0][2]
            self.assertEqual(structured_data['event_type'], 'trading_decision')
    
    def test_log_model_prediction(self):
        """Test model prediction logging."""
        with patch('src.utils.logging_utils.get_logger') as mock_get_logger, \
             patch('src.utils.logging_utils.TradingSystemLogger') as mock_logger_class:
            
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger
            mock_structured_logger = MagicMock()
            mock_logger_class.return_value = mock_structured_logger
            
            log_model_prediction("LSTM", "TCS.NS", 0.75, 0.9)
            
            # Verify logging was called
            self.assertTrue(mock_logger.info.called)
            mock_structured_logger.log_structured.assert_called_once()
    
    def test_log_risk_management_action(self):
        """Test risk management action logging."""
        details = {
            'stop_loss_price': 2850.0,
            'position_size': 0.02,
            'risk_score': 0.3
        }
        
        with patch('src.utils.logging_utils.get_logger') as mock_get_logger, \
             patch('src.utils.logging_utils.TradingSystemLogger') as mock_logger_class:
            
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger
            mock_structured_logger = MagicMock()
            mock_logger_class.return_value = mock_structured_logger
            
            log_risk_management_action("stop_loss_triggered", "HDFC.NS", details)
            
            # Verify logging was called
            self.assertTrue(mock_logger.info.called)
            mock_structured_logger.log_structured.assert_called_once()
    
    def test_log_system_health(self):
        """Test system health logging."""
        metrics = {
            'cpu_usage': 45.2,
            'memory_usage': 78.5,
            'disk_space': 85.0
        }
        
        with patch('src.utils.logging_utils.get_logger') as mock_get_logger, \
             patch('src.utils.logging_utils.TradingSystemLogger') as mock_logger_class:
            
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger
            mock_structured_logger = MagicMock()
            mock_logger_class.return_value = mock_structured_logger
            
            log_system_health("DataProcessor", "healthy", metrics)
            
            # Verify logging was called
            self.assertTrue(mock_logger.info.called)
            mock_structured_logger.log_structured.assert_called_once()
    
    def test_log_performance_alert(self):
        """Test performance alert logging."""
        metrics = {'drawdown': 0.09, 'sharpe_ratio': 1.2}
        
        with patch('src.utils.logging_utils.get_logger') as mock_get_logger, \
             patch('src.utils.logging_utils.TradingSystemLogger') as mock_logger_class:
            
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger
            mock_structured_logger = MagicMock()
            mock_logger_class.return_value = mock_structured_logger
            
            log_performance_alert("high_drawdown", "Drawdown exceeded limit", "WARNING", metrics)
            
            # Verify logging was called with correct level
            mock_logger.log.assert_called()
            mock_structured_logger.log_structured.assert_called_once()
    
    def test_log_data_quality_issue(self):
        """Test data quality issue logging."""
        details = {
            'missing_values': 5,
            'date_range': '2024-01-01 to 2024-01-05',
            'severity': 'medium'
        }
        
        with patch('src.utils.logging_utils.get_logger') as mock_get_logger, \
             patch('src.utils.logging_utils.TradingSystemLogger') as mock_logger_class:
            
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger
            mock_structured_logger = MagicMock()
            mock_logger_class.return_value = mock_structured_logger
            
            log_data_quality_issue("ICICI.NS", "missing_data", details)
            
            # Verify warning level logging
            self.assertTrue(mock_logger.warning.called)
            mock_structured_logger.log_structured.assert_called_once()


class TestLogArchiver(unittest.TestCase):
    """Test log archiving functionality."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        
        # Create some test log files
        self.log_files = []
        for i in range(3):
            log_file = Path(self.temp_dir) / f"test_{i}.log"
            log_file.write_text(f"Test log content {i}")
            self.log_files.append(log_file)
        
        # Mock config
        self.config_patcher = patch('src.utils.logging_utils.config')
        mock_config = self.config_patcher.start()
        mock_config.logging.log_directory = self.temp_dir
    
    def tearDown(self):
        self.config_patcher.stop()
        shutil.rmtree(self.temp_dir)
    
    def test_log_archiver_creation(self):
        """Test log archiver function creation."""
        archiver = create_log_archiver()
        self.assertTrue(callable(archiver))
    
    def test_log_archiving(self):
        """Test actual log archiving functionality."""
        # Make files appear old by modifying their timestamps
        import time
        old_time = time.time() - (35 * 24 * 60 * 60)  # 35 days ago
        
        for log_file in self.log_files:
            os.utime(log_file, (old_time, old_time))
        
        archiver = create_log_archiver()
        
        with patch('src.utils.logging_utils.get_logger') as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger
            
            archiver(days_to_keep=30)
            
            # Check that archive directory was created
            archive_dir = Path(self.temp_dir) / "archive"
            self.assertTrue(archive_dir.exists())
            
            # Check that original files were removed and archived files exist
            for log_file in self.log_files:
                self.assertFalse(log_file.exists())
                archived_file = archive_dir / f"{log_file.name}.gz"
                self.assertTrue(archived_file.exists())


if __name__ == '__main__':
    unittest.main()