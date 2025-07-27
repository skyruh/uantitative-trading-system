"""
Logging utilities for the quantitative trading system.
Provides centralized logging configuration and helper functions.
"""

import logging
import os
import json
import traceback
import sys
from datetime import datetime
from typing import Optional, Dict, Any, Union
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from pathlib import Path

from src.config.settings import config


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured logging with JSON output."""
    
    def format(self, record):
        """Format log record as structured JSON."""
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'message': record.getMessage(),
            'thread': record.thread,
            'process': record.process
        }
        
        # Add extra fields if present
        if hasattr(record, 'extra_data'):
            log_entry['extra_data'] = record.extra_data
        
        # Add exception info if present
        if record.exc_info and record.exc_info != (None, None, None):
            log_entry['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        return json.dumps(log_entry, default=str)


class TradingSystemLogger:
    """Centralized logger for the trading system with enhanced features."""
    
    def __init__(self, name: str = "TradingSystem"):
        self.name = name
        self.logger = logging.getLogger(name)
        self._setup_logger()
    
    def _setup_logger(self):
        """Set up logger with file and console handlers."""
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Set log level
        log_level = getattr(logging, config.logging.log_level.upper())
        self.logger.setLevel(log_level)
        
        # Create logs directory if it doesn't exist
        log_dir = Path(config.logging.log_directory)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Console handler with standard formatting
        if config.logging.enable_console_logging:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(log_level)
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
        
        # File handlers
        if config.logging.enable_file_logging:
            self._setup_file_handlers(log_dir, log_level)
    
    def _setup_file_handlers(self, log_dir: Path, log_level: int):
        """Set up various file handlers for different log types."""
        
        # Main application log with rotation
        main_log_file = log_dir / config.logging.log_file_format.format(
            date=datetime.now().strftime("%Y%m%d")
        )
        main_handler = RotatingFileHandler(
            main_log_file,
            maxBytes=self._parse_size(config.logging.log_rotation_size),
            backupCount=config.logging.max_log_files
        )
        main_handler.setLevel(log_level)
        main_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        main_handler.setFormatter(main_formatter)
        self.logger.addHandler(main_handler)
        
        # Structured JSON log for trading decisions and model predictions
        structured_log_file = log_dir / f"structured_{datetime.now().strftime('%Y%m%d')}.jsonl"
        structured_handler = TimedRotatingFileHandler(
            structured_log_file,
            when='midnight',
            interval=1,
            backupCount=config.logging.max_log_files
        )
        structured_handler.setLevel(logging.INFO)
        structured_handler.setFormatter(StructuredFormatter())
        
        # Filter for structured logs (only specific loggers)
        structured_handler.addFilter(self._structured_log_filter)
        self.logger.addHandler(structured_handler)
        
        # Error log with detailed stack traces
        error_log_file = log_dir / f"errors_{datetime.now().strftime('%Y%m%d')}.log"
        error_handler = TimedRotatingFileHandler(
            error_log_file,
            when='midnight',
            interval=1,
            backupCount=config.logging.max_log_files
        )
        error_handler.setLevel(logging.ERROR)
        error_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d\n'
            'Message: %(message)s\n'
            'Exception: %(exc_text)s\n'
            '%(separator)s\n',
            defaults={'separator': '-' * 80}
        )
        error_handler.setFormatter(error_formatter)
        self.logger.addHandler(error_handler)
    
    def _structured_log_filter(self, record):
        """Filter for structured logs - only allow specific logger names."""
        structured_loggers = [
            'TradingSystem.Trading',
            'TradingSystem.ModelPrediction',
            'TradingSystem.Performance',
            'TradingSystem.RiskManagement'
        ]
        return record.name in structured_loggers
    
    def _parse_size(self, size_str: str) -> int:
        """Parse size string (e.g., '10MB') to bytes."""
        size_str = size_str.upper()
        if size_str.endswith('KB'):
            return int(size_str[:-2]) * 1024
        elif size_str.endswith('MB'):
            return int(size_str[:-2]) * 1024 * 1024
        elif size_str.endswith('GB'):
            return int(size_str[:-2]) * 1024 * 1024 * 1024
        else:
            return int(size_str)
    
    def get_logger(self) -> logging.Logger:
        """Get the configured logger instance."""
        return self.logger
    
    def log_structured(self, level: str, message: str, extra_data: Dict[str, Any] = None):
        """Log structured data with additional context."""
        logger = self.get_logger()
        log_level = getattr(logging, level.upper())
        
        # Create log record with extra data
        record = logger.makeRecord(
            logger.name, log_level, __file__, 0, message, (), None
        )
        if extra_data:
            record.extra_data = extra_data
        
        logger.handle(record)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get a logger instance for the specified name."""
    logger_name = f"TradingSystem.{name}" if name else "TradingSystem"
    return TradingSystemLogger(logger_name).get_logger()


def log_system_startup():
    """Log system startup information."""
    logger = get_logger("Startup")
    logger.info("=" * 50)
    logger.info("Quantitative Trading System Starting Up")
    logger.info("=" * 50)
    logger.info(f"Environment: {config.environment}")
    logger.info(f"Data Directory: {config.data.data_directory}")
    logger.info(f"Model Directory: {config.model.model_save_directory}")
    logger.info(f"Log Directory: {config.logging.log_directory}")
    logger.info("Configuration validation: %s", "PASSED" if config.validate_config() else "FAILED")


def log_system_shutdown():
    """Log system shutdown information."""
    logger = get_logger("Shutdown")
    logger.info("=" * 50)
    logger.info("Quantitative Trading System Shutting Down")
    logger.info("=" * 50)


def log_error_with_context(logger: logging.Logger, error: Exception, context: str, 
                          extra_data: Dict[str, Any] = None):
    """Log error with additional context information and structured data."""
    error_data = {
        'context': context,
        'error_type': type(error).__name__,
        'error_message': str(error),
        'traceback': traceback.format_exc(),
        'system_info': {
            'python_version': sys.version,
            'platform': sys.platform
        }
    }
    
    if extra_data:
        error_data['extra_context'] = extra_data
    
    # Log structured error
    structured_logger = TradingSystemLogger("TradingSystem.Error")
    structured_logger.log_structured("ERROR", f"Error in {context}", error_data)
    
    # Also log to regular logger
    logger.error(f"Error in {context}: {str(error)}")
    logger.error(f"Error type: {type(error).__name__}")
    logger.exception("Full traceback:")


def log_performance_metrics(metrics: dict, logger_name: str = "Performance"):
    """Log performance metrics in a structured format."""
    logger = get_logger(logger_name)
    logger.info("Performance Metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.4f}")
        else:
            logger.info(f"  {key}: {value}")


def log_trading_decision(signal: dict, logger_name: str = "Trading"):
    """Log trading decision with all relevant information."""
    logger = get_logger(logger_name)
    
    # Standard logging
    logger.info(f"Trading Decision - Symbol: {signal.get('symbol', 'N/A')}")
    logger.info(f"  Action: {signal.get('action', 'N/A')}")
    logger.info(f"  Confidence: {signal.get('confidence', 0):.4f}")
    logger.info(f"  LSTM Prediction: {signal.get('lstm_prediction', 0):.4f}")
    logger.info(f"  Position Size: {signal.get('risk_adjusted_size', 0):.4f}")
    
    # Structured logging for analysis
    structured_logger = TradingSystemLogger("TradingSystem.Trading")
    structured_data = {
        'event_type': 'trading_decision',
        'symbol': signal.get('symbol'),
        'action': signal.get('action'),
        'confidence': signal.get('confidence', 0),
        'lstm_prediction': signal.get('lstm_prediction', 0),
        'position_size': signal.get('risk_adjusted_size', 0),
        'timestamp': datetime.now().isoformat(),
        'dqn_q_values': signal.get('dqn_q_values', {}),
        'risk_metrics': signal.get('risk_metrics', {})
    }
    structured_logger.log_structured("INFO", "Trading decision made", structured_data)


def log_data_collection_progress(symbol: str, progress: int, total: int, 
                               logger_name: str = "DataCollection"):
    """Log data collection progress."""
    logger = get_logger(logger_name)
    percentage = (progress / total) * 100 if total > 0 else 0
    logger.info(f"Data Collection Progress: {symbol} ({progress}/{total}) - {percentage:.1f}%")


def log_model_training_progress(epoch: int, total_epochs: int, loss: float,
                              logger_name: str = "ModelTraining"):
    """Log model training progress."""
    logger = get_logger(logger_name)
    percentage = (epoch / total_epochs) * 100 if total_epochs > 0 else 0
    logger.info(f"Training Progress: Epoch {epoch}/{total_epochs} ({percentage:.1f}%) - Loss: {loss:.6f}")


def log_model_prediction(model_type: str, symbol: str, prediction: Union[float, Dict], 
                        confidence: float = None, logger_name: str = "ModelPrediction"):
    """Log model predictions with structured data."""
    logger = get_logger(logger_name)
    
    # Standard logging
    logger.info(f"Model Prediction - {model_type} for {symbol}")
    if isinstance(prediction, dict):
        for key, value in prediction.items():
            logger.info(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
    else:
        logger.info(f"  Prediction: {prediction:.4f}" if isinstance(prediction, float) else f"  Prediction: {prediction}")
    
    if confidence is not None:
        logger.info(f"  Confidence: {confidence:.4f}")
    
    # Structured logging
    structured_logger = TradingSystemLogger("TradingSystem.ModelPrediction")
    structured_data = {
        'event_type': 'model_prediction',
        'model_type': model_type,
        'symbol': symbol,
        'prediction': prediction,
        'confidence': confidence,
        'timestamp': datetime.now().isoformat()
    }
    structured_logger.log_structured("INFO", f"{model_type} prediction made", structured_data)


def log_risk_management_action(action_type: str, symbol: str, details: Dict[str, Any],
                              logger_name: str = "RiskManagement"):
    """Log risk management actions with detailed context."""
    logger = get_logger(logger_name)
    
    # Standard logging
    logger.info(f"Risk Management Action - {action_type} for {symbol}")
    for key, value in details.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.4f}")
        else:
            logger.info(f"  {key}: {value}")
    
    # Structured logging
    structured_logger = TradingSystemLogger("TradingSystem.RiskManagement")
    structured_data = {
        'event_type': 'risk_management_action',
        'action_type': action_type,
        'symbol': symbol,
        'details': details,
        'timestamp': datetime.now().isoformat()
    }
    structured_logger.log_structured("INFO", f"Risk management action: {action_type}", structured_data)


def log_position_update(symbol: str, action: str, position_data: Dict[str, Any],
                       logger_name: str = "PositionManagement"):
    """Log position updates with comprehensive data."""
    logger = get_logger(logger_name)
    
    # Standard logging
    logger.info(f"Position Update - {action} for {symbol}")
    for key, value in position_data.items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.4f}")
        else:
            logger.info(f"  {key}: {value}")
    
    # Structured logging
    structured_logger = TradingSystemLogger("TradingSystem.Trading")
    structured_data = {
        'event_type': 'position_update',
        'symbol': symbol,
        'action': action,
        'position_data': position_data,
        'timestamp': datetime.now().isoformat()
    }
    structured_logger.log_structured("INFO", f"Position {action}", structured_data)


def log_system_health(component: str, status: str, metrics: Dict[str, Any] = None,
                     logger_name: str = "SystemHealth"):
    """Log system health status and metrics."""
    logger = get_logger(logger_name)
    
    # Standard logging
    logger.info(f"System Health - {component}: {status}")
    if metrics:
        for key, value in metrics.items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.4f}")
            else:
                logger.info(f"  {key}: {value}")
    
    # Structured logging for monitoring
    structured_logger = TradingSystemLogger("TradingSystem.SystemHealth")
    structured_data = {
        'event_type': 'system_health_check',
        'component': component,
        'status': status,
        'metrics': metrics or {},
        'timestamp': datetime.now().isoformat()
    }
    structured_logger.log_structured("INFO", f"System health check: {component}", structured_data)


def log_performance_alert(alert_type: str, message: str, severity: str = "WARNING",
                         metrics: Dict[str, Any] = None, logger_name: str = "PerformanceAlert"):
    """Log performance alerts with severity levels."""
    logger = get_logger(logger_name)
    
    # Log at appropriate level based on severity
    log_level = getattr(logging, severity.upper(), logging.WARNING)
    logger.log(log_level, f"Performance Alert - {alert_type}: {message}")
    
    if metrics:
        for key, value in metrics.items():
            if isinstance(value, float):
                logger.log(log_level, f"  {key}: {value:.4f}")
            else:
                logger.log(log_level, f"  {key}: {value}")
    
    # Structured logging for alerts
    structured_logger = TradingSystemLogger("TradingSystem.Performance")
    structured_data = {
        'event_type': 'performance_alert',
        'alert_type': alert_type,
        'message': message,
        'severity': severity,
        'metrics': metrics or {},
        'timestamp': datetime.now().isoformat()
    }
    structured_logger.log_structured(severity, f"Performance alert: {alert_type}", structured_data)


def log_data_quality_issue(symbol: str, issue_type: str, details: Dict[str, Any],
                          logger_name: str = "DataQuality"):
    """Log data quality issues for monitoring and debugging."""
    logger = get_logger(logger_name)
    
    # Standard logging
    logger.warning(f"Data Quality Issue - {symbol}: {issue_type}")
    for key, value in details.items():
        logger.warning(f"  {key}: {value}")
    
    # Structured logging
    structured_logger = TradingSystemLogger("TradingSystem.DataQuality")
    structured_data = {
        'event_type': 'data_quality_issue',
        'symbol': symbol,
        'issue_type': issue_type,
        'details': details,
        'timestamp': datetime.now().isoformat()
    }
    structured_logger.log_structured("WARNING", f"Data quality issue: {issue_type}", structured_data)


def create_log_archiver():
    """Create a log archiver for old log files."""
    import shutil
    import gzip
    from pathlib import Path
    
    def archive_old_logs(days_to_keep: int = 30):
        """Archive log files older than specified days."""
        logger = get_logger("LogArchiver")
        log_dir = Path(config.logging.log_directory)
        
        if not log_dir.exists():
            return
        
        cutoff_date = datetime.now().timestamp() - (days_to_keep * 24 * 60 * 60)
        archive_dir = log_dir / "archive"
        archive_dir.mkdir(exist_ok=True)
        
        for log_file in log_dir.glob("*.log*"):
            if log_file.is_file() and log_file.stat().st_mtime < cutoff_date:
                try:
                    # Compress and move to archive
                    archive_path = archive_dir / f"{log_file.name}.gz"
                    with open(log_file, 'rb') as f_in:
                        with gzip.open(archive_path, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    
                    log_file.unlink()  # Remove original
                    logger.info(f"Archived log file: {log_file.name}")
                    
                except Exception as e:
                    logger.error(f"Failed to archive {log_file.name}: {e}")
    
    return archive_old_logs


# Initialize logging on module import
if not logging.getLogger("TradingSystem").handlers:
    get_logger()  # Initialize the main logger