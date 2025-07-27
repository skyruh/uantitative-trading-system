"""
Configuration settings for the quantitative trading system.
Centralized configuration management with environment-specific settings.
"""

import os
from dataclasses import dataclass
from typing import List, Dict, Any
from datetime import datetime


@dataclass
class DataConfig:
    """Data collection and storage configuration."""
    data_directory: str = "data"
    start_date: str = "1995-01-01"
    end_date: str = datetime.now().strftime("%Y-%m-%d")
    stock_symbols_file: str = "config/indian_stocks.txt"
    yfinance_timeout: int = 30
    max_retries: int = 3
    retry_delay: int = 5


@dataclass
class ModelConfig:
    """Model training configuration."""
    # LSTM Configuration
    lstm_layers: int = 2
    lstm_units: int = 50
    lstm_dropout: float = 0.2
    lstm_epochs: int = 100
    lstm_batch_size: int = 32
    lstm_validation_split: float = 0.2
    
    # DQN Configuration
    dqn_learning_rate: float = 0.001
    dqn_epsilon: float = 1.0
    dqn_epsilon_min: float = 0.01
    dqn_epsilon_decay: float = 0.995
    dqn_memory_size: int = 10000
    dqn_batch_size: int = 32
    dqn_target_update_freq: int = 100
    
    # General Model Settings
    model_save_directory: str = "models"
    random_seed: int = 42


@dataclass
class TechnicalIndicatorConfig:
    """Technical indicator calculation configuration."""
    rsi_period: int = 14
    sma_period: int = 50
    bollinger_period: int = 20
    bollinger_std: float = 2.0


@dataclass
class SentimentConfig:
    """Sentiment analysis configuration."""
    model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"
    batch_size: int = 16
    max_length: int = 512
    neutral_score: float = 0.0


@dataclass
class RiskConfig:
    """Risk management configuration."""
    stop_loss_percentage: float = 0.05  # 5%
    position_size_percentage: float = 0.02  # 2% of capital
    max_position_size_percentage: float = 0.01  # 1% minimum
    max_positions: int = 30
    min_positions: int = 20
    sentiment_adjustment_max: float = 0.20  # ±20%
    portfolio_concentration_limit: float = 0.10  # 10% max per stock


@dataclass
class BacktestConfig:
    """Backtesting configuration."""
    initial_capital: float = 1000000.0  # ₹10,00,000
    transaction_cost: float = 0.001  # 0.1%
    slippage: float = 0.0005  # 0.05%
    use_leverage: bool = False
    allow_short_selling: bool = False  # Enable short selling for intraday trading
    train_start_date: str = "1995-01-01"
    train_end_date: str = "2019-12-31"
    test_start_date: str = "2020-01-01"
    test_end_date: str = datetime.now().strftime("%Y-%m-%d")


@dataclass
class PerformanceConfig:
    """Performance monitoring configuration."""
    target_annual_return: float = 0.175  # 17.5% (midpoint of 15-20%)
    target_sharpe_ratio: float = 1.8
    max_drawdown_limit: float = 0.08  # 8%
    target_win_rate: float = 0.625  # 62.5% (midpoint of 60-65%)
    benchmark_symbol: str = "^NSEI"  # NIFTY 50
    rebalance_frequency: str = "daily"


@dataclass
class LoggingConfig:
    """Logging configuration."""
    log_level: str = "INFO"
    log_directory: str = "logs"
    log_file_format: str = "trading_system_{date}.log"
    max_log_files: int = 30
    log_rotation_size: str = "10MB"
    enable_console_logging: bool = True
    enable_file_logging: bool = True


class TradingSystemConfig:
    """Main configuration class that combines all settings."""
    
    def __init__(self, environment: str = "development"):
        self.environment = environment
        self.data = DataConfig()
        self.model = ModelConfig()
        self.technical_indicators = TechnicalIndicatorConfig()
        self.sentiment = SentimentConfig()
        self.risk = RiskConfig()
        self.backtest = BacktestConfig()
        self.performance = PerformanceConfig()
        self.logging = LoggingConfig()
        
        # Load environment-specific overrides
        self._load_environment_config()
    
    def _load_environment_config(self):
        """Load environment-specific configuration overrides."""
        if self.environment == "production":
            self.logging.log_level = "WARNING"
            self.model.random_seed = None  # Use random seed in production
        elif self.environment == "testing":
            self.data.start_date = "2020-01-01"  # Smaller dataset for testing
            self.model.lstm_epochs = 10  # Faster training for tests
            self.backtest.initial_capital = 100000.0  # Smaller capital for tests
    
    def get_stock_symbols(self) -> List[str]:
        """Get list of Indian stock symbols to trade."""
        # Default NIFTY 50 symbols (subset for initial implementation)
        default_symbols = [
            "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "HINDUNILVR.NS",
            "ICICIBANK.NS", "KOTAKBANK.NS", "SBIN.NS", "BHARTIARTL.NS", "ITC.NS",
            "ASIANPAINT.NS", "LT.NS", "AXISBANK.NS", "MARUTI.NS", "SUNPHARMA.NS",
            "TITAN.NS", "ULTRACEMCO.NS", "NESTLEIND.NS", "WIPRO.NS", "M&M.NS"
        ]
        
        try:
            # Try to load from file if it exists
            if os.path.exists(self.data.stock_symbols_file):
                with open(self.data.stock_symbols_file, 'r') as f:
                    symbols = [line.strip() for line in f if line.strip()]
                return symbols if symbols else default_symbols
        except Exception:
            pass
        
        return default_symbols
    
    def validate_config(self) -> bool:
        """Validate configuration settings."""
        try:
            # Validate data configuration
            assert self.data.start_date < self.data.end_date
            assert self.data.max_retries > 0
            assert self.data.yfinance_timeout > 0
            
            # Validate model configuration
            assert 0 < self.model.lstm_dropout < 1
            assert self.model.lstm_layers > 0
            assert self.model.lstm_units > 0
            assert 0 < self.model.dqn_epsilon_min < self.model.dqn_epsilon <= 1
            
            # Validate risk configuration
            assert 0 < self.risk.stop_loss_percentage < 1
            assert 0 < self.risk.position_size_percentage < 1
            assert self.risk.min_positions <= self.risk.max_positions
            
            # Validate backtest configuration
            assert self.backtest.initial_capital > 0
            assert 0 <= self.backtest.transaction_cost < 1
            
            return True
        except AssertionError:
            return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "environment": self.environment,
            "data": self.data.__dict__,
            "model": self.model.__dict__,
            "technical_indicators": self.technical_indicators.__dict__,
            "sentiment": self.sentiment.__dict__,
            "risk": self.risk.__dict__,
            "backtest": self.backtest.__dict__,
            "performance": self.performance.__dict__,
            "logging": self.logging.__dict__
        }


# Global configuration instance
config = TradingSystemConfig()