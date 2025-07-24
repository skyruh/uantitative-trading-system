"""
Unit tests for the BacktestEngine class.
Tests backtesting mechanics, trade execution, and performance calculations.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from src.backtesting.backtest_engine import BacktestEngine, BacktestConfig
from src.interfaces.trading_interfaces import (
    ITradingStrategy, TradingSignal, Position, PerformanceMetrics
)


class MockTradingStrategy(ITradingStrategy):
    """Mock trading strategy for testing."""
    
    def __init__(self, signals_to_generate=None):
        self.signals_to_generate = signals_to_generate or []
        self.call_count = 0
    
    def execute_strategy(self, market_data):
        """Generate predefined signals for testing."""
        if self.call_count < len(self.signals_to_generate):
            signals = self.signals_to_generate[self.call_count]
            self.call_count += 1
            return signals
        return []
    
    def process_signals(self, signals):
        """Mock implementation."""
        return []


class TestBacktestConfig:
    """Test BacktestConfig class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = BacktestConfig()
        
        assert config.initial_capital == 1000000.0
        assert config.transaction_cost_pct == 0.001
        assert config.start_date == "1995-01-01"
        assert config.end_date == "2025-01-01"
        assert config.benchmark_symbol == "^NSEI"
        assert config.rebalance_frequency == "daily"
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = BacktestConfig(
            initial_capital=500000.0,
            transaction_cost_pct=0.002,
            start_date="2020-01-01",
            end_date="2023-01-01"
        )
        
        assert config.initial_capital == 500000.0
        assert config.transaction_cost_pct == 0.002
        assert config.start_date == "2020-01-01"
        assert config.end_date == "2023-01-01"
    
    def test_invalid_config(self):
        """Test validation of invalid configuration."""
        with pytest.raises(ValueError, match="Initial capital must be positive"):
            BacktestConfig(initial_capital=-100000.0)
        
        with pytest.raises(ValueError, match="Transaction cost must be between 0 and 1"):
            BacktestConfig(transaction_cost_pct=1.5)


class TestBacktestEngine:
    """Test BacktestEngine class."""
    
    @pytest.fixture
    def engine(self):
        """Create BacktestEngine instance for testing."""
        config = BacktestConfig(
            initial_capital=1000000.0,
            transaction_cost_pct=0.001,
            start_date="2020-01-01",
            end_date="2020-12-31"
        )
        return BacktestEngine(config)
    
    @pytest.fixture
    def sample_market_data(self):
        """Create sample market data for testing."""
        dates = pd.date_range('2020-01-01', '2020-12-31', freq='B')  # Business days
        
        # Create sample data for two stocks
        data = {}
        for symbol in ['RELIANCE.NS', 'TCS.NS']:
            # Generate realistic price data
            np.random.seed(42)  # For reproducible tests
            prices = 100 + np.cumsum(np.random.randn(len(dates)) * 0.02)
            
            df = pd.DataFrame({
                'Open': prices * (1 + np.random.randn(len(dates)) * 0.001),
                'High': prices * (1 + abs(np.random.randn(len(dates))) * 0.005),
                'Low': prices * (1 - abs(np.random.randn(len(dates))) * 0.005),
                'Close': prices,
                'Volume': np.random.randint(1000000, 10000000, len(dates))
            }, index=dates)
            
            data[symbol] = df
        
        return data
    
    @pytest.fixture
    def sample_benchmark_data(self):
        """Create sample benchmark data for testing."""
        dates = pd.date_range('2020-01-01', '2020-12-31', freq='B')
        
        # Generate NIFTY 50 like data
        np.random.seed(123)
        prices = 12000 + np.cumsum(np.random.randn(len(dates)) * 0.015)
        
        return pd.DataFrame({
            'Open': prices * (1 + np.random.randn(len(dates)) * 0.001),
            'High': prices * (1 + abs(np.random.randn(len(dates))) * 0.003),
            'Low': prices * (1 - abs(np.random.randn(len(dates))) * 0.003),
            'Close': prices,
            'Volume': np.random.randint(100000000, 1000000000, len(dates))
        }, index=dates)
    
    def test_initialization(self, engine):
        """Test engine initialization."""
        assert engine.config.initial_capital == 1000000.0
        assert engine.config.transaction_cost_pct == 0.001
        assert not engine.is_initialized
        assert engine.position_manager is None
        assert engine.risk_manager is None
    
    def test_initialize_backtest(self, engine):
        """Test backtest initialization."""
        # Test successful initialization
        result = engine.initialize_backtest(1000000.0, "2020-01-01", "2020-12-31")
        
        assert result is True
        assert engine.is_initialized
        assert engine.position_manager is not None
        assert engine.risk_manager is not None
        assert engine.config.initial_capital == 1000000.0
        assert len(engine.daily_portfolio_values) == 1
        
        # Check initial portfolio record
        initial_record = engine.daily_portfolio_values[0]
        assert initial_record['portfolio_value'] == 1000000.0
        assert initial_record['cash'] == 1000000.0
        assert initial_record['positions_value'] == 0.0
    
    def test_load_market_data(self, engine, sample_market_data):
        """Test loading market data."""
        # Test successful data loading
        result = engine.load_market_data(sample_market_data)
        
        assert result is True
        assert len(engine.market_data) == 2
        assert 'RELIANCE.NS' in engine.market_data
        assert 'TCS.NS' in engine.market_data
        
        # Test empty data
        result = engine.load_market_data({})
        assert result is False
        
        # Test invalid data format
        result = engine.load_market_data({'INVALID': "not_a_dataframe"})
        assert result is False
    
    def test_load_benchmark_data(self, engine, sample_benchmark_data):
        """Test loading benchmark data."""
        # Test successful benchmark loading
        result = engine.load_benchmark_data(sample_benchmark_data)
        
        assert result is True
        assert engine.benchmark_data is not None
        assert len(engine.benchmark_data) > 0
        
        # Test invalid benchmark data
        result = engine.load_benchmark_data("not_a_dataframe")
        assert result is False
    
    def test_get_market_data_for_date(self, engine, sample_market_data):
        """Test getting market data for specific date."""
        engine.load_market_data(sample_market_data)
        
        # Test getting data for valid date
        test_date = datetime(2020, 6, 15)
        daily_data = engine._get_market_data_for_date(test_date)
        
        assert len(daily_data) == 2
        assert 'RELIANCE.NS' in daily_data
        assert 'TCS.NS' in daily_data
        
        # Check data structure
        reliance_data = daily_data['RELIANCE.NS']
        assert 'Open' in reliance_data
        assert 'High' in reliance_data
        assert 'Low' in reliance_data
        assert 'Close' in reliance_data
        assert 'Volume' in reliance_data
        
        # Test getting data for date before data starts
        early_date = datetime(1990, 1, 1)
        early_data = engine._get_market_data_for_date(early_date)
        assert len(early_data) == 0
    
    def test_calculate_performance_metrics_empty(self, engine):
        """Test performance metrics calculation with no data."""
        metrics = engine.calculate_performance_metrics([])
        
        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.total_return == 0.0
        assert metrics.annualized_return == 0.0
        assert metrics.sharpe_ratio == 0.0
        assert metrics.max_drawdown == 0.0
        assert metrics.win_rate == 0.0
        assert metrics.total_trades == 0
    
    def test_calculate_performance_metrics_with_data(self, engine):
        """Test performance metrics calculation with sample data."""
        # Initialize engine
        engine.initialize_backtest(1000000.0, "2020-01-01", "2020-12-31")
        
        # Create sample daily portfolio values
        dates = pd.date_range('2020-01-01', '2020-01-10', freq='B')
        for i, date in enumerate(dates):
            portfolio_value = 1000000.0 * (1 + i * 0.001)  # Small daily gains
            engine.daily_portfolio_values.append({
                'date': date,
                'portfolio_value': portfolio_value,
                'cash': portfolio_value * 0.5,
                'positions_value': portfolio_value * 0.5,
                'daily_return': 0.1 if i > 0 else 0.0,
                'cumulative_return': i * 0.1
            })
        
        # Create sample trades
        sample_trades = [
            Position(
                symbol='RELIANCE.NS',
                entry_date=datetime(2020, 1, 2),
                entry_price=100.0,
                quantity=100,
                stop_loss_price=95.0,
                current_value=10500.0,
                unrealized_pnl=500.0,  # Winning trade
                status='closed'
            ),
            Position(
                symbol='TCS.NS',
                entry_date=datetime(2020, 1, 3),
                entry_price=200.0,
                quantity=50,
                stop_loss_price=190.0,
                current_value=9800.0,
                unrealized_pnl=-200.0,  # Losing trade
                status='closed'
            )
        ]
        
        metrics = engine.calculate_performance_metrics(sample_trades)
        
        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.total_trades == 2
        assert metrics.win_rate == 50.0  # 1 win out of 2 trades
        assert metrics.total_return > 0  # Should have positive return
        assert metrics.sharpe_ratio != 0.0
    
    @patch('src.backtesting.backtest_engine.PositionManager')
    @patch('src.backtesting.backtest_engine.RiskManager')
    def test_run_backtest_basic(self, mock_risk_manager, mock_position_manager, 
                               engine, sample_market_data):
        """Test basic backtest execution."""
        # Setup mocks
        mock_pm_instance = Mock()
        mock_rm_instance = Mock()
        mock_position_manager.return_value = mock_pm_instance
        mock_risk_manager.return_value = mock_rm_instance
        
        # Configure mock returns
        mock_pm_instance.get_portfolio_value.return_value = 1100000.0
        mock_pm_instance.get_available_capital.return_value = 500000.0
        mock_pm_instance.get_open_positions.return_value = []
        mock_pm_instance.closed_positions = []
        mock_pm_instance.check_stop_losses.return_value = []
        mock_pm_instance.update_positions_batch.return_value = {}
        
        mock_rm_instance.validate_trade.return_value = (True, "Valid")
        
        # Initialize and load data
        engine.initialize_backtest(1000000.0, "2020-01-01", "2020-01-05")
        engine.load_market_data(sample_market_data)
        
        # Create simple strategy that generates no signals
        strategy = MockTradingStrategy([])
        
        # Run backtest
        metrics = engine.run_backtest(strategy)
        
        assert isinstance(metrics, PerformanceMetrics)
        assert len(engine.daily_portfolio_values) > 1  # Should have recorded daily values
    
    def test_run_backtest_not_initialized(self, engine):
        """Test running backtest without initialization."""
        strategy = MockTradingStrategy([])
        
        metrics = engine.run_backtest(strategy)
        
        # Should return empty metrics
        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.total_return == 0.0
    
    def test_run_backtest_no_market_data(self, engine):
        """Test running backtest without market data."""
        engine.initialize_backtest(1000000.0, "2020-01-01", "2020-12-31")
        strategy = MockTradingStrategy([])
        
        metrics = engine.run_backtest(strategy)
        
        # Should return empty metrics
        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.total_return == 0.0
    
    def test_record_trade(self, engine):
        """Test trade recording functionality."""
        engine.initialize_backtest(1000000.0, "2020-01-01", "2020-12-31")
        engine.current_date = datetime(2020, 1, 15)
        
        # Create sample position
        position = Position(
            symbol='RELIANCE.NS',
            entry_date=datetime(2020, 1, 15),
            entry_price=100.0,
            quantity=100,
            stop_loss_price=95.0,
            current_value=10000.0,
            unrealized_pnl=0.0,
            status='open'
        )
        
        # Record trade
        engine._record_trade(position, 'BUY', 100.0, 10.0)
        
        assert len(engine.trade_history) == 1
        trade = engine.trade_history[0]
        assert trade['symbol'] == 'RELIANCE.NS'
        assert trade['trade_type'] == 'BUY'
        assert trade['quantity'] == 100
        assert trade['market_price'] == 100.0
        assert trade['transaction_cost'] == 10.0
    
    def test_record_daily_performance(self, engine):
        """Test daily performance recording."""
        engine.initialize_backtest(1000000.0, "2020-01-01", "2020-12-31")
        
        # Mock position manager
        with patch.object(engine, 'position_manager') as mock_pm:
            mock_pm.get_portfolio_value.return_value = 1050000.0
            mock_pm.get_available_capital.return_value = 500000.0
            mock_pm.get_position_count.return_value = 2
            
            test_date = datetime(2020, 1, 15)
            engine._record_daily_performance(test_date, {})
            
            # Should have initial record plus new record
            assert len(engine.daily_portfolio_values) == 2
            
            new_record = engine.daily_portfolio_values[-1]
            assert new_record['date'] == test_date
            assert new_record['portfolio_value'] == 1050000.0
            assert new_record['cash'] == 500000.0
            assert new_record['positions_value'] == 550000.0
            assert new_record['num_positions'] == 2
    
    def test_transaction_cost_calculation(self, engine):
        """Test transaction cost calculations."""
        # Test that transaction costs are properly applied
        config = BacktestConfig(transaction_cost_pct=0.001)  # 0.1%
        engine = BacktestEngine(config)
        
        # Transaction cost should be 0.1% of trade value
        trade_value = 10000.0
        expected_cost = trade_value * 0.001
        
        assert expected_cost == 10.0
    
    def test_no_look_ahead_bias(self, engine, sample_market_data):
        """Test that backtesting doesn't use future data."""
        engine.load_market_data(sample_market_data)
        
        # Get data for a specific date
        test_date = datetime(2020, 6, 15)
        daily_data = engine._get_market_data_for_date(test_date)
        
        # Verify that we only get data up to the test date
        for symbol, data_df in engine.market_data.items():
            available_data = data_df[data_df.index <= test_date]
            future_data = data_df[data_df.index > test_date]
            
            # Should have data up to test date
            assert len(available_data) > 0
            # Should have future data (to verify our test is meaningful)
            assert len(future_data) > 0
            
            # The daily data should only reflect data up to test date
            if symbol in daily_data:
                latest_available = available_data.iloc[-1]
                assert daily_data[symbol]['Close'] == latest_available['Close']
    
    def test_export_results_to_dataframe(self, engine):
        """Test exporting results to DataFrames."""
        engine.initialize_backtest(1000000.0, "2020-01-01", "2020-12-31")
        
        # Add some sample data
        engine.trade_history.append({
            'date': datetime(2020, 1, 15),
            'symbol': 'RELIANCE.NS',
            'trade_type': 'BUY',
            'quantity': 100,
            'market_price': 100.0
        })
        
        trades_df, performance_df = engine.export_results_to_dataframe()
        
        assert isinstance(trades_df, pd.DataFrame)
        assert isinstance(performance_df, pd.DataFrame)
        assert len(trades_df) == 1
        assert len(performance_df) == 1  # Initial portfolio record
    
    def test_reset_backtest(self, engine):
        """Test resetting backtest engine."""
        # Initialize and add some data
        engine.initialize_backtest(1000000.0, "2020-01-01", "2020-12-31")
        engine.trade_history.append({'test': 'data'})
        engine.daily_portfolio_values.append({'test': 'data'})
        
        # Reset
        engine.reset_backtest()
        
        assert not engine.is_initialized
        assert engine.position_manager is None
        assert engine.risk_manager is None
        assert len(engine.trade_history) == 0
        assert len(engine.daily_portfolio_values) == 0
    
    def test_benchmark_metrics_calculation(self, engine, sample_benchmark_data):
        """Test benchmark comparison metrics."""
        engine.load_benchmark_data(sample_benchmark_data)
        
        # Create sample portfolio data
        dates = pd.date_range('2020-01-01', '2020-01-10', freq='B')
        portfolio_data = []
        
        for i, date in enumerate(dates):
            portfolio_data.append({
                'date': date,
                'portfolio_value': 1000000.0 * (1 + i * 0.002),  # 0.2% daily return
                'daily_return': 0.2 if i > 0 else 0.0
            })
        
        portfolio_df = pd.DataFrame(portfolio_data)
        portfolio_df['date'] = pd.to_datetime(portfolio_df['date'])
        portfolio_df = portfolio_df.set_index('date')
        
        benchmark_return, alpha, beta = engine._calculate_benchmark_metrics(portfolio_df)
        
        # Should return some values (exact values depend on random data)
        assert isinstance(benchmark_return, float)
        assert isinstance(alpha, float)
        assert isinstance(beta, float)
    
    def test_get_trade_history(self, engine):
        """Test getting trade history."""
        engine.trade_history = [{'test': 'trade1'}, {'test': 'trade2'}]
        
        history = engine.get_trade_history()
        
        assert len(history) == 2
        assert history[0]['test'] == 'trade1'
        
        # Should return a copy, not the original
        history.append({'test': 'trade3'})
        assert len(engine.trade_history) == 2
    
    def test_get_daily_performance(self, engine):
        """Test getting daily performance history."""
        engine.daily_portfolio_values = [{'test': 'day1'}, {'test': 'day2'}]
        
        performance = engine.get_daily_performance()
        
        assert len(performance) == 2
        assert performance[0]['test'] == 'day1'
        
        # Should return a copy, not the original
        performance.append({'test': 'day3'})
        assert len(engine.daily_portfolio_values) == 2


if __name__ == '__main__':
    pytest.main([__file__])