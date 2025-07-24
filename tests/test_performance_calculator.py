"""
Unit tests for the PerformanceCalculator class.
Tests performance metric calculations, benchmark comparisons, and target validation.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from src.backtesting.performance_calculator import PerformanceCalculator, PerformanceTargets
from src.interfaces.trading_interfaces import PerformanceMetrics, Position


class TestPerformanceTargets:
    """Test PerformanceTargets class."""
    
    def test_default_targets(self):
        """Test default target values."""
        targets = PerformanceTargets()
        
        assert targets.target_annual_return_min == 15.0
        assert targets.target_annual_return_max == 20.0
        assert targets.target_sharpe_ratio == 1.8
        assert targets.target_max_drawdown == 8.0
        assert targets.target_win_rate_min == 60.0
        assert targets.target_win_rate_max == 65.0
    
    def test_custom_targets(self):
        """Test custom target values."""
        targets = PerformanceTargets(
            target_annual_return_min=12.0,
            target_annual_return_max=18.0,
            target_sharpe_ratio=1.5,
            target_max_drawdown=10.0,
            target_win_rate_min=55.0,
            target_win_rate_max=70.0
        )
        
        assert targets.target_annual_return_min == 12.0
        assert targets.target_annual_return_max == 18.0
        assert targets.target_sharpe_ratio == 1.5
        assert targets.target_max_drawdown == 10.0
        assert targets.target_win_rate_min == 55.0
        assert targets.target_win_rate_max == 70.0


class TestPerformanceCalculator:
    """Test PerformanceCalculator class."""
    
    @pytest.fixture
    def calculator(self):
        """Create PerformanceCalculator instance for testing."""
        return PerformanceCalculator()
    
    @pytest.fixture
    def sample_portfolio_data(self):
        """Create sample portfolio performance data."""
        dates = pd.date_range('2020-01-01', '2020-12-31', freq='B')  # Business days
        
        # Generate realistic portfolio performance
        np.random.seed(42)  # For reproducible tests
        initial_value = 1000000.0
        
        portfolio_data = []
        current_value = initial_value
        
        for i, date in enumerate(dates):
            # Simulate daily returns with slight positive bias
            daily_return = np.random.normal(0.0008, 0.015)  # ~20% annual return, 15% volatility
            current_value *= (1 + daily_return)
            
            portfolio_data.append({
                'date': date,
                'portfolio_value': current_value,
                'cash': current_value * 0.3,
                'positions_value': current_value * 0.7,
                'daily_return': daily_return * 100,
                'cumulative_return': ((current_value / initial_value) - 1) * 100
            })
        
        return portfolio_data
    
    @pytest.fixture
    def sample_trades(self):
        """Create sample trade data."""
        trades = []
        
        # Create mix of winning and losing trades
        for i in range(20):
            # 60% winning trades, 40% losing trades
            if i < 12:  # Winning trades
                pnl = np.random.uniform(500, 2000)
            else:  # Losing trades
                pnl = np.random.uniform(-1500, -200)
            
            trade = Position(
                symbol=f'STOCK{i:02d}.NS',
                entry_date=datetime(2020, 1, 1) + timedelta(days=i*10),
                entry_price=100.0 + np.random.uniform(-10, 10),
                quantity=100,
                stop_loss_price=95.0,
                current_value=10000.0 + pnl,
                unrealized_pnl=pnl,
                status='closed'
            )
            trades.append(trade)
        
        return trades
    
    @pytest.fixture
    def sample_benchmark_data(self):
        """Create sample benchmark data."""
        dates = pd.date_range('2020-01-01', '2020-12-31', freq='B')
        
        # Generate NIFTY 50 like data with lower returns than portfolio
        np.random.seed(123)
        initial_price = 12000.0
        current_price = initial_price
        
        benchmark_data = []
        for date in dates:
            daily_return = np.random.normal(0.0005, 0.012)  # ~12% annual return
            current_price *= (1 + daily_return)
            
            benchmark_data.append({
                'Open': current_price * (1 + np.random.uniform(-0.001, 0.001)),
                'High': current_price * (1 + abs(np.random.uniform(0, 0.005))),
                'Low': current_price * (1 - abs(np.random.uniform(0, 0.005))),
                'Close': current_price,
                'Volume': np.random.randint(100000000, 1000000000)
            })
        
        return pd.DataFrame(benchmark_data, index=dates)
    
    def test_initialization(self, calculator):
        """Test calculator initialization."""
        assert calculator.risk_free_rate == 0.06
        assert calculator.trading_days_per_year == 252
        assert isinstance(calculator.targets, PerformanceTargets)
    
    def test_custom_targets_initialization(self):
        """Test initialization with custom targets."""
        custom_targets = PerformanceTargets(target_sharpe_ratio=2.0)
        calculator = PerformanceCalculator(custom_targets)
        
        assert calculator.targets.target_sharpe_ratio == 2.0
    
    def test_calculate_return_metrics(self, calculator, sample_portfolio_data):
        """Test return metrics calculation."""
        portfolio_df = pd.DataFrame(sample_portfolio_data)
        portfolio_df['date'] = pd.to_datetime(portfolio_df['date'])
        portfolio_df = portfolio_df.set_index('date')
        
        metrics = calculator._calculate_return_metrics(portfolio_df, 1000000.0)
        
        assert 'total_return' in metrics
        assert 'annualized_return' in metrics
        assert 'mean_daily_return' in metrics
        assert 'volatility' in metrics
        assert 'days_traded' in metrics
        
        # Check that returns are reasonable
        assert -50 <= metrics['total_return'] <= 100  # Between -50% and 100%
        assert -50 <= metrics['annualized_return'] <= 100
        assert metrics['days_traded'] > 0
    
    def test_calculate_risk_metrics(self, calculator, sample_portfolio_data):
        """Test risk metrics calculation."""
        portfolio_df = pd.DataFrame(sample_portfolio_data)
        portfolio_df['date'] = pd.to_datetime(portfolio_df['date'])
        portfolio_df = portfolio_df.set_index('date')
        
        metrics = calculator._calculate_risk_metrics(portfolio_df)
        
        assert 'sharpe_ratio' in metrics
        assert 'max_drawdown' in metrics
        assert 'sortino_ratio' in metrics
        assert 'downside_deviation' in metrics
        assert 'var_95' in metrics
        
        # Check that metrics are reasonable
        assert -5 <= metrics['sharpe_ratio'] <= 10  # Reasonable Sharpe ratio range
        assert 0 <= metrics['max_drawdown'] <= 100  # Drawdown should be positive percentage
    
    def test_calculate_trade_metrics(self, calculator, sample_trades):
        """Test trade metrics calculation."""
        metrics = calculator._calculate_trade_metrics(sample_trades)
        
        assert 'win_rate' in metrics
        assert 'total_trades' in metrics
        assert 'winning_trades' in metrics
        assert 'losing_trades' in metrics
        assert 'average_win' in metrics
        assert 'average_loss' in metrics
        assert 'profit_factor' in metrics
        
        # Check basic trade statistics
        assert metrics['total_trades'] == 20
        assert metrics['winning_trades'] == 12  # 60% win rate as designed
        assert metrics['losing_trades'] == 8
        assert metrics['win_rate'] == 60.0
        assert metrics['average_win'] > 0
        assert metrics['average_loss'] < 0
    
    def test_calculate_trade_metrics_empty(self, calculator):
        """Test trade metrics with no trades."""
        metrics = calculator._calculate_trade_metrics([])
        
        assert metrics['win_rate'] == 0.0
        assert metrics['total_trades'] == 0
        assert metrics['average_win'] == 0.0
        assert metrics['average_loss'] == 0.0
        assert metrics['profit_factor'] == 0.0
    
    def test_calculate_benchmark_comparison(self, calculator, sample_portfolio_data, sample_benchmark_data):
        """Test benchmark comparison metrics."""
        portfolio_df = pd.DataFrame(sample_portfolio_data)
        portfolio_df['date'] = pd.to_datetime(portfolio_df['date'])
        portfolio_df = portfolio_df.set_index('date')
        
        metrics = calculator._calculate_benchmark_comparison(portfolio_df, sample_benchmark_data)
        
        assert 'benchmark_return' in metrics
        assert 'alpha' in metrics
        assert 'beta' in metrics
        assert 'information_ratio' in metrics
        assert 'tracking_error' in metrics
        assert 'correlation' in metrics
        
        # Check that metrics are reasonable
        assert -50 <= metrics['benchmark_return'] <= 100
        assert -50 <= metrics['beta'] <= 50  # Beta can be extreme with random data
        assert -1 <= metrics['correlation'] <= 1  # Correlation should be between -1 and 1
    
    def test_calculate_benchmark_comparison_no_benchmark(self, calculator, sample_portfolio_data):
        """Test benchmark comparison with no benchmark data."""
        portfolio_df = pd.DataFrame(sample_portfolio_data)
        portfolio_df['date'] = pd.to_datetime(portfolio_df['date'])
        portfolio_df = portfolio_df.set_index('date')
        
        metrics = calculator._calculate_benchmark_comparison(portfolio_df, None)
        
        assert metrics['benchmark_return'] == 0.0
        assert metrics['alpha'] == 0.0
        assert metrics['beta'] == 0.0
        assert metrics['information_ratio'] == 0.0
        assert metrics['tracking_error'] == 0.0
    
    def test_calculate_comprehensive_metrics(self, calculator, sample_portfolio_data, 
                                          sample_trades, sample_benchmark_data):
        """Test comprehensive metrics calculation."""
        metrics = calculator.calculate_comprehensive_metrics(
            sample_portfolio_data, sample_trades, sample_benchmark_data, 1000000.0
        )
        
        assert isinstance(metrics, PerformanceMetrics)
        assert hasattr(metrics, 'total_return')
        assert hasattr(metrics, 'annualized_return')
        assert hasattr(metrics, 'sharpe_ratio')
        assert hasattr(metrics, 'max_drawdown')
        assert hasattr(metrics, 'win_rate')
        assert hasattr(metrics, 'total_trades')
        assert hasattr(metrics, 'benchmark_return')
        assert hasattr(metrics, 'alpha')
        assert hasattr(metrics, 'beta')
        
        # Check that values are reasonable
        assert metrics.total_trades == 20
        assert 0 <= metrics.max_drawdown <= 100
    
    def test_calculate_comprehensive_metrics_empty(self, calculator):
        """Test comprehensive metrics with empty data."""
        metrics = calculator.calculate_comprehensive_metrics([], [], None, 1000000.0)
        
        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.total_return == 0.0
        assert metrics.annualized_return == 0.0
        assert metrics.sharpe_ratio == 0.0
        assert metrics.max_drawdown == 0.0
        assert metrics.win_rate == 0.0
        assert metrics.total_trades == 0
    
    def test_validate_performance_targets_all_pass(self, calculator):
        """Test performance validation when all targets are met."""
        # Create metrics that meet all targets
        metrics = PerformanceMetrics(
            total_return=18.0,
            annualized_return=18.0,  # Within 15-20% range
            sharpe_ratio=2.0,        # > 1.8
            max_drawdown=6.0,        # < 8%
            win_rate=62.0,           # Within 60-65% range
            total_trades=100,
            benchmark_return=12.0,
            alpha=6.0,
            beta=1.2
        )
        
        validation = calculator.validate_performance_targets(metrics)
        
        assert validation['meets_all_targets'] is True
        assert validation['summary']['targets_passed'] == 4
        assert validation['summary']['total_targets'] == 4
        assert validation['summary']['pass_rate'] == 100.0
        assert validation['summary']['overall_status'] == 'PASS'
        
        # Check individual target results
        assert validation['target_results']['annual_return']['meets_target'] is True
        assert validation['target_results']['sharpe_ratio']['meets_target'] is True
        assert validation['target_results']['max_drawdown']['meets_target'] is True
        assert validation['target_results']['win_rate']['meets_target'] is True
    
    def test_validate_performance_targets_partial_pass(self, calculator):
        """Test performance validation when some targets are not met."""
        # Create metrics that meet only some targets
        metrics = PerformanceMetrics(
            total_return=12.0,
            annualized_return=12.0,  # Below 15% minimum
            sharpe_ratio=2.2,        # > 1.8 (PASS)
            max_drawdown=10.0,       # > 8% (FAIL)
            win_rate=63.0,           # Within 60-65% range (PASS)
            total_trades=100,
            benchmark_return=10.0,
            alpha=2.0,
            beta=1.1
        )
        
        validation = calculator.validate_performance_targets(metrics)
        
        assert validation['meets_all_targets'] is False
        assert validation['summary']['targets_passed'] == 2
        assert validation['summary']['total_targets'] == 4
        assert validation['summary']['pass_rate'] == 50.0
        assert validation['summary']['overall_status'] == 'FAIL'
        
        # Check individual target results
        assert validation['target_results']['annual_return']['meets_target'] is False
        assert validation['target_results']['sharpe_ratio']['meets_target'] is True
        assert validation['target_results']['max_drawdown']['meets_target'] is False
        assert validation['target_results']['win_rate']['meets_target'] is True
    
    def test_validate_performance_targets_all_fail(self, calculator):
        """Test performance validation when no targets are met."""
        # Create metrics that meet no targets
        metrics = PerformanceMetrics(
            total_return=5.0,
            annualized_return=5.0,   # Below 15% minimum
            sharpe_ratio=0.8,        # < 1.8
            max_drawdown=15.0,       # > 8%
            win_rate=45.0,           # Below 60% minimum
            total_trades=50,
            benchmark_return=8.0,
            alpha=-3.0,
            beta=1.5
        )
        
        validation = calculator.validate_performance_targets(metrics)
        
        assert validation['meets_all_targets'] is False
        assert validation['summary']['targets_passed'] == 0
        assert validation['summary']['total_targets'] == 4
        assert validation['summary']['pass_rate'] == 0.0
        assert validation['summary']['overall_status'] == 'FAIL'
        
        # Check that all targets failed
        for target_result in validation['target_results'].values():
            assert target_result['meets_target'] is False
            assert target_result['status'] == 'FAIL'
    
    def test_generate_performance_report(self, calculator):
        """Test performance report generation."""
        metrics = PerformanceMetrics(
            total_return=18.5,
            annualized_return=18.5,
            sharpe_ratio=2.1,
            max_drawdown=6.5,
            win_rate=62.5,
            total_trades=150,
            benchmark_return=12.0,
            alpha=6.5,
            beta=1.15
        )
        
        validation_results = calculator.validate_performance_targets(metrics)
        report = calculator.generate_performance_report(metrics, validation_results)
        
        assert 'timestamp' in report
        assert 'performance_summary' in report
        assert 'benchmark_comparison' in report
        assert 'risk_assessment' in report
        assert 'performance_grade' in report
        assert 'target_validation' in report
        
        # Check performance summary
        summary = report['performance_summary']
        assert summary['total_return'] == "18.50%"
        assert summary['annualized_return'] == "18.50%"
        assert summary['sharpe_ratio'] == "2.10"
        assert summary['max_drawdown'] == "6.50%"
        assert summary['win_rate'] == "62.5%"
        assert summary['total_trades'] == 150
        
        # Check benchmark comparison
        benchmark = report['benchmark_comparison']
        assert benchmark['benchmark_return'] == "12.00%"
        assert benchmark['alpha'] == "6.50%"
        assert benchmark['beta'] == "1.15"
        assert benchmark['excess_return'] == "6.50%"
    
    def test_assess_risk_level(self, calculator):
        """Test risk level assessment."""
        # Test low risk scenario
        low_risk_metrics = PerformanceMetrics(
            total_return=20.0, annualized_return=20.0, sharpe_ratio=2.5,
            max_drawdown=3.0, win_rate=65.0, total_trades=100,
            benchmark_return=12.0, alpha=8.0, beta=1.0
        )
        
        risk_assessment = calculator._assess_risk_level(low_risk_metrics)
        assert risk_assessment['risk_level'] == "LOW"
        assert risk_assessment['sharpe_category'] == "EXCELLENT"
        assert risk_assessment['drawdown_category'] == "EXCELLENT"
        
        # Test high risk scenario
        high_risk_metrics = PerformanceMetrics(
            total_return=15.0, annualized_return=15.0, sharpe_ratio=0.5,
            max_drawdown=25.0, win_rate=45.0, total_trades=50,
            benchmark_return=12.0, alpha=3.0, beta=1.8
        )
        
        risk_assessment = calculator._assess_risk_level(high_risk_metrics)
        assert risk_assessment['risk_level'] == "HIGH"
        assert risk_assessment['sharpe_category'] == "POOR"
        assert risk_assessment['drawdown_category'] == "POOR"
    
    def test_categorize_sharpe_ratio(self, calculator):
        """Test Sharpe ratio categorization."""
        assert calculator._categorize_sharpe_ratio(3.5) == "EXCEPTIONAL"
        assert calculator._categorize_sharpe_ratio(2.5) == "EXCELLENT"
        assert calculator._categorize_sharpe_ratio(1.8) == "GOOD"
        assert calculator._categorize_sharpe_ratio(1.2) == "ACCEPTABLE"
        assert calculator._categorize_sharpe_ratio(0.5) == "POOR"
    
    def test_categorize_drawdown(self, calculator):
        """Test drawdown categorization."""
        assert calculator._categorize_drawdown(3.0) == "EXCELLENT"
        assert calculator._categorize_drawdown(7.0) == "GOOD"
        assert calculator._categorize_drawdown(12.0) == "ACCEPTABLE"
        assert calculator._categorize_drawdown(18.0) == "CONCERNING"
        assert calculator._categorize_drawdown(25.0) == "POOR"
    
    def test_calculate_performance_grade(self, calculator):
        """Test performance grade calculation."""
        # Test A+ grade (excellent performance)
        excellent_metrics = PerformanceMetrics(
            total_return=25.0, annualized_return=25.0, sharpe_ratio=2.5,
            max_drawdown=4.0, win_rate=68.0, total_trades=100,
            benchmark_return=12.0, alpha=13.0, beta=1.0
        )
        
        grade = calculator._calculate_performance_grade(excellent_metrics)
        assert grade in ["A+", "A", "A-"]  # Should be high grade
        
        # Test poor performance
        poor_metrics = PerformanceMetrics(
            total_return=2.0, annualized_return=2.0, sharpe_ratio=0.2,
            max_drawdown=30.0, win_rate=35.0, total_trades=20,
            benchmark_return=8.0, alpha=-6.0, beta=1.5
        )
        
        grade = calculator._calculate_performance_grade(poor_metrics)
        assert grade == "F"
    
    def test_export_metrics_to_dict(self, calculator):
        """Test exporting metrics to dictionary."""
        metrics = PerformanceMetrics(
            total_return=18.5,
            annualized_return=18.5,
            sharpe_ratio=2.1,
            max_drawdown=6.5,
            win_rate=62.5,
            total_trades=150,
            benchmark_return=12.0,
            alpha=6.5,
            beta=1.15
        )
        
        metrics_dict = calculator.export_metrics_to_dict(metrics)
        
        assert metrics_dict['total_return'] == 18.5
        assert metrics_dict['annualized_return'] == 18.5
        assert metrics_dict['sharpe_ratio'] == 2.1
        assert metrics_dict['max_drawdown'] == 6.5
        assert metrics_dict['win_rate'] == 62.5
        assert metrics_dict['total_trades'] == 150
        assert metrics_dict['benchmark_return'] == 12.0
        assert metrics_dict['alpha'] == 6.5
        assert metrics_dict['beta'] == 1.15
    
    def test_empty_metrics(self, calculator):
        """Test empty metrics creation."""
        empty_metrics = calculator._empty_metrics()
        
        assert isinstance(empty_metrics, PerformanceMetrics)
        assert empty_metrics.total_return == 0.0
        assert empty_metrics.annualized_return == 0.0
        assert empty_metrics.sharpe_ratio == 0.0
        assert empty_metrics.max_drawdown == 0.0
        assert empty_metrics.win_rate == 0.0
        assert empty_metrics.total_trades == 0
        assert empty_metrics.benchmark_return == 0.0
        assert empty_metrics.alpha == 0.0
        assert empty_metrics.beta == 0.0
    
    def test_error_handling(self, calculator):
        """Test error handling in various methods."""
        # Test with invalid data
        invalid_data = [{'invalid': 'data'}]
        
        metrics = calculator.calculate_comprehensive_metrics(
            invalid_data, [], None, 1000000.0
        )
        
        # Should return empty metrics on error
        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.total_return == 0.0
    
    def test_performance_with_different_time_periods(self, calculator):
        """Test performance calculation with different time periods."""
        # Test with short period (1 month)
        short_dates = pd.date_range('2020-01-01', '2020-01-31', freq='B')
        short_data = []
        
        for i, date in enumerate(short_dates):
            short_data.append({
                'date': date,
                'portfolio_value': 1000000.0 * (1 + i * 0.001),
                'daily_return': 0.1 if i > 0 else 0.0
            })
        
        portfolio_df = pd.DataFrame(short_data)
        portfolio_df['date'] = pd.to_datetime(portfolio_df['date'])
        portfolio_df = portfolio_df.set_index('date')
        
        metrics = calculator._calculate_return_metrics(portfolio_df, 1000000.0)
        
        assert 'total_return' in metrics
        assert 'annualized_return' in metrics
        # After dropna(), we lose one day due to pct_change(), so expect len-1
        assert metrics['days_traded'] == len(short_dates) - 1


if __name__ == '__main__':
    pytest.main([__file__])