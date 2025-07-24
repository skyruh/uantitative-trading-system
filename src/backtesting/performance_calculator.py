"""
Performance Calculator for the quantitative trading system.
Calculates comprehensive performance metrics and benchmark comparisons.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass

from ..interfaces.trading_interfaces import PerformanceMetrics, Position


@dataclass
class PerformanceTargets:
    """Target performance metrics for validation."""
    target_annual_return_min: float = 15.0  # 15% minimum
    target_annual_return_max: float = 20.0  # 20% maximum
    target_sharpe_ratio: float = 1.8  # >1.8 Sharpe ratio
    target_max_drawdown: float = 8.0  # <8% maximum drawdown
    target_win_rate_min: float = 60.0  # 60% minimum win rate
    target_win_rate_max: float = 65.0  # 65% maximum win rate


class PerformanceCalculator:
    """
    Advanced performance calculator with comprehensive metrics and benchmark comparison.
    
    Features:
    - Returns, Sharpe ratio, drawdown, win rate calculations
    - Benchmark comparison against NIFTY 50
    - Performance validation against target metrics
    - Risk-adjusted performance metrics
    - Statistical significance testing
    
    Requirements: 8.5, 9.1, 9.2, 9.3, 9.4, 9.5
    """
    
    def __init__(self, targets: Optional[PerformanceTargets] = None):
        """
        Initialize performance calculator.
        
        Args:
            targets: Target performance metrics for validation
        """
        self.targets = targets or PerformanceTargets()
        self.logger = logging.getLogger(__name__)
        
        # Risk-free rate for India (6% annual)
        self.risk_free_rate = 0.06
        self.trading_days_per_year = 252
        
    def calculate_comprehensive_metrics(self, 
                                      daily_portfolio_values: List[Dict],
                                      trades: List[Position],
                                      benchmark_data: Optional[pd.DataFrame] = None,
                                      initial_capital: float = 1000000.0) -> PerformanceMetrics:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            daily_portfolio_values: Daily portfolio value records
            trades: List of completed trades
            benchmark_data: Benchmark price data for comparison
            initial_capital: Initial portfolio capital
            
        Returns:
            Comprehensive performance metrics
            
        Requirements: 8.5 - Performance metric calculations
        """
        try:
            if not daily_portfolio_values:
                self.logger.warning("No portfolio data available for metrics calculation")
                return self._empty_metrics()
            
            # Convert to DataFrame for easier processing
            portfolio_df = pd.DataFrame(daily_portfolio_values)
            portfolio_df['date'] = pd.to_datetime(portfolio_df['date'])
            portfolio_df = portfolio_df.set_index('date').sort_index()
            
            # Calculate basic return metrics
            returns_metrics = self._calculate_return_metrics(portfolio_df, initial_capital)
            
            # Calculate risk metrics
            risk_metrics = self._calculate_risk_metrics(portfolio_df)
            
            # Calculate trade-based metrics
            trade_metrics = self._calculate_trade_metrics(trades)
            
            # Calculate benchmark comparison
            benchmark_metrics = self._calculate_benchmark_comparison(
                portfolio_df, benchmark_data
            )
            
            # Combine all metrics
            comprehensive_metrics = PerformanceMetrics(
                total_return=returns_metrics['total_return'],
                annualized_return=returns_metrics['annualized_return'],
                sharpe_ratio=risk_metrics['sharpe_ratio'],
                max_drawdown=risk_metrics['max_drawdown'],
                win_rate=trade_metrics['win_rate'],
                total_trades=trade_metrics['total_trades'],
                benchmark_return=benchmark_metrics['benchmark_return'],
                alpha=benchmark_metrics['alpha'],
                beta=benchmark_metrics['beta']
            )
            
            self.logger.info(f"Calculated comprehensive performance metrics: "
                           f"Return={comprehensive_metrics.total_return:.2f}%, "
                           f"Sharpe={comprehensive_metrics.sharpe_ratio:.2f}, "
                           f"Drawdown={comprehensive_metrics.max_drawdown:.2f}%")
            
            return comprehensive_metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating comprehensive metrics: {e}")
            return self._empty_metrics()
    
    def _calculate_return_metrics(self, portfolio_df: pd.DataFrame, 
                                initial_capital: float) -> Dict[str, float]:
        """
        Calculate return-based performance metrics.
        
        Args:
            portfolio_df: Portfolio performance DataFrame
            initial_capital: Initial capital amount
            
        Returns:
            Dictionary of return metrics
        """
        try:
            if len(portfolio_df) == 0:
                return {'total_return': 0.0, 'annualized_return': 0.0}
            
            # Calculate daily returns
            portfolio_df['daily_return'] = portfolio_df['portfolio_value'].pct_change()
            portfolio_df = portfolio_df.dropna()
            
            if len(portfolio_df) == 0:
                return {'total_return': 0.0, 'annualized_return': 0.0}
            
            # Total return
            final_value = portfolio_df['portfolio_value'].iloc[-1]
            initial_value = portfolio_df['portfolio_value'].iloc[0]
            total_return = (final_value / initial_value - 1) * 100
            
            # Annualized return
            days = len(portfolio_df)
            years = days / self.trading_days_per_year
            
            if years > 0:
                annualized_return = ((final_value / initial_value) ** (1/years) - 1) * 100
            else:
                annualized_return = 0.0
            
            # Additional return metrics
            mean_daily_return = portfolio_df['daily_return'].mean()
            volatility = portfolio_df['daily_return'].std()
            annualized_volatility = volatility * np.sqrt(self.trading_days_per_year) * 100
            
            return {
                'total_return': total_return,
                'annualized_return': annualized_return,
                'mean_daily_return': mean_daily_return * 100,
                'volatility': annualized_volatility,
                'days_traded': days
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating return metrics: {e}")
            return {'total_return': 0.0, 'annualized_return': 0.0}
    
    def _calculate_risk_metrics(self, portfolio_df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate risk-based performance metrics.
        
        Args:
            portfolio_df: Portfolio performance DataFrame
            
        Returns:
            Dictionary of risk metrics
        """
        try:
            if len(portfolio_df) == 0:
                return {'sharpe_ratio': 0.0, 'max_drawdown': 0.0}
            
            # Ensure daily returns are calculated
            if 'daily_return' not in portfolio_df.columns:
                portfolio_df['daily_return'] = portfolio_df['portfolio_value'].pct_change()
            
            portfolio_df = portfolio_df.dropna()
            
            if len(portfolio_df) == 0:
                return {'sharpe_ratio': 0.0, 'max_drawdown': 0.0}
            
            # Sharpe ratio calculation
            daily_risk_free = self.risk_free_rate / self.trading_days_per_year
            excess_returns = portfolio_df['daily_return'] - daily_risk_free
            
            if excess_returns.std() > 0:
                sharpe_ratio = (excess_returns.mean() / excess_returns.std() * 
                              np.sqrt(self.trading_days_per_year))
            else:
                sharpe_ratio = 0.0
            
            # Maximum drawdown calculation
            portfolio_df = portfolio_df.copy()  # Avoid SettingWithCopyWarning
            portfolio_df['cumulative_max'] = portfolio_df['portfolio_value'].expanding().max()
            portfolio_df['drawdown'] = (portfolio_df['portfolio_value'] / 
                                      portfolio_df['cumulative_max'] - 1) * 100
            max_drawdown = abs(portfolio_df['drawdown'].min())
            
            # Additional risk metrics
            downside_returns = portfolio_df['daily_return'][portfolio_df['daily_return'] < 0]
            downside_deviation = downside_returns.std() * np.sqrt(self.trading_days_per_year) * 100
            
            # Sortino ratio (using downside deviation instead of total volatility)
            if downside_deviation > 0:
                sortino_ratio = ((portfolio_df['daily_return'].mean() - daily_risk_free) * 
                               self.trading_days_per_year * 100) / downside_deviation
            else:
                sortino_ratio = 0.0
            
            # Value at Risk (95% confidence)
            var_95 = np.percentile(portfolio_df['daily_return'], 5) * 100
            
            return {
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'sortino_ratio': sortino_ratio,
                'downside_deviation': downside_deviation,
                'var_95': var_95,
                'average_drawdown': portfolio_df['drawdown'].mean()
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating risk metrics: {e}")
            return {'sharpe_ratio': 0.0, 'max_drawdown': 0.0}
    
    def _calculate_trade_metrics(self, trades: List[Position]) -> Dict[str, float]:
        """
        Calculate trade-based performance metrics.
        
        Args:
            trades: List of completed trades
            
        Returns:
            Dictionary of trade metrics
        """
        try:
            if not trades:
                return {
                    'win_rate': 0.0,
                    'total_trades': 0,
                    'average_win': 0.0,
                    'average_loss': 0.0,
                    'profit_factor': 0.0
                }
            
            # Separate winning and losing trades
            winning_trades = [trade for trade in trades if trade.unrealized_pnl > 0]
            losing_trades = [trade for trade in trades if trade.unrealized_pnl <= 0]
            
            # Basic trade statistics
            total_trades = len(trades)
            win_count = len(winning_trades)
            loss_count = len(losing_trades)
            
            win_rate = (win_count / total_trades) * 100 if total_trades > 0 else 0.0
            
            # Average win/loss amounts
            average_win = (np.mean([trade.unrealized_pnl for trade in winning_trades]) 
                          if winning_trades else 0.0)
            average_loss = (np.mean([trade.unrealized_pnl for trade in losing_trades]) 
                           if losing_trades else 0.0)
            
            # Profit factor (gross profit / gross loss)
            gross_profit = sum(trade.unrealized_pnl for trade in winning_trades)
            gross_loss = abs(sum(trade.unrealized_pnl for trade in losing_trades))
            
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0
            
            # Additional trade metrics
            largest_win = max([trade.unrealized_pnl for trade in trades]) if trades else 0.0
            largest_loss = min([trade.unrealized_pnl for trade in trades]) if trades else 0.0
            
            # Average trade duration (if entry_date is available)
            trade_durations = []
            for trade in trades:
                if hasattr(trade, 'entry_date') and hasattr(trade, 'exit_date'):
                    if trade.exit_date and trade.entry_date:
                        duration = (trade.exit_date - trade.entry_date).days
                        trade_durations.append(duration)
            
            average_trade_duration = (np.mean(trade_durations) 
                                    if trade_durations else 0.0)
            
            return {
                'win_rate': win_rate,
                'total_trades': total_trades,
                'winning_trades': win_count,
                'losing_trades': loss_count,
                'average_win': average_win,
                'average_loss': average_loss,
                'profit_factor': profit_factor,
                'largest_win': largest_win,
                'largest_loss': largest_loss,
                'gross_profit': gross_profit,
                'gross_loss': gross_loss,
                'average_trade_duration': average_trade_duration
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating trade metrics: {e}")
            return {'win_rate': 0.0, 'total_trades': 0}
    
    def _calculate_benchmark_comparison(self, portfolio_df: pd.DataFrame,
                                      benchmark_data: Optional[pd.DataFrame]) -> Dict[str, float]:
        """
        Calculate benchmark comparison metrics.
        
        Args:
            portfolio_df: Portfolio performance DataFrame
            benchmark_data: Benchmark price data
            
        Returns:
            Dictionary of benchmark comparison metrics
            
        Requirements: 9.5 - Benchmark comparison against NIFTY 50
        """
        try:
            if benchmark_data is None or len(portfolio_df) == 0:
                return {
                    'benchmark_return': 0.0,
                    'alpha': 0.0,
                    'beta': 0.0,
                    'information_ratio': 0.0,
                    'tracking_error': 0.0
                }
            
            # Align benchmark data with portfolio dates
            benchmark_aligned = benchmark_data.reindex(portfolio_df.index, method='ffill')
            benchmark_aligned = benchmark_aligned.dropna()
            
            if len(benchmark_aligned) == 0:
                return {
                    'benchmark_return': 0.0,
                    'alpha': 0.0,
                    'beta': 0.0,
                    'information_ratio': 0.0,
                    'tracking_error': 0.0
                }
            
            # Calculate benchmark returns
            benchmark_returns = benchmark_aligned['Close'].pct_change().dropna()
            
            # Calculate benchmark total return
            benchmark_total_return = ((benchmark_aligned['Close'].iloc[-1] / 
                                     benchmark_aligned['Close'].iloc[0] - 1) * 100)
            
            # Ensure portfolio returns are calculated
            if 'daily_return' not in portfolio_df.columns:
                portfolio_df['daily_return'] = portfolio_df['portfolio_value'].pct_change()
            
            portfolio_returns = portfolio_df['daily_return'].dropna()
            
            # Align returns for comparison
            common_dates = portfolio_returns.index.intersection(benchmark_returns.index)
            
            if len(common_dates) <= 1:
                return {
                    'benchmark_return': benchmark_total_return,
                    'alpha': 0.0,
                    'beta': 0.0,
                    'information_ratio': 0.0,
                    'tracking_error': 0.0
                }
            
            port_returns_aligned = portfolio_returns.loc[common_dates]
            bench_returns_aligned = benchmark_returns.loc[common_dates]
            
            # Calculate beta (covariance / variance)
            covariance = np.cov(port_returns_aligned, bench_returns_aligned)[0, 1]
            benchmark_variance = np.var(bench_returns_aligned)
            beta = covariance / benchmark_variance if benchmark_variance > 0 else 0.0
            
            # Calculate alpha (CAPM)
            daily_risk_free = self.risk_free_rate / self.trading_days_per_year
            
            portfolio_excess_return = port_returns_aligned.mean() - daily_risk_free
            benchmark_excess_return = bench_returns_aligned.mean() - daily_risk_free
            
            alpha = (portfolio_excess_return - beta * benchmark_excess_return) * self.trading_days_per_year
            
            # Information ratio and tracking error
            active_returns = port_returns_aligned - bench_returns_aligned
            tracking_error = active_returns.std() * np.sqrt(self.trading_days_per_year) * 100
            
            information_ratio = (active_returns.mean() * self.trading_days_per_year * 100 / 
                               tracking_error) if tracking_error > 0 else 0.0
            
            # Correlation
            correlation = np.corrcoef(port_returns_aligned, bench_returns_aligned)[0, 1]
            
            return {
                'benchmark_return': benchmark_total_return,
                'alpha': alpha * 100,  # Convert to percentage
                'beta': beta,
                'information_ratio': information_ratio,
                'tracking_error': tracking_error,
                'correlation': correlation,
                'excess_return': (port_returns_aligned.mean() - bench_returns_aligned.mean()) * self.trading_days_per_year * 100
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating benchmark comparison: {e}")
            return {
                'benchmark_return': 0.0,
                'alpha': 0.0,
                'beta': 0.0,
                'information_ratio': 0.0,
                'tracking_error': 0.0
            }
    
    def validate_performance_targets(self, metrics: PerformanceMetrics) -> Dict[str, Any]:
        """
        Validate performance against target metrics.
        
        Args:
            metrics: Performance metrics to validate
            
        Returns:
            Dictionary with validation results
            
        Requirements: 9.1, 9.2, 9.3, 9.4 - Target validation
        """
        try:
            validation_results = {
                'meets_all_targets': True,
                'target_results': {},
                'summary': {}
            }
            
            # Annual return validation (15-20%)
            annual_return_valid = (self.targets.target_annual_return_min <= 
                                 metrics.annualized_return <= 
                                 self.targets.target_annual_return_max)
            
            validation_results['target_results']['annual_return'] = {
                'target_range': f"{self.targets.target_annual_return_min}-{self.targets.target_annual_return_max}%",
                'actual': f"{metrics.annualized_return:.2f}%",
                'meets_target': annual_return_valid,
                'status': 'PASS' if annual_return_valid else 'FAIL'
            }
            
            # Sharpe ratio validation (>1.8)
            sharpe_valid = metrics.sharpe_ratio > self.targets.target_sharpe_ratio
            
            validation_results['target_results']['sharpe_ratio'] = {
                'target': f">{self.targets.target_sharpe_ratio}",
                'actual': f"{metrics.sharpe_ratio:.2f}",
                'meets_target': sharpe_valid,
                'status': 'PASS' if sharpe_valid else 'FAIL'
            }
            
            # Maximum drawdown validation (<8%)
            drawdown_valid = metrics.max_drawdown < self.targets.target_max_drawdown
            
            validation_results['target_results']['max_drawdown'] = {
                'target': f"<{self.targets.target_max_drawdown}%",
                'actual': f"{metrics.max_drawdown:.2f}%",
                'meets_target': drawdown_valid,
                'status': 'PASS' if drawdown_valid else 'FAIL'
            }
            
            # Win rate validation (60-65%)
            win_rate_valid = (self.targets.target_win_rate_min <= 
                            metrics.win_rate <= 
                            self.targets.target_win_rate_max)
            
            validation_results['target_results']['win_rate'] = {
                'target_range': f"{self.targets.target_win_rate_min}-{self.targets.target_win_rate_max}%",
                'actual': f"{metrics.win_rate:.1f}%",
                'meets_target': win_rate_valid,
                'status': 'PASS' if win_rate_valid else 'FAIL'
            }
            
            # Overall validation
            all_targets_met = all([
                annual_return_valid,
                sharpe_valid,
                drawdown_valid,
                win_rate_valid
            ])
            
            validation_results['meets_all_targets'] = all_targets_met
            
            # Summary statistics
            targets_passed = sum([
                annual_return_valid,
                sharpe_valid,
                drawdown_valid,
                win_rate_valid
            ])
            
            validation_results['summary'] = {
                'targets_passed': targets_passed,
                'total_targets': 4,
                'pass_rate': (targets_passed / 4) * 100,
                'overall_status': 'PASS' if all_targets_met else 'FAIL'
            }
            
            self.logger.info(f"Performance validation: {targets_passed}/4 targets met "
                           f"({validation_results['summary']['pass_rate']:.1f}%)")
            
            return validation_results
            
        except Exception as e:
            self.logger.error(f"Error validating performance targets: {e}")
            return {
                'meets_all_targets': False,
                'target_results': {},
                'summary': {'error': str(e)}
            }
    
    def generate_performance_report(self, metrics: PerformanceMetrics,
                                  validation_results: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Generate comprehensive performance report.
        
        Args:
            metrics: Performance metrics
            validation_results: Target validation results
            
        Returns:
            Comprehensive performance report
        """
        try:
            report = {
                'timestamp': datetime.now().isoformat(),
                'performance_summary': {
                    'total_return': f"{metrics.total_return:.2f}%",
                    'annualized_return': f"{metrics.annualized_return:.2f}%",
                    'sharpe_ratio': f"{metrics.sharpe_ratio:.2f}",
                    'max_drawdown': f"{metrics.max_drawdown:.2f}%",
                    'win_rate': f"{metrics.win_rate:.1f}%",
                    'total_trades': metrics.total_trades
                },
                'benchmark_comparison': {
                    'benchmark_return': f"{metrics.benchmark_return:.2f}%",
                    'alpha': f"{metrics.alpha:.2f}%",
                    'beta': f"{metrics.beta:.2f}",
                    'excess_return': f"{metrics.total_return - metrics.benchmark_return:.2f}%"
                },
                'risk_assessment': self._assess_risk_level(metrics),
                'performance_grade': self._calculate_performance_grade(metrics)
            }
            
            if validation_results:
                report['target_validation'] = validation_results
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating performance report: {e}")
            return {'error': str(e)}
    
    def _assess_risk_level(self, metrics: PerformanceMetrics) -> Dict[str, str]:
        """Assess risk level based on metrics."""
        try:
            # Risk assessment based on Sharpe ratio and drawdown
            if metrics.sharpe_ratio > 2.0 and metrics.max_drawdown < 5.0:
                risk_level = "LOW"
                risk_description = "Excellent risk-adjusted returns with minimal drawdown"
            elif metrics.sharpe_ratio > 1.5 and metrics.max_drawdown < 10.0:
                risk_level = "MODERATE"
                risk_description = "Good risk-adjusted returns with acceptable drawdown"
            elif metrics.sharpe_ratio > 1.0 and metrics.max_drawdown < 15.0:
                risk_level = "MODERATE-HIGH"
                risk_description = "Reasonable returns but with elevated risk"
            else:
                risk_level = "HIGH"
                risk_description = "High risk with potentially inadequate compensation"
            
            return {
                'risk_level': risk_level,
                'description': risk_description,
                'sharpe_category': self._categorize_sharpe_ratio(metrics.sharpe_ratio),
                'drawdown_category': self._categorize_drawdown(metrics.max_drawdown)
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _categorize_sharpe_ratio(self, sharpe_ratio: float) -> str:
        """Categorize Sharpe ratio performance."""
        if sharpe_ratio > 3.0:
            return "EXCEPTIONAL"
        elif sharpe_ratio > 2.0:
            return "EXCELLENT"
        elif sharpe_ratio > 1.5:
            return "GOOD"
        elif sharpe_ratio > 1.0:
            return "ACCEPTABLE"
        else:
            return "POOR"
    
    def _categorize_drawdown(self, max_drawdown: float) -> str:
        """Categorize maximum drawdown performance."""
        if max_drawdown < 5.0:
            return "EXCELLENT"
        elif max_drawdown < 10.0:
            return "GOOD"
        elif max_drawdown < 15.0:
            return "ACCEPTABLE"
        elif max_drawdown < 20.0:
            return "CONCERNING"
        else:
            return "POOR"
    
    def _calculate_performance_grade(self, metrics: PerformanceMetrics) -> str:
        """Calculate overall performance grade."""
        try:
            score = 0
            
            # Return score (0-25 points)
            if metrics.annualized_return >= 20.0:
                score += 25
            elif metrics.annualized_return >= 15.0:
                score += 20
            elif metrics.annualized_return >= 10.0:
                score += 15
            elif metrics.annualized_return >= 5.0:
                score += 10
            
            # Sharpe ratio score (0-25 points)
            if metrics.sharpe_ratio >= 2.0:
                score += 25
            elif metrics.sharpe_ratio >= 1.5:
                score += 20
            elif metrics.sharpe_ratio >= 1.0:
                score += 15
            elif metrics.sharpe_ratio >= 0.5:
                score += 10
            
            # Drawdown score (0-25 points)
            if metrics.max_drawdown <= 5.0:
                score += 25
            elif metrics.max_drawdown <= 8.0:
                score += 20
            elif metrics.max_drawdown <= 12.0:
                score += 15
            elif metrics.max_drawdown <= 20.0:
                score += 10
            
            # Win rate score (0-25 points)
            if metrics.win_rate >= 65.0:
                score += 25
            elif metrics.win_rate >= 60.0:
                score += 20
            elif metrics.win_rate >= 55.0:
                score += 15
            elif metrics.win_rate >= 50.0:
                score += 10
            
            # Convert to letter grade
            if score >= 90:
                return "A+"
            elif score >= 85:
                return "A"
            elif score >= 80:
                return "A-"
            elif score >= 75:
                return "B+"
            elif score >= 70:
                return "B"
            elif score >= 65:
                return "B-"
            elif score >= 60:
                return "C+"
            elif score >= 55:
                return "C"
            elif score >= 50:
                return "C-"
            else:
                return "F"
                
        except Exception as e:
            self.logger.error(f"Error calculating performance grade: {e}")
            return "N/A"
    
    def _empty_metrics(self) -> PerformanceMetrics:
        """Return empty performance metrics."""
        return PerformanceMetrics(
            total_return=0.0,
            annualized_return=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            win_rate=0.0,
            total_trades=0,
            benchmark_return=0.0,
            alpha=0.0,
            beta=0.0
        )
    
    def export_metrics_to_dict(self, metrics: PerformanceMetrics) -> Dict[str, Any]:
        """Export performance metrics to dictionary format."""
        return {
            'total_return': metrics.total_return,
            'annualized_return': metrics.annualized_return,
            'sharpe_ratio': metrics.sharpe_ratio,
            'max_drawdown': metrics.max_drawdown,
            'win_rate': metrics.win_rate,
            'total_trades': metrics.total_trades,
            'benchmark_return': metrics.benchmark_return,
            'alpha': metrics.alpha,
            'beta': metrics.beta
        }