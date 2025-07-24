"""
Visualization system for performance monitoring and analysis.
Implements comprehensive plotting and charting for trading system performance.
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid TCL/TK issues
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
import os
from dataclasses import asdict

from ..interfaces.trading_interfaces import PerformanceMetrics, Position, TradingSignal
from ..monitoring.performance_tracker import PerformanceSnapshot, TradeLog
from ..utils.logging_utils import get_logger


class PerformanceVisualizer:
    """
    Comprehensive visualization system for trading performance analysis.
    
    Features:
    - Cumulative returns plotting with benchmark comparison
    - Trade signal visualization with sentiment scores
    - Drawdown and Sharpe ratio time series plots
    - Interactive performance dashboards
    - Export capabilities for reports
    
    Requirements: 9.6
    """
    
    def __init__(self, output_dir: str = "data/visualizations"):
        """
        Initialize performance visualizer.
        
        Args:
            output_dir: Directory to save visualization outputs
        """
        self.output_dir = output_dir
        self.logger = get_logger("PerformanceVisualizer")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up plotting style
        self._setup_plotting_style()
        
        self.logger.info(f"Performance visualizer initialized with output dir: {output_dir}")
    
    def _setup_plotting_style(self):
        """Set up consistent plotting style."""
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Set default figure parameters
        plt.rcParams.update({
            'figure.figsize': (12, 8),
            'font.size': 10,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 16
        })
    
    def plot_cumulative_returns(self, portfolio_values: List[Dict], 
                              benchmark_data: Optional[pd.DataFrame] = None,
                              save_path: Optional[str] = None) -> str:
        """
        Plot cumulative returns with benchmark comparison.
        
        Args:
            portfolio_values: Daily portfolio value records
            benchmark_data: Benchmark price data for comparison
            save_path: Optional custom save path
            
        Returns:
            Path to saved plot
            
        Requirements: 9.6 - Cumulative returns plotting with benchmark comparison
        """
        try:
            if not portfolio_values:
                self.logger.warning("No portfolio data available for cumulative returns plot")
                return ""
            
            # Convert to DataFrame
            portfolio_df = pd.DataFrame(portfolio_values)
            portfolio_df['date'] = pd.to_datetime(portfolio_df['date'])
            portfolio_df = portfolio_df.set_index('date').sort_index()
            
            # Calculate cumulative returns
            initial_value = portfolio_df['portfolio_value'].iloc[0]
            portfolio_df['cumulative_return'] = (portfolio_df['portfolio_value'] / initial_value - 1) * 100
            
            # Create figure
            fig, ax = plt.subplots(figsize=(14, 8))
            
            # Plot portfolio cumulative returns
            ax.plot(portfolio_df.index, portfolio_df['cumulative_return'], 
                   linewidth=2, label='Portfolio', color='#2E86AB')
            
            # Plot benchmark if available
            if benchmark_data is not None:
                benchmark_aligned = benchmark_data.reindex(portfolio_df.index, method='ffill')
                benchmark_aligned = benchmark_aligned.dropna()
                
                if len(benchmark_aligned) > 0:
                    initial_benchmark = benchmark_aligned['Close'].iloc[0]
                    benchmark_returns = (benchmark_aligned['Close'] / initial_benchmark - 1) * 100
                    
                    ax.plot(benchmark_returns.index, benchmark_returns, 
                           linewidth=2, label='NIFTY 50 Benchmark', color='#A23B72', alpha=0.8)
            
            # Formatting
            ax.set_title('Cumulative Returns Comparison', fontsize=16, fontweight='bold')
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Cumulative Return (%)', fontsize=12)
            ax.legend(loc='upper left')
            ax.grid(True, alpha=0.3)
            
            # Format x-axis
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            plt.xticks(rotation=45)
            
            # Add zero line
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            # Add performance statistics text box
            if len(portfolio_df) > 0:
                final_return = portfolio_df['cumulative_return'].iloc[-1]
                max_return = portfolio_df['cumulative_return'].max()
                min_return = portfolio_df['cumulative_return'].min()
                
                stats_text = f'Final Return: {final_return:.2f}%\nMax Return: {max_return:.2f}%\nMax Drawdown: {abs(min_return):.2f}%'
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            plt.tight_layout()
            
            # Save plot
            if save_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                save_path = os.path.join(self.output_dir, f"cumulative_returns_{timestamp}.png")
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Cumulative returns plot saved to: {save_path}")
            return save_path
            
        except Exception as e:
            self.logger.error(f"Error creating cumulative returns plot: {e}")
            return ""
    
    def plot_trade_signals_with_sentiment(self, price_data: pd.DataFrame, 
                                        trade_logs: List[TradeLog],
                                        symbol: str,
                                        save_path: Optional[str] = None) -> str:
        """
        Plot trade signals with sentiment scores on price charts.
        
        Args:
            price_data: Stock price data (OHLCV)
            trade_logs: List of trade log entries
            symbol: Stock symbol for the chart
            save_path: Optional custom save path
            
        Returns:
            Path to saved plot
            
        Requirements: 9.6 - Trade signal visualization with sentiment scores on price charts
        """
        try:
            if price_data.empty or not trade_logs:
                self.logger.warning(f"Insufficient data for trade signals plot for {symbol}")
                return ""
            
            # Filter trade logs for this symbol
            symbol_trades = [trade for trade in trade_logs if trade.symbol == symbol]
            
            if not symbol_trades:
                self.logger.warning(f"No trades found for symbol {symbol}")
                return ""
            
            # Create figure with subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), 
                                         gridspec_kw={'height_ratios': [3, 1]})
            
            # Plot price data
            ax1.plot(price_data.index, price_data['Close'], linewidth=1.5, 
                    label='Close Price', color='#2E86AB')
            
            # Plot moving averages if available
            if 'SMA_50' in price_data.columns:
                ax1.plot(price_data.index, price_data['SMA_50'], linewidth=1, 
                        label='SMA 50', color='orange', alpha=0.7)
            
            # Plot Bollinger Bands if available
            if all(col in price_data.columns for col in ['BB_Upper', 'BB_Lower']):
                ax1.fill_between(price_data.index, price_data['BB_Upper'], price_data['BB_Lower'],
                               alpha=0.2, color='gray', label='Bollinger Bands')
            
            # Plot trade signals
            buy_trades = [trade for trade in symbol_trades if trade.action == 'buy']
            sell_trades = [trade for trade in symbol_trades if trade.action == 'sell']
            
            if buy_trades:
                buy_dates = [trade.timestamp for trade in buy_trades]
                buy_prices = [trade.price for trade in buy_trades]
                buy_sentiments = [trade.sentiment_score for trade in buy_trades]
                
                # Color code by sentiment (green for positive, red for negative)
                buy_colors = ['green' if s > 0 else 'red' if s < 0 else 'gray' for s in buy_sentiments]
                
                ax1.scatter(buy_dates, buy_prices, marker='^', s=100, 
                          c=buy_colors, label='Buy Signals', alpha=0.8, edgecolors='black')
            
            if sell_trades:
                sell_dates = [trade.timestamp for trade in sell_trades]
                sell_prices = [trade.price for trade in sell_trades]
                sell_sentiments = [trade.sentiment_score for trade in sell_trades]
                
                # Color code by sentiment
                sell_colors = ['green' if s > 0 else 'red' if s < 0 else 'gray' for s in sell_sentiments]
                
                ax1.scatter(sell_dates, sell_prices, marker='v', s=100, 
                          c=sell_colors, label='Sell Signals', alpha=0.8, edgecolors='black')
            
            # Format price chart
            ax1.set_title(f'{symbol} - Price Chart with Trade Signals', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Price ($)', fontsize=12)
            ax1.legend(loc='upper left')
            ax1.grid(True, alpha=0.3)
            
            # Plot sentiment scores over time
            sentiment_dates = [trade.timestamp for trade in symbol_trades]
            sentiment_scores = [trade.sentiment_score for trade in symbol_trades]
            
            ax2.scatter(sentiment_dates, sentiment_scores, c=sentiment_scores, 
                       cmap='RdYlGn', s=50, alpha=0.7)
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            ax2.set_ylabel('Sentiment Score', fontsize=12)
            ax2.set_xlabel('Date', fontsize=12)
            ax2.set_title('Sentiment Scores Over Time', fontsize=12)
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(-1.1, 1.1)
            
            # Format x-axis for both subplots
            for ax in [ax1, ax2]:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            
            plt.tight_layout()
            
            # Save plot
            if save_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                save_path = os.path.join(self.output_dir, f"trade_signals_{symbol}_{timestamp}.png")
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Trade signals plot for {symbol} saved to: {save_path}")
            return save_path
            
        except Exception as e:
            self.logger.error(f"Error creating trade signals plot for {symbol}: {e}")
            return ""
    
    def plot_drawdown_analysis(self, portfolio_values: List[Dict],
                             save_path: Optional[str] = None) -> str:
        """
        Plot drawdown analysis over time.
        
        Args:
            portfolio_values: Daily portfolio value records
            save_path: Optional custom save path
            
        Returns:
            Path to saved plot
            
        Requirements: 9.6 - Drawdown time series plots
        """
        try:
            if not portfolio_values:
                self.logger.warning("No portfolio data available for drawdown analysis")
                return ""
            
            # Convert to DataFrame
            portfolio_df = pd.DataFrame(portfolio_values)
            portfolio_df['date'] = pd.to_datetime(portfolio_df['date'])
            portfolio_df = portfolio_df.set_index('date').sort_index()
            
            # Calculate drawdown
            portfolio_df['cumulative_max'] = portfolio_df['portfolio_value'].expanding().max()
            portfolio_df['drawdown'] = (portfolio_df['portfolio_value'] / portfolio_df['cumulative_max'] - 1) * 100
            
            # Create figure
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
            
            # Plot portfolio value and running maximum
            ax1.plot(portfolio_df.index, portfolio_df['portfolio_value'], 
                    linewidth=2, label='Portfolio Value', color='#2E86AB')
            ax1.plot(portfolio_df.index, portfolio_df['cumulative_max'], 
                    linewidth=1, label='Running Maximum', color='red', alpha=0.7)
            
            ax1.set_title('Portfolio Value and Running Maximum', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot drawdown
            ax2.fill_between(portfolio_df.index, portfolio_df['drawdown'], 0, 
                           color='red', alpha=0.3, label='Drawdown')
            ax2.plot(portfolio_df.index, portfolio_df['drawdown'], 
                    linewidth=1, color='red')
            
            ax2.set_title('Drawdown Analysis', fontsize=14, fontweight='bold')
            ax2.set_ylabel('Drawdown (%)', fontsize=12)
            ax2.set_xlabel('Date', fontsize=12)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Add statistics
            max_drawdown = abs(portfolio_df['drawdown'].min())
            avg_drawdown = abs(portfolio_df['drawdown'].mean())
            
            stats_text = f'Max Drawdown: {max_drawdown:.2f}%\nAvg Drawdown: {avg_drawdown:.2f}%'
            ax2.text(0.02, 0.02, stats_text, transform=ax2.transAxes, 
                    verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            # Format x-axis
            for ax in [ax1, ax2]:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            
            plt.tight_layout()
            
            # Save plot
            if save_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                save_path = os.path.join(self.output_dir, f"drawdown_analysis_{timestamp}.png")
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Drawdown analysis plot saved to: {save_path}")
            return save_path
            
        except Exception as e:
            self.logger.error(f"Error creating drawdown analysis plot: {e}")
            return ""
    
    def plot_sharpe_ratio_evolution(self, performance_snapshots: List[PerformanceSnapshot],
                                  save_path: Optional[str] = None) -> str:
        """
        Plot Sharpe ratio evolution over time.
        
        Args:
            performance_snapshots: List of performance snapshots
            save_path: Optional custom save path
            
        Returns:
            Path to saved plot
            
        Requirements: 9.6 - Sharpe ratio time series plots
        """
        try:
            if not performance_snapshots:
                self.logger.warning("No performance snapshots available for Sharpe ratio plot")
                return ""
            
            # Convert to DataFrame
            snapshots_data = [asdict(snapshot) for snapshot in performance_snapshots]
            snapshots_df = pd.DataFrame(snapshots_data)
            snapshots_df['timestamp'] = pd.to_datetime(snapshots_df['timestamp'])
            snapshots_df = snapshots_df.set_index('timestamp').sort_index()
            
            # Create figure
            fig, ax = plt.subplots(figsize=(14, 8))
            
            # Plot Sharpe ratio evolution
            ax.plot(snapshots_df.index, snapshots_df['sharpe_ratio'], 
                   linewidth=2, label='Sharpe Ratio', color='#2E86AB')
            
            # Add target line
            target_sharpe = 1.8
            ax.axhline(y=target_sharpe, color='green', linestyle='--', 
                      label=f'Target Sharpe Ratio ({target_sharpe})', alpha=0.7)
            
            # Add warning line
            warning_sharpe = 1.0
            ax.axhline(y=warning_sharpe, color='orange', linestyle='--', 
                      label=f'Warning Level ({warning_sharpe})', alpha=0.7)
            
            # Highlight periods below target
            below_target = snapshots_df['sharpe_ratio'] < target_sharpe
            if below_target.any():
                ax.fill_between(snapshots_df.index, snapshots_df['sharpe_ratio'], target_sharpe,
                              where=below_target, color='red', alpha=0.2, 
                              label='Below Target')
            
            # Formatting
            ax.set_title('Sharpe Ratio Evolution Over Time', fontsize=16, fontweight='bold')
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Sharpe Ratio', fontsize=12)
            ax.legend(loc='upper left')
            ax.grid(True, alpha=0.3)
            
            # Format x-axis
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
            plt.xticks(rotation=45)
            
            # Add statistics
            if len(snapshots_df) > 0:
                current_sharpe = snapshots_df['sharpe_ratio'].iloc[-1]
                max_sharpe = snapshots_df['sharpe_ratio'].max()
                min_sharpe = snapshots_df['sharpe_ratio'].min()
                avg_sharpe = snapshots_df['sharpe_ratio'].mean()
                
                stats_text = f'Current: {current_sharpe:.2f}\nMax: {max_sharpe:.2f}\nMin: {min_sharpe:.2f}\nAvg: {avg_sharpe:.2f}'
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            plt.tight_layout()
            
            # Save plot
            if save_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                save_path = os.path.join(self.output_dir, f"sharpe_ratio_evolution_{timestamp}.png")
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Sharpe ratio evolution plot saved to: {save_path}")
            return save_path
            
        except Exception as e:
            self.logger.error(f"Error creating Sharpe ratio evolution plot: {e}")
            return ""
    
    def create_performance_dashboard(self, portfolio_values: List[Dict],
                                   performance_snapshots: List[PerformanceSnapshot],
                                   trade_logs: List[TradeLog],
                                   benchmark_data: Optional[pd.DataFrame] = None,
                                   save_path: Optional[str] = None) -> str:
        """
        Create comprehensive performance dashboard.
        
        Args:
            portfolio_values: Daily portfolio value records
            performance_snapshots: List of performance snapshots
            trade_logs: List of trade log entries
            benchmark_data: Benchmark data for comparison
            save_path: Optional custom save path
            
        Returns:
            Path to saved dashboard
        """
        try:
            # Create figure with subplots
            fig = plt.figure(figsize=(20, 16))
            
            # Define subplot layout
            gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
            
            # 1. Cumulative Returns (top row, spans 2 columns)
            ax1 = fig.add_subplot(gs[0, :2])
            self._plot_cumulative_returns_subplot(ax1, portfolio_values, benchmark_data)
            
            # 2. Key Metrics Summary (top right)
            ax2 = fig.add_subplot(gs[0, 2])
            self._plot_metrics_summary_subplot(ax2, performance_snapshots)
            
            # 3. Drawdown Analysis (middle left)
            ax3 = fig.add_subplot(gs[1, 0])
            self._plot_drawdown_subplot(ax3, portfolio_values)
            
            # 4. Sharpe Ratio Evolution (middle center)
            ax4 = fig.add_subplot(gs[1, 1])
            self._plot_sharpe_subplot(ax4, performance_snapshots)
            
            # 5. Trade Distribution (middle right)
            ax5 = fig.add_subplot(gs[1, 2])
            self._plot_trade_distribution_subplot(ax5, trade_logs)
            
            # 6. Monthly Returns Heatmap (bottom row, spans all columns)
            ax6 = fig.add_subplot(gs[2, :])
            self._plot_monthly_returns_heatmap_subplot(ax6, portfolio_values)
            
            # Add main title
            fig.suptitle('Trading System Performance Dashboard', fontsize=20, fontweight='bold')
            
            # Save dashboard
            if save_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                save_path = os.path.join(self.output_dir, f"performance_dashboard_{timestamp}.png")
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Performance dashboard saved to: {save_path}")
            return save_path
            
        except Exception as e:
            self.logger.error(f"Error creating performance dashboard: {e}")
            return ""
    
    def _plot_cumulative_returns_subplot(self, ax, portfolio_values: List[Dict], 
                                       benchmark_data: Optional[pd.DataFrame]):
        """Plot cumulative returns in subplot."""
        if not portfolio_values:
            ax.text(0.5, 0.5, 'No Data Available', ha='center', va='center', transform=ax.transAxes)
            return
        
        portfolio_df = pd.DataFrame(portfolio_values)
        portfolio_df['date'] = pd.to_datetime(portfolio_df['date'])
        portfolio_df = portfolio_df.set_index('date').sort_index()
        
        initial_value = portfolio_df['portfolio_value'].iloc[0]
        portfolio_df['cumulative_return'] = (portfolio_df['portfolio_value'] / initial_value - 1) * 100
        
        ax.plot(portfolio_df.index, portfolio_df['cumulative_return'], 
               linewidth=2, label='Portfolio', color='#2E86AB')
        
        if benchmark_data is not None:
            benchmark_aligned = benchmark_data.reindex(portfolio_df.index, method='ffill').dropna()
            if len(benchmark_aligned) > 0:
                initial_benchmark = benchmark_aligned['Close'].iloc[0]
                benchmark_returns = (benchmark_aligned['Close'] / initial_benchmark - 1) * 100
                ax.plot(benchmark_returns.index, benchmark_returns, 
                       linewidth=2, label='Benchmark', color='#A23B72', alpha=0.8)
        
        ax.set_title('Cumulative Returns', fontweight='bold')
        ax.set_ylabel('Return (%)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_metrics_summary_subplot(self, ax, performance_snapshots: List[PerformanceSnapshot]):
        """Plot key metrics summary in subplot."""
        ax.axis('off')
        
        if not performance_snapshots:
            ax.text(0.5, 0.5, 'No Metrics Available', ha='center', va='center', transform=ax.transAxes)
            return
        
        latest = performance_snapshots[-1]
        
        metrics_text = f"""
        Portfolio Value: ${latest.portfolio_value:,.0f}
        Total Return: {latest.total_return:.2f}%
        Daily Return: {latest.daily_return:.2f}%
        Sharpe Ratio: {latest.sharpe_ratio:.2f}
        Max Drawdown: {latest.max_drawdown:.2f}%
        Win Rate: {latest.win_rate:.1f}%
        Total Trades: {latest.total_trades}
        Open Positions: {latest.open_positions}
        """
        
        ax.text(0.1, 0.9, 'Key Metrics', fontsize=14, fontweight='bold', transform=ax.transAxes)
        ax.text(0.1, 0.1, metrics_text, fontsize=10, transform=ax.transAxes, 
               verticalalignment='bottom', fontfamily='monospace')
    
    def _plot_drawdown_subplot(self, ax, portfolio_values: List[Dict]):
        """Plot drawdown in subplot."""
        if not portfolio_values:
            ax.text(0.5, 0.5, 'No Data Available', ha='center', va='center', transform=ax.transAxes)
            return
        
        portfolio_df = pd.DataFrame(portfolio_values)
        portfolio_df['date'] = pd.to_datetime(portfolio_df['date'])
        portfolio_df = portfolio_df.set_index('date').sort_index()
        
        portfolio_df['cumulative_max'] = portfolio_df['portfolio_value'].expanding().max()
        portfolio_df['drawdown'] = (portfolio_df['portfolio_value'] / portfolio_df['cumulative_max'] - 1) * 100
        
        ax.fill_between(portfolio_df.index, portfolio_df['drawdown'], 0, 
                       color='red', alpha=0.3)
        ax.plot(portfolio_df.index, portfolio_df['drawdown'], linewidth=1, color='red')
        
        ax.set_title('Drawdown', fontweight='bold')
        ax.set_ylabel('Drawdown (%)')
        ax.grid(True, alpha=0.3)
    
    def _plot_sharpe_subplot(self, ax, performance_snapshots: List[PerformanceSnapshot]):
        """Plot Sharpe ratio evolution in subplot."""
        if not performance_snapshots:
            ax.text(0.5, 0.5, 'No Data Available', ha='center', va='center', transform=ax.transAxes)
            return
        
        snapshots_data = [asdict(snapshot) for snapshot in performance_snapshots]
        snapshots_df = pd.DataFrame(snapshots_data)
        snapshots_df['timestamp'] = pd.to_datetime(snapshots_df['timestamp'])
        snapshots_df = snapshots_df.set_index('timestamp').sort_index()
        
        ax.plot(snapshots_df.index, snapshots_df['sharpe_ratio'], 
               linewidth=2, color='#2E86AB')
        ax.axhline(y=1.8, color='green', linestyle='--', alpha=0.7)
        
        ax.set_title('Sharpe Ratio Evolution', fontweight='bold')
        ax.set_ylabel('Sharpe Ratio')
        ax.grid(True, alpha=0.3)
    
    def _plot_trade_distribution_subplot(self, ax, trade_logs: List[TradeLog]):
        """Plot trade distribution in subplot."""
        if not trade_logs:
            ax.text(0.5, 0.5, 'No Trades Available', ha='center', va='center', transform=ax.transAxes)
            return
        
        # Count trades by action
        actions = [trade.action for trade in trade_logs]
        action_counts = pd.Series(actions).value_counts()
        
        colors = {'buy': 'green', 'sell': 'red', 'hold': 'gray'}
        plot_colors = [colors.get(action, 'blue') for action in action_counts.index]
        
        ax.pie(action_counts.values, labels=action_counts.index, autopct='%1.1f%%',
               colors=plot_colors, startangle=90)
        ax.set_title('Trade Distribution', fontweight='bold')
    
    def _plot_monthly_returns_heatmap_subplot(self, ax, portfolio_values: List[Dict]):
        """Plot monthly returns heatmap in subplot."""
        if not portfolio_values:
            ax.text(0.5, 0.5, 'No Data Available', ha='center', va='center', transform=ax.transAxes)
            return
        
        try:
            portfolio_df = pd.DataFrame(portfolio_values)
            portfolio_df['date'] = pd.to_datetime(portfolio_df['date'])
            portfolio_df = portfolio_df.set_index('date').sort_index()
            
            # Calculate monthly returns
            monthly_values = portfolio_df['portfolio_value'].resample('M').last()
            monthly_returns = monthly_values.pct_change() * 100
            
            # Create pivot table for heatmap
            monthly_returns_df = monthly_returns.to_frame('return')
            monthly_returns_df['year'] = monthly_returns_df.index.year
            monthly_returns_df['month'] = monthly_returns_df.index.month
            
            pivot_table = monthly_returns_df.pivot(index='year', columns='month', values='return')
            
            # Plot heatmap
            sns.heatmap(pivot_table, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
                       ax=ax, cbar_kws={'label': 'Monthly Return (%)'})
            
            ax.set_title('Monthly Returns Heatmap', fontweight='bold')
            ax.set_xlabel('Month')
            ax.set_ylabel('Year')
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center', transform=ax.transAxes)
    
    def export_all_visualizations(self, portfolio_values: List[Dict],
                                performance_snapshots: List[PerformanceSnapshot],
                                trade_logs: List[TradeLog],
                                benchmark_data: Optional[pd.DataFrame] = None) -> Dict[str, str]:
        """
        Export all visualization types.
        
        Args:
            portfolio_values: Daily portfolio value records
            performance_snapshots: List of performance snapshots
            trade_logs: List of trade log entries
            benchmark_data: Benchmark data for comparison
            
        Returns:
            Dictionary mapping visualization type to file path
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            export_paths = {}
            
            # Export cumulative returns
            cumulative_path = self.plot_cumulative_returns(
                portfolio_values, benchmark_data,
                os.path.join(self.output_dir, f"cumulative_returns_{timestamp}.png")
            )
            if cumulative_path:
                export_paths['cumulative_returns'] = cumulative_path
            
            # Export drawdown analysis
            drawdown_path = self.plot_drawdown_analysis(
                portfolio_values,
                os.path.join(self.output_dir, f"drawdown_analysis_{timestamp}.png")
            )
            if drawdown_path:
                export_paths['drawdown_analysis'] = drawdown_path
            
            # Export Sharpe ratio evolution
            sharpe_path = self.plot_sharpe_ratio_evolution(
                performance_snapshots,
                os.path.join(self.output_dir, f"sharpe_evolution_{timestamp}.png")
            )
            if sharpe_path:
                export_paths['sharpe_evolution'] = sharpe_path
            
            # Export performance dashboard
            dashboard_path = self.create_performance_dashboard(
                portfolio_values, performance_snapshots, trade_logs, benchmark_data,
                os.path.join(self.output_dir, f"dashboard_{timestamp}.png")
            )
            if dashboard_path:
                export_paths['dashboard'] = dashboard_path
            
            # Export trade signals for top symbols
            symbol_counts = {}
            for trade in trade_logs:
                symbol_counts[trade.symbol] = symbol_counts.get(trade.symbol, 0) + 1
            
            top_symbols = sorted(symbol_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            
            for symbol, count in top_symbols:
                # Note: This would require price data for each symbol
                # For now, we'll skip individual symbol charts
                pass
            
            self.logger.info(f"Exported {len(export_paths)} visualizations")
            return export_paths
            
        except Exception as e:
            self.logger.error(f"Error exporting visualizations: {e}")
            return {}

class TradingVisualizer:
    """
    Visualization system for trading signals and market data.
    Provides visualizations for trading decisions, market conditions, and signal analysis.
    """
    
    def __init__(self, output_dir: str = "data/visualizations/trading"):
        """
        Initialize trading visualizer.
        
        Args:
            output_dir: Directory to save visualization outputs
        """
        self.output_dir = output_dir
        self.logger = get_logger("TradingVisualizer")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up plotting style
        self._setup_plotting_style()
        
        self.logger.info(f"Trading visualizer initialized with output dir: {output_dir}")
    
    def _setup_plotting_style(self):
        """Set up consistent plotting style."""
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Set default figure parameters
        plt.rcParams.update({
            'figure.figsize': (12, 8),
            'font.size': 10,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 16
        })
    
    def create_backtest_visualizations(self, backtest_results: Dict, performance_metrics: Dict) -> bool:
        """
        Create visualizations for backtest results.
        
        Args:
            backtest_results: Dictionary containing backtest results
            performance_metrics: Dictionary containing performance metrics
            
        Returns:
            Boolean indicating success
        """
        try:
            self.logger.info("Creating backtest visualizations...")
            
            # Extract data from backtest results
            portfolio_values = backtest_results.get('portfolio_values', [])
            trade_logs = backtest_results.get('trade_logs', [])
            benchmark_data = backtest_results.get('benchmark_data')
            
            if not portfolio_values:
                self.logger.warning("No portfolio values available for visualization")
                return False
            
            # Create performance visualizer
            perf_visualizer = PerformanceVisualizer()
            
            # Generate visualizations
            viz_paths = []
            
            # 1. Cumulative returns plot
            cum_returns_path = perf_visualizer.plot_cumulative_returns(
                portfolio_values, benchmark_data)
            if cum_returns_path:
                viz_paths.append(cum_returns_path)
            
            # 2. Drawdown analysis
            drawdown_path = perf_visualizer.plot_drawdown_analysis(portfolio_values)
            if drawdown_path:
                viz_paths.append(drawdown_path)
            
            # 3. Trade signals for top stocks
            if trade_logs:
                # Get top 5 stocks by trade count
                stock_counts = {}
                for trade in trade_logs:
                    stock_counts[trade.symbol] = stock_counts.get(trade.symbol, 0) + 1
                
                top_stocks = sorted(stock_counts.items(), key=lambda x: x[1], reverse=True)[:5]
                
                for symbol, _ in top_stocks:
                    # Get price data for this symbol
                    price_data = backtest_results.get('price_data', {}).get(symbol)
                    if price_data is not None:
                        signal_path = perf_visualizer.plot_trade_signals_with_sentiment(
                            price_data, trade_logs, symbol)
                        if signal_path:
                            viz_paths.append(signal_path)
            
            # 4. Performance dashboard
            dashboard_path = perf_visualizer.create_performance_dashboard(
                portfolio_values, [], trade_logs, benchmark_data)
            if dashboard_path:
                viz_paths.append(dashboard_path)
            
            self.logger.info(f"Created {len(viz_paths)} backtest visualizations")
            return len(viz_paths) > 0
            
        except Exception as e:
            self.logger.error(f"Error creating backtest visualizations: {e}")
            return False
    
    def plot_trading_signals(self, price_data: pd.DataFrame, signals: List[Dict], 
                           symbol: str, save_path: Optional[str] = None) -> str:
        """
        Plot trading signals on price chart.
        
        Args:
            price_data: Stock price data
            signals: List of trading signals
            symbol: Stock symbol
            save_path: Optional custom save path
            
        Returns:
            Path to saved plot
        """
        try:
            if price_data.empty or not signals:
                self.logger.warning(f"Insufficient data for trading signals plot for {symbol}")
                return ""
            
            # Create figure
            fig, ax = plt.subplots(figsize=(14, 8))
            
            # Plot price data
            ax.plot(price_data.index, price_data['Close'], linewidth=1.5, 
                   label='Close Price', color='#2E86AB')
            
            # Plot signals
            buy_signals = [s for s in signals if s['action'] == 'buy']
            sell_signals = [s for s in signals if s['action'] == 'sell']
            
            if buy_signals:
                buy_dates = [s['timestamp'] for s in buy_signals]
                buy_prices = [price_data.loc[s['timestamp']]['Close'] for s in buy_signals 
                            if s['timestamp'] in price_data.index]
                buy_dates = [d for i, d in enumerate(buy_dates) if d in price_data.index]
                
                if buy_dates:
                    ax.scatter(buy_dates, buy_prices, marker='^', s=100, 
                              color='green', label='Buy Signals', alpha=0.8)
            
            if sell_signals:
                sell_dates = [s['timestamp'] for s in sell_signals]
                sell_prices = [price_data.loc[s['timestamp']]['Close'] for s in sell_signals 
                             if s['timestamp'] in price_data.index]
                sell_dates = [d for i, d in enumerate(sell_dates) if d in price_data.index]
                
                if sell_dates:
                    ax.scatter(sell_dates, sell_prices, marker='v', s=100, 
                              color='red', label='Sell Signals', alpha=0.8)
            
            # Formatting
            ax.set_title(f'{symbol} - Trading Signals', fontsize=16, fontweight='bold')
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Price', fontsize=12)
            ax.legend(loc='upper left')
            ax.grid(True, alpha=0.3)
            
            # Format x-axis
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            
            # Save plot
            if save_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                save_path = os.path.join(self.output_dir, f"trading_signals_{symbol}_{timestamp}.png")
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Trading signals plot for {symbol} saved to: {save_path}")
            return save_path
            
        except Exception as e:
            self.logger.error(f"Error creating trading signals plot for {symbol}: {e}")
            return ""
    
    def plot_model_predictions(self, actual_prices: pd.Series, predicted_prices: pd.Series,
                              symbol: str, save_path: Optional[str] = None) -> str:
        """
        Plot model predictions against actual prices.
        
        Args:
            actual_prices: Series of actual prices
            predicted_prices: Series of predicted prices
            symbol: Stock symbol
            save_path: Optional custom save path
            
        Returns:
            Path to saved plot
        """
        try:
            if actual_prices.empty or predicted_prices.empty:
                self.logger.warning(f"Insufficient data for model predictions plot for {symbol}")
                return ""
            
            # Create figure
            fig, ax = plt.subplots(figsize=(14, 8))
            
            # Plot actual and predicted prices
            ax.plot(actual_prices.index, actual_prices, linewidth=1.5, 
                   label='Actual Price', color='#2E86AB')
            ax.plot(predicted_prices.index, predicted_prices, linewidth=1.5, 
                   label='Predicted Price', color='#A23B72', linestyle='--')
            
            # Calculate error metrics
            mse = ((actual_prices - predicted_prices) ** 2).mean()
            rmse = np.sqrt(mse)
            mae = (actual_prices - predicted_prices).abs().mean()
            
            # Add error metrics to plot
            metrics_text = f'RMSE: {rmse:.4f}\nMAE: {mae:.4f}'
            ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            # Formatting
            ax.set_title(f'{symbol} - Model Predictions vs Actual Prices', fontsize=16, fontweight='bold')
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Price', fontsize=12)
            ax.legend(loc='upper left')
            ax.grid(True, alpha=0.3)
            
            # Format x-axis
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            
            # Save plot
            if save_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                save_path = os.path.join(self.output_dir, f"model_predictions_{symbol}_{timestamp}.png")
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Model predictions plot for {symbol} saved to: {save_path}")
            return save_path
            
        except Exception as e:
            self.logger.error(f"Error creating model predictions plot for {symbol}: {e}")
            return ""

class TradingVisualizer:
    """
    Visualization system for trading signals and market data.
    Provides visualizations for trading decisions, market conditions, and signal analysis.
    """
    
    def __init__(self, output_dir: str = "data/visualizations/trading"):
        """
        Initialize trading visualizer.
        
        Args:
            output_dir: Directory to save visualization outputs
        """
        self.output_dir = output_dir
        self.logger = get_logger("TradingVisualizer")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up plotting style
        self._setup_plotting_style()
        
        self.logger.info(f"Trading visualizer initialized with output dir: {output_dir}")
    
    def _setup_plotting_style(self):
        """Set up consistent plotting style."""
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Set default figure parameters
        plt.rcParams.update({
            'figure.figsize': (12, 8),
            'font.size': 10,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 16
        })
    
    def create_backtest_visualizations(self, backtest_results: Dict, performance_metrics: Dict) -> bool:
        """
        Create visualizations for backtest results.
        
        Args:
            backtest_results: Dictionary containing backtest results
            performance_metrics: Dictionary containing performance metrics
            
        Returns:
            Boolean indicating success
        """
        try:
            self.logger.info("Creating backtest visualizations...")
            
            # Extract data from backtest results
            portfolio_values = backtest_results.get('portfolio_values', [])
            trade_logs = backtest_results.get('trade_logs', [])
            benchmark_data = backtest_results.get('benchmark_data')
            
            if not portfolio_values:
                self.logger.warning("No portfolio values available for visualization")
                return False
            
            # Create performance visualizer
            perf_visualizer = PerformanceVisualizer()
            
            # Generate visualizations
            viz_paths = []
            
            # 1. Cumulative returns plot
            cum_returns_path = perf_visualizer.plot_cumulative_returns(
                portfolio_values, benchmark_data)
            if cum_returns_path:
                viz_paths.append(cum_returns_path)
            
            # 2. Drawdown analysis
            drawdown_path = perf_visualizer.plot_drawdown_analysis(portfolio_values)
            if drawdown_path:
                viz_paths.append(drawdown_path)
            
            # 3. Trade signals for top stocks
            if trade_logs:
                # Get top 5 stocks by trade count
                stock_counts = {}
                for trade in trade_logs:
                    stock_counts[trade.symbol] = stock_counts.get(trade.symbol, 0) + 1
                
                top_stocks = sorted(stock_counts.items(), key=lambda x: x[1], reverse=True)[:5]
                
                for symbol, _ in top_stocks:
                    # Get price data for this symbol
                    price_data = backtest_results.get('price_data', {}).get(symbol)
                    if price_data is not None:
                        signal_path = perf_visualizer.plot_trade_signals_with_sentiment(
                            price_data, trade_logs, symbol)
                        if signal_path:
                            viz_paths.append(signal_path)
            
            # 4. Performance dashboard
            dashboard_path = perf_visualizer.create_performance_dashboard(
                portfolio_values, [], trade_logs, benchmark_data)
            if dashboard_path:
                viz_paths.append(dashboard_path)
            
            self.logger.info(f"Created {len(viz_paths)} backtest visualizations")
            return len(viz_paths) > 0
            
        except Exception as e:
            self.logger.error(f"Error creating backtest visualizations: {e}")
            return False
