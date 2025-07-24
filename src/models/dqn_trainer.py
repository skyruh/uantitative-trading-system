"""
DQN Training System for the quantitative trading system.
Implements comprehensive training pipeline with state representation, reward calculation,
and integration with LSTM predictions and sentiment analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

from src.models.dqn_agent import DQNAgent
from src.models.lstm_model import LSTMModel
from src.interfaces.model_interfaces import IDQNAgent, ILSTMModel


@dataclass
class TradingState:
    """Represents the state of the trading environment."""
    lstm_prediction: float
    sentiment_score: float
    price_features: np.ndarray  # RSI, SMA, Bollinger Bands, etc.
    portfolio_features: np.ndarray  # Current position, cash, portfolio value
    market_features: np.ndarray  # Volume, volatility, etc.


@dataclass
class TradingAction:
    """Represents a trading action with metadata."""
    action: str  # 'buy', 'sell', 'hold'
    confidence: float
    timestamp: datetime
    symbol: str


@dataclass
class TradingReward:
    """Represents reward calculation details."""
    portfolio_return: float
    benchmark_return: float
    sharpe_ratio: float
    reward_value: float
    calculation_details: Dict[str, float]


class PortfolioSimulator:
    """Simulates portfolio for DQN training."""
    
    def __init__(self, initial_capital: float = 1000000.0, transaction_cost: float = 0.001):
        """
        Initialize portfolio simulator.
        
        Args:
            initial_capital: Initial capital in rupees
            transaction_cost: Transaction cost as fraction (0.1% = 0.001)
        """
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.reset()
        self.logger = logging.getLogger(__name__)
    
    def reset(self) -> None:
        """Reset portfolio to initial state."""
        self.cash = self.initial_capital
        self.positions = {}  # symbol -> quantity
        self.portfolio_value = self.initial_capital
        self.trade_history = []
        self.returns_history = []
        self.benchmark_returns = []
    
    def execute_trade(self, symbol: str, action: str, price: float, 
                     quantity: Optional[int] = None) -> bool:
        """
        Execute a trade in the portfolio.
        
        Args:
            symbol: Stock symbol
            action: 'buy', 'sell', 'hold'
            price: Current stock price
            quantity: Number of shares (auto-calculated if None)
            
        Returns:
            True if trade executed successfully
        """
        try:
            if action == 'hold':
                return True
            
            # Calculate position size (1-2% of portfolio value)
            if quantity is None:
                position_size = self.portfolio_value * 0.015  # 1.5% of portfolio
                quantity = int(position_size / price)
            
            if quantity <= 0:
                return False
            
            trade_value = quantity * price
            cost = trade_value * self.transaction_cost
            
            if action == 'buy':
                total_cost = trade_value + cost
                if self.cash >= total_cost:
                    self.cash -= total_cost
                    self.positions[symbol] = self.positions.get(symbol, 0) + quantity
                    
                    self.trade_history.append({
                        'symbol': symbol,
                        'action': action,
                        'quantity': quantity,
                        'price': price,
                        'cost': cost,
                        'timestamp': datetime.now()
                    })
                    return True
                    
            elif action == 'sell':
                current_position = self.positions.get(symbol, 0)
                sell_quantity = min(quantity, current_position)
                
                if sell_quantity > 0:
                    proceeds = sell_quantity * price - cost
                    self.cash += proceeds
                    self.positions[symbol] -= sell_quantity
                    
                    if self.positions[symbol] == 0:
                        del self.positions[symbol]
                    
                    self.trade_history.append({
                        'symbol': symbol,
                        'action': action,
                        'quantity': sell_quantity,
                        'price': price,
                        'cost': cost,
                        'timestamp': datetime.now()
                    })
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error executing trade: {str(e)}")
            return False
    
    def update_portfolio_value(self, prices: Dict[str, float]) -> float:
        """
        Update portfolio value based on current prices.
        
        Args:
            prices: Dictionary of symbol -> current price
            
        Returns:
            Current portfolio value
        """
        try:
            position_value = 0.0
            for symbol, quantity in self.positions.items():
                if symbol in prices:
                    position_value += quantity * prices[symbol]
            
            self.portfolio_value = self.cash + position_value
            return self.portfolio_value
            
        except Exception as e:
            self.logger.error(f"Error updating portfolio value: {str(e)}")
            return self.portfolio_value
    
    def calculate_return(self) -> float:
        """Calculate portfolio return since last calculation."""
        try:
            if not self.returns_history:
                # First calculation
                current_return = (self.portfolio_value - self.initial_capital) / self.initial_capital
            else:
                # Calculate return since last update
                last_value = self.returns_history[-1]['portfolio_value']
                current_return = (self.portfolio_value - last_value) / last_value
            
            self.returns_history.append({
                'portfolio_value': self.portfolio_value,
                'return': current_return,
                'timestamp': datetime.now()
            })
            
            return current_return
            
        except Exception as e:
            self.logger.error(f"Error calculating return: {str(e)}")
            return 0.0
    
    def get_portfolio_stats(self) -> Dict[str, float]:
        """Get portfolio statistics."""
        try:
            if not self.returns_history:
                return {}
            
            returns = [r['return'] for r in self.returns_history]
            
            total_return = (self.portfolio_value - self.initial_capital) / self.initial_capital
            avg_return = np.mean(returns) if returns else 0.0
            volatility = np.std(returns) if len(returns) > 1 else 0.0
            sharpe_ratio = avg_return / volatility if volatility > 0 else 0.0
            
            return {
                'total_return': total_return,
                'average_return': avg_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'num_trades': len(self.trade_history),
                'cash': self.cash,
                'portfolio_value': self.portfolio_value
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio stats: {str(e)}")
            return {}


class StateBuilder:
    """Builds state representation for DQN training."""
    
    def __init__(self, lstm_model: Optional[ILSTMModel] = None):
        """
        Initialize state builder.
        
        Args:
            lstm_model: Trained LSTM model for predictions
        """
        self.lstm_model = lstm_model
        self.logger = logging.getLogger(__name__)
    
    def build_state(self, data: pd.DataFrame, portfolio: PortfolioSimulator,
                   symbol: str, current_index: int) -> np.ndarray:
        """
        Build state vector for DQN.
        
        Args:
            data: Market data DataFrame
            portfolio: Portfolio simulator
            symbol: Current stock symbol
            current_index: Current data index
            
        Returns:
            State vector as numpy array
        """
        try:
            # Get current row data
            current_data = data.iloc[current_index]
            
            # LSTM prediction (if available)
            lstm_pred = 0.5  # Default neutral prediction
            if self.lstm_model and self.lstm_model.is_trained:
                try:
                    # Get sequence data for LSTM
                    sequence_start = max(0, current_index - 60)  # 60-day sequence
                    sequence_data = data.iloc[sequence_start:current_index + 1]
                    
                    if len(sequence_data) >= 60:
                        lstm_pred = self.lstm_model.predict_price_movement(
                            self._prepare_lstm_sequence(sequence_data)
                        )
                except Exception as e:
                    self.logger.warning(f"LSTM prediction failed: {str(e)}")
            
            # Sentiment score
            sentiment = current_data.get('sentiment_score', 0.0)
            
            # Technical indicators
            rsi = current_data.get('rsi_14', 50.0) / 100.0  # Normalize to 0-1
            sma_ratio = current_data.get('close', 100.0) / current_data.get('sma_50', 100.0)
            bb_position = self._calculate_bb_position(current_data)
            
            # Price features
            price_change = self._calculate_price_change(data, current_index)
            volume_ratio = self._calculate_volume_ratio(data, current_index)
            
            # Portfolio features
            portfolio_stats = portfolio.get_portfolio_stats()
            cash_ratio = portfolio.cash / portfolio.portfolio_value if portfolio.portfolio_value > 0 else 1.0
            current_position = portfolio.positions.get(symbol, 0)
            position_ratio = (current_position * current_data.get('close', 0)) / portfolio.portfolio_value if portfolio.portfolio_value > 0 else 0.0
            
            # Build state vector
            state = np.array([
                lstm_pred,           # LSTM prediction (0-1)
                sentiment,           # Sentiment score (-1 to 1)
                rsi,                 # RSI (0-1)
                sma_ratio,           # Price/SMA ratio
                bb_position,         # Bollinger Band position
                price_change,        # Recent price change
                volume_ratio,        # Volume ratio
                cash_ratio,          # Cash ratio (0-1)
                position_ratio,      # Position ratio (0-1)
                portfolio_stats.get('sharpe_ratio', 0.0)  # Portfolio Sharpe ratio
            ])
            
            return state
            
        except Exception as e:
            self.logger.error(f"Error building state: {str(e)}")
            # Return default state
            return np.zeros(10)
    
    def _prepare_lstm_sequence(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare sequence data for LSTM prediction."""
        try:
            # Use same features as LSTM training
            feature_columns = ['close', 'rsi_14', 'sma_50', 'bb_upper', 'bb_middle', 
                             'bb_lower', 'sentiment_score', 'volume', 'high', 'low']
            
            # Fill missing columns with defaults
            for col in feature_columns:
                if col not in data.columns:
                    if col == 'sentiment_score':
                        data[col] = 0.0
                    elif col in ['rsi_14']:
                        data[col] = 50.0
                    else:
                        data[col] = data.get('close', 100.0)
            
            # Get last 60 rows and required columns
            sequence_data = data[feature_columns].tail(60).values
            
            # Ensure we have the right shape
            if sequence_data.shape[0] < 60:
                # Pad with first row if insufficient data
                padding = np.tile(sequence_data[0], (60 - sequence_data.shape[0], 1))
                sequence_data = np.vstack([padding, sequence_data])
            
            return sequence_data
            
        except Exception as e:
            self.logger.error(f"Error preparing LSTM sequence: {str(e)}")
            return np.zeros((60, 10))
    
    def _calculate_bb_position(self, data: pd.Series) -> float:
        """Calculate position within Bollinger Bands."""
        try:
            close = data.get('close', 100.0)
            bb_upper = data.get('bb_upper', close)
            bb_lower = data.get('bb_lower', close)
            
            if bb_upper == bb_lower:
                return 0.5
            
            position = (close - bb_lower) / (bb_upper - bb_lower)
            return np.clip(position, 0.0, 1.0)
            
        except Exception:
            return 0.5
    
    def _calculate_price_change(self, data: pd.DataFrame, current_index: int) -> float:
        """Calculate recent price change."""
        try:
            if current_index < 5:
                return 0.0
            
            current_price = data.iloc[current_index]['close']
            past_price = data.iloc[current_index - 5]['close']
            
            change = (current_price - past_price) / past_price
            return np.tanh(change * 10)  # Normalize to [-1, 1]
            
        except Exception:
            return 0.0
    
    def _calculate_volume_ratio(self, data: pd.DataFrame, current_index: int) -> float:
        """Calculate volume ratio compared to recent average."""
        try:
            if current_index < 10:
                return 1.0
            
            current_volume = data.iloc[current_index]['volume']
            avg_volume = data.iloc[current_index - 10:current_index]['volume'].mean()
            
            if avg_volume == 0:
                return 1.0
            
            ratio = current_volume / avg_volume
            return np.tanh(ratio - 1)  # Normalize around 0
            
        except Exception:
            return 0.0


class DQNTrainer:
    """
    Comprehensive DQN training system with Sharpe ratio-based rewards,
    LSTM integration, and sentiment analysis.
    """
    
    def __init__(self, dqn_agent: IDQNAgent, lstm_model: Optional[ILSTMModel] = None,
                 initial_capital: float = 1000000.0, transaction_cost: float = 0.001):
        """
        Initialize DQN trainer.
        
        Args:
            dqn_agent: DQN agent to train
            lstm_model: Optional LSTM model for predictions
            initial_capital: Initial capital for simulation
            transaction_cost: Transaction cost fraction
        """
        self.dqn_agent = dqn_agent
        self.lstm_model = lstm_model
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        
        self.portfolio = PortfolioSimulator(initial_capital, transaction_cost)
        self.state_builder = StateBuilder(lstm_model)
        
        self.training_history = []
        self.episode_rewards = []
        self.benchmark_returns = []
        
        self.logger = logging.getLogger(__name__)
    
    def calculate_sharpe_reward(self, portfolio_return: float, benchmark_return: float,
                              portfolio_volatility: float = None) -> TradingReward:
        """
        Calculate reward based on Sharpe ratio improvement.
        
        Args:
            portfolio_return: Portfolio return for the period
            benchmark_return: Benchmark return for the period
            portfolio_volatility: Portfolio volatility (optional)
            
        Returns:
            TradingReward object with calculation details
        """
        try:
            # Calculate excess return
            excess_return = portfolio_return - benchmark_return
            
            # Get portfolio statistics
            portfolio_stats = self.portfolio.get_portfolio_stats()
            portfolio_sharpe = portfolio_stats.get('sharpe_ratio', 0.0)
            
            # Base reward on excess return
            base_reward = np.tanh(excess_return * 10)  # Scale to [-1, 1]
            
            # Bonus for positive Sharpe ratio
            sharpe_bonus = 0.0
            if portfolio_sharpe > 0:
                sharpe_bonus = min(portfolio_sharpe * 0.1, 0.5)  # Max 0.5 bonus
            
            # Penalty for high volatility
            volatility_penalty = 0.0
            if portfolio_volatility and portfolio_volatility > 0.02:  # 2% daily volatility
                volatility_penalty = -min((portfolio_volatility - 0.02) * 5, 0.3)
            
            # Final reward
            final_reward = base_reward + sharpe_bonus + volatility_penalty
            final_reward = np.clip(final_reward, -1.0, 1.0)
            
            return TradingReward(
                portfolio_return=portfolio_return,
                benchmark_return=benchmark_return,
                sharpe_ratio=portfolio_sharpe,
                reward_value=final_reward,
                calculation_details={
                    'excess_return': excess_return,
                    'base_reward': base_reward,
                    'sharpe_bonus': sharpe_bonus,
                    'volatility_penalty': volatility_penalty,
                    'portfolio_volatility': portfolio_volatility or 0.0
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating Sharpe reward: {str(e)}")
            return TradingReward(
                portfolio_return=portfolio_return,
                benchmark_return=benchmark_return,
                sharpe_ratio=0.0,
                reward_value=0.0,
                calculation_details={}
            )
    
    def train_episode(self, data: pd.DataFrame, symbol: str, 
                     benchmark_data: pd.DataFrame = None) -> Dict[str, float]:
        """
        Train DQN for one episode using provided data.
        
        Args:
            data: Stock data for training
            symbol: Stock symbol
            benchmark_data: Benchmark data for comparison
            
        Returns:
            Episode training metrics
        """
        try:
            self.portfolio.reset()
            episode_rewards = []
            episode_actions = []
            
            # Minimum data required for meaningful training
            min_data_points = 100
            if len(data) < min_data_points:
                self.logger.warning(f"Insufficient data for episode: {len(data)} < {min_data_points}")
                return {}
            
            # Training loop
            for i in range(60, len(data) - 1):  # Start after sequence length
                try:
                    # Build current state
                    current_state = self.state_builder.build_state(data, self.portfolio, symbol, i)
                    
                    # Select action
                    action = self.dqn_agent.select_action(current_state)
                    episode_actions.append(action)
                    
                    # Execute trade
                    current_price = data.iloc[i]['close']
                    trade_success = self.portfolio.execute_trade(symbol, action, current_price)
                    
                    # Update portfolio value
                    next_price = data.iloc[i + 1]['close']
                    self.portfolio.update_portfolio_value({symbol: next_price})
                    
                    # Calculate returns
                    portfolio_return = self.portfolio.calculate_return()
                    
                    # Calculate benchmark return
                    benchmark_return = 0.0
                    if benchmark_data is not None and i < len(benchmark_data) - 1:
                        benchmark_return = (benchmark_data.iloc[i + 1]['close'] - 
                                         benchmark_data.iloc[i]['close']) / benchmark_data.iloc[i]['close']
                    
                    # Calculate reward
                    reward_info = self.calculate_sharpe_reward(portfolio_return, benchmark_return)
                    episode_rewards.append(reward_info.reward_value)
                    
                    # Build next state
                    next_state = self.state_builder.build_state(data, self.portfolio, symbol, i + 1)
                    
                    # Update experience
                    done = (i == len(data) - 2)  # Last step
                    self.dqn_agent.update_experience(
                        current_state, action, reward_info.reward_value, next_state, done
                    )
                    
                    # Perform training step
                    if i % 10 == 0:  # Train every 10 steps
                        loss = self.dqn_agent.train_step()
                        if loss is not None:
                            self.logger.debug(f"Training step {i}: loss={loss:.4f}")
                    
                except Exception as e:
                    self.logger.error(f"Error in training step {i}: {str(e)}")
                    continue
            
            # Episode statistics
            episode_stats = {
                'total_reward': sum(episode_rewards),
                'average_reward': np.mean(episode_rewards) if episode_rewards else 0.0,
                'num_actions': len(episode_actions),
                'buy_actions': episode_actions.count('buy'),
                'sell_actions': episode_actions.count('sell'),
                'hold_actions': episode_actions.count('hold'),
                'final_portfolio_value': self.portfolio.portfolio_value,
                'total_return': (self.portfolio.portfolio_value - self.initial_capital) / self.initial_capital
            }
            
            # Add portfolio statistics
            portfolio_stats = self.portfolio.get_portfolio_stats()
            episode_stats.update(portfolio_stats)
            
            self.episode_rewards.append(episode_stats)
            
            self.logger.info(f"Episode completed: reward={episode_stats['total_reward']:.4f}, "
                           f"return={episode_stats['total_return']:.4f}")
            
            return episode_stats
            
        except Exception as e:
            self.logger.error(f"Error in training episode: {str(e)}")
            return {}
    
    def train(self, training_data: Dict[str, pd.DataFrame], 
              benchmark_data: pd.DataFrame = None, episodes: int = 100) -> bool:
        """
        Train DQN agent on multiple stocks.
        
        Args:
            training_data: Dictionary of symbol -> DataFrame
            benchmark_data: Benchmark data for comparison
            episodes: Number of training episodes
            
        Returns:
            True if training successful
        """
        try:
            self.logger.info(f"Starting DQN training for {episodes} episodes on {len(training_data)} stocks")
            
            symbols = list(training_data.keys())
            
            for episode in range(episodes):
                episode_metrics = []
                
                # Train on each stock
                for symbol in symbols:
                    stock_data = training_data[symbol]
                    
                    if len(stock_data) < 100:  # Skip stocks with insufficient data
                        continue
                    
                    episode_stats = self.train_episode(stock_data, symbol, benchmark_data)
                    if episode_stats:
                        episode_metrics.append(episode_stats)
                
                # Log episode summary
                if episode_metrics:
                    avg_reward = np.mean([m['total_reward'] for m in episode_metrics])
                    avg_return = np.mean([m['total_return'] for m in episode_metrics])
                    
                    self.training_history.append({
                        'episode': episode,
                        'average_reward': avg_reward,
                        'average_return': avg_return,
                        'num_stocks': len(episode_metrics),
                        'epsilon': self.dqn_agent.epsilon
                    })
                    
                    if episode % 10 == 0:
                        self.logger.info(f"Episode {episode}: avg_reward={avg_reward:.4f}, "
                                       f"avg_return={avg_return:.4f}, epsilon={self.dqn_agent.epsilon:.4f}")
            
            self.logger.info("DQN training completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error in DQN training: {str(e)}")
            return False
    
    def get_training_metrics(self) -> Dict[str, Any]:
        """Get comprehensive training metrics."""
        try:
            if not self.training_history:
                return {}
            
            rewards = [h['average_reward'] for h in self.training_history]
            returns = [h['average_return'] for h in self.training_history]
            
            return {
                'total_episodes': len(self.training_history),
                'final_average_reward': rewards[-1] if rewards else 0.0,
                'best_average_reward': max(rewards) if rewards else 0.0,
                'final_average_return': returns[-1] if returns else 0.0,
                'best_average_return': max(returns) if returns else 0.0,
                'reward_improvement': rewards[-1] - rewards[0] if len(rewards) > 1 else 0.0,
                'return_improvement': returns[-1] - returns[0] if len(returns) > 1 else 0.0,
                'training_stability': np.std(rewards[-10:]) if len(rewards) >= 10 else 0.0,
                'agent_metrics': self.dqn_agent.get_training_metrics()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting training metrics: {str(e)}")
            return {}