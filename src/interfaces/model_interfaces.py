"""
Model interfaces for the quantitative trading system.
Defines contracts for LSTM and DQN models.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd


class IModel(ABC):
    """Base interface for all models."""
    
    @abstractmethod
    def train(self, data: pd.DataFrame) -> bool:
        """Train the model with provided data."""
        pass
    
    @abstractmethod
    def predict(self, data: pd.DataFrame) -> Any:
        """Make predictions using the trained model."""
        pass
    
    @abstractmethod
    def save_model(self, path: str) -> bool:
        """Save the trained model to disk."""
        pass
    
    @abstractmethod
    def load_model(self, path: str) -> bool:
        """Load a trained model from disk."""
        pass


class ILSTMModel(IModel):
    """Interface for LSTM price prediction model."""
    
    @abstractmethod
    def predict_price_movement(self, sequence_data: np.ndarray) -> float:
        """Predict next-day price movement probability."""
        pass
    
    @abstractmethod
    def get_model_confidence(self) -> float:
        """Get model prediction confidence score."""
        pass


class IDQNAgent(IModel):
    """Interface for Deep Q-Network trading agent."""
    
    @abstractmethod
    def select_action(self, state: np.ndarray) -> str:
        """Select trading action (buy, sell, hold) based on current state."""
        pass
    
    @abstractmethod
    def get_q_values(self, state: np.ndarray) -> Dict[str, float]:
        """Get Q-values for all possible actions."""
        pass
    
    @abstractmethod
    def update_experience(self, state: np.ndarray, action: str, reward: float, 
                         next_state: np.ndarray, done: bool) -> None:
        """Update experience replay buffer."""
        pass
    
    @abstractmethod
    def calculate_reward(self, portfolio_return: float, benchmark_return: float) -> float:
        """Calculate reward based on Sharpe ratio improvement."""
        pass


class IModelEvaluator(ABC):
    """Interface for model evaluation."""
    
    @abstractmethod
    def evaluate_lstm_performance(self, model: ILSTMModel, test_data: pd.DataFrame) -> Dict[str, float]:
        """Evaluate LSTM model performance metrics."""
        pass
    
    @abstractmethod
    def evaluate_dqn_performance(self, agent: IDQNAgent, test_data: pd.DataFrame) -> Dict[str, float]:
        """Evaluate DQN agent performance metrics."""
        pass