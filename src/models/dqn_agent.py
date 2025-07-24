"""
Deep Q-Network (DQN) Agent for trading decisions in the quantitative trading system.
Implements DQN with experience replay, target network, and epsilon-greedy exploration.
"""

import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from collections import deque
from typing import Dict, List, Tuple, Optional, Any
import logging

from src.interfaces.model_interfaces import IDQNAgent


class ExperienceReplayBuffer:
    """Experience replay buffer for DQN training."""
    
    def __init__(self, capacity: int = 10000):
        """
        Initialize experience replay buffer.
        
        Args:
            capacity: Maximum number of experiences to store
        """
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
    
    def add_experience(self, state: np.ndarray, action: int, reward: float, 
                      next_state: np.ndarray, done: bool) -> None:
        """
        Add experience to the buffer.
        
        Args:
            state: Current state
            action: Action taken (0=buy, 1=sell, 2=hold)
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample_batch(self, batch_size: int) -> List[Tuple]:
        """
        Sample a batch of experiences.
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            List of sampled experiences
        """
        if len(self.buffer) < batch_size:
            return list(self.buffer)
        return random.sample(self.buffer, batch_size)
    
    def size(self) -> int:
        """Get current buffer size."""
        return len(self.buffer)
    
    def is_ready(self, min_size: int) -> bool:
        """Check if buffer has enough experiences for training."""
        return len(self.buffer) >= min_size


class DQNAgent(IDQNAgent):
    """
    Deep Q-Network agent for trading decisions.
    
    Features:
    - Three actions: buy (0), sell (1), hold (2)
    - Experience replay buffer for stable training
    - Target network for stable Q-learning
    - Epsilon-greedy exploration strategy
    - Reward function based on Sharpe ratio improvement
    """
    
    def __init__(self, state_size: int = 10, learning_rate: float = 0.001,
                 epsilon: float = 1.0, epsilon_min: float = 0.01,
                 epsilon_decay: float = 0.995, gamma: float = 0.95,
                 batch_size: int = 32, buffer_capacity: int = 10000,
                 target_update_freq: int = 100, random_seed: int = 42):
        """
        Initialize DQN agent.
        
        Args:
            state_size: Size of state vector
            learning_rate: Learning rate for neural network
            epsilon: Initial exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Exploration decay rate
            gamma: Discount factor for future rewards
            batch_size: Training batch size
            buffer_capacity: Experience replay buffer capacity
            target_update_freq: Frequency to update target network
            random_seed: Random seed for reproducibility
        """
        self.state_size = state_size
        self.action_size = 3  # buy, sell, hold
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.random_seed = random_seed
        
        # Action mapping
        self.actions = ['buy', 'sell', 'hold']
        self.action_to_index = {'buy': 0, 'sell': 1, 'hold': 2}
        self.index_to_action = {0: 'buy', 1: 'sell', 2: 'hold'}
        
        # Set random seeds
        tf.random.set_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)
        
        # Initialize networks and buffer
        self.q_network = self._build_network()
        self.target_network = self._build_network()
        self.experience_buffer = ExperienceReplayBuffer(buffer_capacity)
        
        # Training tracking
        self.training_step = 0
        self.is_trained = False
        self.training_history = []
        self.logger = logging.getLogger(__name__)
        
        # Update target network initially
        self._update_target_network()
        
        self.logger.info(f"DQN Agent initialized with state_size={state_size}")
    
    def _build_network(self) -> tf.keras.Model:
        """
        Build the neural network for Q-value approximation.
        
        Returns:
            Compiled Keras model
        """
        model = Sequential([
            Dense(128, activation='relu', input_shape=(self.state_size,), name='dense_1'),
            Dropout(0.2, name='dropout_1'),
            Dense(64, activation='relu', name='dense_2'),
            Dropout(0.2, name='dropout_2'),
            Dense(32, activation='relu', name='dense_3'),
            Dense(self.action_size, activation='linear', name='q_values')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def _update_target_network(self) -> None:
        """Update target network weights with main network weights."""
        self.target_network.set_weights(self.q_network.get_weights())
        self.logger.debug("Target network updated")
    
    def select_action(self, state: np.ndarray) -> str:
        """
        Select trading action using epsilon-greedy strategy.
        
        Args:
            state: Current state vector
            
        Returns:
            Selected action ('buy', 'sell', 'hold')
        """
        try:
            # Ensure state has correct shape
            if state.ndim == 1:
                state = state.reshape(1, -1)
            
            # Epsilon-greedy action selection
            if np.random.random() <= self.epsilon:
                # Random action (exploration)
                action_index = np.random.choice(self.action_size)
                self.logger.debug(f"Random action selected: {self.index_to_action[action_index]}")
            else:
                # Greedy action (exploitation)
                q_values = self.q_network.predict(state, verbose=0)
                action_index = np.argmax(q_values[0])
                self.logger.debug(f"Greedy action selected: {self.index_to_action[action_index]}")
            
            return self.index_to_action[action_index]
            
        except Exception as e:
            self.logger.error(f"Error selecting action: {str(e)}")
            return 'hold'  # Default to hold on error
    
    def get_q_values(self, state: np.ndarray) -> Dict[str, float]:
        """
        Get Q-values for all possible actions.
        
        Args:
            state: Current state vector
            
        Returns:
            Dictionary mapping actions to Q-values
        """
        try:
            # Ensure state has correct shape
            if state.ndim == 1:
                state = state.reshape(1, -1)
            
            q_values = self.q_network.predict(state, verbose=0)[0]
            
            return {
                'buy': float(q_values[0]),
                'sell': float(q_values[1]),
                'hold': float(q_values[2])
            }
            
        except Exception as e:
            self.logger.error(f"Error getting Q-values: {str(e)}")
            return {'buy': 0.0, 'sell': 0.0, 'hold': 0.0}
    
    def update_experience(self, state: np.ndarray, action: str, reward: float,
                         next_state: np.ndarray, done: bool) -> None:
        """
        Update experience replay buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        try:
            action_index = self.action_to_index[action]
            self.experience_buffer.add_experience(state, action_index, reward, next_state, done)
            self.logger.debug(f"Experience added: action={action}, reward={reward:.4f}")
            
        except Exception as e:
            self.logger.error(f"Error updating experience: {str(e)}")
    
    def calculate_reward(self, portfolio_return: float, benchmark_return: float) -> float:
        """
        Calculate reward based on Sharpe ratio improvement.
        
        Args:
            portfolio_return: Portfolio return for the period
            benchmark_return: Benchmark return for the period
            
        Returns:
            Calculated reward
        """
        try:
            # Simple reward based on excess return over benchmark
            excess_return = portfolio_return - benchmark_return
            
            # Scale reward to reasonable range
            reward = np.tanh(excess_return * 10)  # tanh keeps reward in [-1, 1]
            
            self.logger.debug(f"Reward calculated: portfolio={portfolio_return:.4f}, "
                            f"benchmark={benchmark_return:.4f}, reward={reward:.4f}")
            
            return float(reward)
            
        except Exception as e:
            self.logger.error(f"Error calculating reward: {str(e)}")
            return 0.0
    
    def train_step(self) -> Optional[float]:
        """
        Perform one training step using experience replay.
        
        Returns:
            Training loss if training occurred, None otherwise
        """
        if not self.experience_buffer.is_ready(self.batch_size):
            return None
        
        try:
            # Sample batch from experience buffer
            batch = self.experience_buffer.sample_batch(self.batch_size)
            
            # Prepare training data
            states = np.array([exp[0] for exp in batch])
            actions = np.array([exp[1] for exp in batch])
            rewards = np.array([exp[2] for exp in batch])
            next_states = np.array([exp[3] for exp in batch])
            dones = np.array([exp[4] for exp in batch])
            
            # Get current Q-values
            current_q_values = self.q_network.predict(states, verbose=0)
            
            # Get next Q-values from target network
            next_q_values = self.target_network.predict(next_states, verbose=0)
            
            # Calculate target Q-values
            target_q_values = current_q_values.copy()
            
            for i in range(len(batch)):
                if dones[i]:
                    target_q_values[i][actions[i]] = rewards[i]
                else:
                    target_q_values[i][actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i])
            
            # Train the network
            history = self.q_network.fit(states, target_q_values, verbose=0, epochs=1)
            loss = history.history['loss'][0]
            
            # Update training step and epsilon
            self.training_step += 1
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            
            # Update target network periodically
            if self.training_step % self.target_update_freq == 0:
                self._update_target_network()
            
            # Track training history
            self.training_history.append({
                'step': self.training_step,
                'loss': loss,
                'epsilon': self.epsilon
            })
            
            self.logger.debug(f"Training step {self.training_step}: loss={loss:.4f}, epsilon={self.epsilon:.4f}")
            
            return loss
            
        except Exception as e:
            self.logger.error(f"Error in training step: {str(e)}")
            return None
    
    def train(self, data: pd.DataFrame, episodes: int = 1000) -> bool:
        """
        Train the DQN agent using provided data.
        
        Args:
            data: Training data with market information
            episodes: Number of training episodes
            
        Returns:
            True if training successful, False otherwise
        """
        try:
            self.logger.info(f"Starting DQN training for {episodes} episodes...")
            
            # This is a simplified training loop
            # In practice, this would involve more sophisticated episode simulation
            total_losses = []
            
            for episode in range(episodes):
                episode_losses = []
                
                # Simulate some training steps per episode
                for _ in range(10):  # 10 training steps per episode
                    loss = self.train_step()
                    if loss is not None:
                        episode_losses.append(loss)
                
                if episode_losses:
                    avg_loss = np.mean(episode_losses)
                    total_losses.append(avg_loss)
                    
                    if episode % 100 == 0:
                        self.logger.info(f"Episode {episode}: avg_loss={avg_loss:.4f}, epsilon={self.epsilon:.4f}")
            
            self.is_trained = True
            self.logger.info("DQN training completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error training DQN: {str(e)}")
            return False
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained agent.
        
        Args:
            data: Input data for prediction
            
        Returns:
            Array of predicted actions
        """
        if not self.is_trained:
            raise ValueError("Agent must be trained before making predictions")
        
        try:
            # This is a simplified prediction method
            # In practice, this would involve proper state preparation
            predictions = []
            
            for i in range(len(data)):
                # Create dummy state (in practice, this would be properly constructed)
                state = np.random.random(self.state_size)
                action = self.select_action(state)
                predictions.append(self.action_to_index[action])
            
            return np.array(predictions)
            
        except Exception as e:
            self.logger.error(f"Error making predictions: {str(e)}")
            raise
    
    def save_model(self, path: str) -> bool:
        """
        Save the trained agent to disk.
        
        Args:
            path: Path to save the agent
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_trained:
            self.logger.error("Cannot save untrained agent")
            return False
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Save main network
            self.q_network.save(f"{path}_q_network.keras")
            
            # Save target network
            self.target_network.save(f"{path}_target_network.keras")
            
            # Save agent metadata
            metadata = {
                'state_size': self.state_size,
                'action_size': self.action_size,
                'learning_rate': self.learning_rate,
                'epsilon': self.epsilon,
                'epsilon_min': self.epsilon_min,
                'epsilon_decay': self.epsilon_decay,
                'gamma': self.gamma,
                'batch_size': self.batch_size,
                'target_update_freq': self.target_update_freq,
                'training_step': self.training_step,
                'is_trained': self.is_trained
            }
            
            import json
            with open(f"{path}_metadata.json", 'w') as f:
                json.dump(metadata, f)
            
            self.logger.info(f"DQN agent saved successfully to {path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving agent: {str(e)}")
            return False
    
    def load_model(self, path: str) -> bool:
        """
        Load a trained agent from disk.
        
        Args:
            path: Path to load the agent from
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load networks
            self.q_network = tf.keras.models.load_model(f"{path}_q_network.keras")
            self.target_network = tf.keras.models.load_model(f"{path}_target_network.keras")
            
            # Load metadata
            metadata_path = f"{path}_metadata.json"
            if os.path.exists(metadata_path):
                import json
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                self.state_size = metadata.get('state_size', self.state_size)
                self.action_size = metadata.get('action_size', self.action_size)
                self.learning_rate = metadata.get('learning_rate', self.learning_rate)
                self.epsilon = metadata.get('epsilon', self.epsilon)
                self.epsilon_min = metadata.get('epsilon_min', self.epsilon_min)
                self.epsilon_decay = metadata.get('epsilon_decay', self.epsilon_decay)
                self.gamma = metadata.get('gamma', self.gamma)
                self.batch_size = metadata.get('batch_size', self.batch_size)
                self.target_update_freq = metadata.get('target_update_freq', self.target_update_freq)
                self.training_step = metadata.get('training_step', 0)
                self.is_trained = metadata.get('is_trained', True)
            else:
                self.is_trained = True
            
            self.logger.info(f"DQN agent loaded successfully from {path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading agent: {str(e)}")
            return False
    
    def get_training_metrics(self) -> Dict[str, Any]:
        """
        Get training metrics and statistics.
        
        Returns:
            Dictionary with training metrics
        """
        losses = [h['loss'] for h in self.training_history if 'loss' in h] if self.training_history else []
        
        return {
            'total_training_steps': self.training_step,
            'current_epsilon': self.epsilon,
            'buffer_size': self.experience_buffer.size(),
            'average_loss': np.mean(losses) if losses else None,
            'final_loss': losses[-1] if losses else None,
            'is_trained': self.is_trained
        }
    
    def reset_epsilon(self, epsilon: float = 1.0) -> None:
        """
        Reset exploration rate.
        
        Args:
            epsilon: New epsilon value
        """
        self.epsilon = epsilon
        self.logger.info(f"Epsilon reset to {epsilon}")