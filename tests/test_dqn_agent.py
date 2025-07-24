"""
Unit tests for DQN Agent implementation.
Tests DQN architecture, action selection, experience replay, and training functionality.
"""

import unittest
import numpy as np
import pandas as pd
import tempfile
import shutil
import os
from unittest.mock import patch, MagicMock

# Add src to path for imports
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.dqn_agent import DQNAgent, ExperienceReplayBuffer


class TestExperienceReplayBuffer(unittest.TestCase):
    """Test cases for Experience Replay Buffer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.buffer = ExperienceReplayBuffer(capacity=100)
    
    def test_buffer_initialization(self):
        """Test buffer initialization."""
        self.assertEqual(self.buffer.capacity, 100)
        self.assertEqual(self.buffer.size(), 0)
        self.assertFalse(self.buffer.is_ready(10))
    
    def test_add_experience(self):
        """Test adding experience to buffer."""
        state = np.array([1, 2, 3])
        next_state = np.array([4, 5, 6])
        
        self.buffer.add_experience(state, 0, 1.0, next_state, False)
        
        self.assertEqual(self.buffer.size(), 1)
        self.assertTrue(self.buffer.is_ready(1))
    
    def test_sample_batch(self):
        """Test sampling batch from buffer."""
        # Add multiple experiences
        for i in range(10):
            state = np.array([i, i+1, i+2])
            next_state = np.array([i+1, i+2, i+3])
            self.buffer.add_experience(state, i % 3, float(i), next_state, False)
        
        # Sample batch
        batch = self.buffer.sample_batch(5)
        self.assertEqual(len(batch), 5)
        
        # Test sampling more than available
        batch_large = self.buffer.sample_batch(20)
        self.assertEqual(len(batch_large), 10)
    
    def test_buffer_capacity_limit(self):
        """Test buffer capacity limit."""
        buffer = ExperienceReplayBuffer(capacity=5)
        
        # Add more experiences than capacity
        for i in range(10):
            state = np.array([i])
            next_state = np.array([i+1])
            buffer.add_experience(state, 0, float(i), next_state, False)
        
        # Should only keep last 5 experiences
        self.assertEqual(buffer.size(), 5)


class TestDQNAgent(unittest.TestCase):
    """Test cases for DQN Agent."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.agent = DQNAgent(
            state_size=5,
            learning_rate=0.001,
            epsilon=1.0,
            batch_size=4,
            buffer_capacity=100,
            random_seed=42
        )
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_agent_initialization(self):
        """Test DQN agent initialization."""
        self.assertEqual(self.agent.state_size, 5)
        self.assertEqual(self.agent.action_size, 3)
        self.assertEqual(self.agent.actions, ['buy', 'sell', 'hold'])
        self.assertEqual(self.agent.epsilon, 1.0)
        self.assertFalse(self.agent.is_trained)
        self.assertIsNotNone(self.agent.q_network)
        self.assertIsNotNone(self.agent.target_network)
        self.assertIsNotNone(self.agent.experience_buffer)
    
    def test_network_architecture(self):
        """Test neural network architecture."""
        # Test input shape
        test_input = np.random.random((1, 5))
        output = self.agent.q_network.predict(test_input, verbose=0)
        
        # Should output Q-values for 3 actions
        self.assertEqual(output.shape, (1, 3))
        
        # Test that networks have same architecture
        q_layers = len(self.agent.q_network.layers)
        target_layers = len(self.agent.target_network.layers)
        self.assertEqual(q_layers, target_layers)
    
    def test_action_selection_exploration(self):
        """Test action selection during exploration (high epsilon)."""
        state = np.random.random(5)
        
        # With epsilon=1.0, should always explore (random actions)
        actions = []
        for _ in range(100):
            action = self.agent.select_action(state)
            actions.append(action)
        
        # Should have variety in actions due to randomness
        unique_actions = set(actions)
        self.assertTrue(len(unique_actions) > 1)
        
        # All actions should be valid
        for action in actions:
            self.assertIn(action, ['buy', 'sell', 'hold'])
    
    def test_action_selection_exploitation(self):
        """Test action selection during exploitation (low epsilon)."""
        # Set epsilon to 0 for pure exploitation
        self.agent.epsilon = 0.0
        
        state = np.random.random(5)
        
        # Should consistently select same action (greedy)
        actions = []
        for _ in range(10):
            action = self.agent.select_action(state)
            actions.append(action)
        
        # All actions should be the same (greedy selection)
        self.assertEqual(len(set(actions)), 1)
        self.assertIn(actions[0], ['buy', 'sell', 'hold'])
    
    def test_get_q_values(self):
        """Test Q-value retrieval."""
        state = np.random.random(5)
        q_values = self.agent.get_q_values(state)
        
        # Should return dict with all actions
        self.assertIsInstance(q_values, dict)
        self.assertEqual(set(q_values.keys()), {'buy', 'sell', 'hold'})
        
        # All values should be floats
        for value in q_values.values():
            self.assertIsInstance(value, float)
    
    def test_update_experience(self):
        """Test experience buffer update."""
        state = np.random.random(5)
        next_state = np.random.random(5)
        
        initial_size = self.agent.experience_buffer.size()
        
        self.agent.update_experience(state, 'buy', 1.0, next_state, False)
        
        self.assertEqual(self.agent.experience_buffer.size(), initial_size + 1)
    
    def test_calculate_reward(self):
        """Test reward calculation."""
        # Test positive excess return
        reward1 = self.agent.calculate_reward(0.15, 0.10)
        self.assertGreater(reward1, 0)
        
        # Test negative excess return
        reward2 = self.agent.calculate_reward(0.05, 0.10)
        self.assertLess(reward2, 0)
        
        # Test equal returns
        reward3 = self.agent.calculate_reward(0.10, 0.10)
        self.assertAlmostEqual(reward3, 0, places=3)
        
        # Reward should be in reasonable range
        self.assertGreaterEqual(reward1, -1)
        self.assertLessEqual(reward1, 1)
    
    def test_train_step_insufficient_data(self):
        """Test training step with insufficient data."""
        # Should return None when buffer doesn't have enough experiences
        loss = self.agent.train_step()
        self.assertIsNone(loss)
    
    def test_train_step_with_data(self):
        """Test training step with sufficient data."""
        # Add experiences to buffer
        for i in range(10):
            state = np.random.random(5)
            next_state = np.random.random(5)
            action = ['buy', 'sell', 'hold'][i % 3]
            reward = np.random.random()
            
            self.agent.update_experience(state, action, reward, next_state, False)
        
        # Should be able to perform training step
        loss = self.agent.train_step()
        self.assertIsNotNone(loss)
        self.assertIsInstance(loss, float)
        self.assertGreaterEqual(loss, 0)
    
    def test_epsilon_decay(self):
        """Test epsilon decay during training."""
        initial_epsilon = self.agent.epsilon
        
        # Add experiences and perform training steps
        for i in range(20):
            state = np.random.random(5)
            next_state = np.random.random(5)
            action = ['buy', 'sell', 'hold'][i % 3]
            reward = np.random.random()
            
            self.agent.update_experience(state, action, reward, next_state, False)
            self.agent.train_step()
        
        # Epsilon should have decayed
        self.assertLess(self.agent.epsilon, initial_epsilon)
        self.assertGreaterEqual(self.agent.epsilon, self.agent.epsilon_min)
    
    def test_target_network_update(self):
        """Test target network update."""
        # Get initial weights
        initial_weights = self.agent.target_network.get_weights()
        
        # Modify main network weights by training
        for i in range(10):
            state = np.random.random(5)
            next_state = np.random.random(5)
            action = ['buy', 'sell', 'hold'][i % 3]
            reward = np.random.random()
            
            self.agent.update_experience(state, action, reward, next_state, False)
            self.agent.train_step()
        
        # Force target network update
        self.agent._update_target_network()
        
        # Target network weights should now match main network
        main_weights = self.agent.q_network.get_weights()
        target_weights = self.agent.target_network.get_weights()
        
        for main_w, target_w in zip(main_weights, target_weights):
            np.testing.assert_array_equal(main_w, target_w)
    
    def test_training_metrics(self):
        """Test training metrics retrieval."""
        # Initially should have minimal metrics
        metrics = self.agent.get_training_metrics()
        self.assertEqual(metrics['total_training_steps'], 0)
        self.assertEqual(metrics['buffer_size'], 0)
        
        # Add some training data
        for i in range(10):
            state = np.random.random(5)
            next_state = np.random.random(5)
            action = ['buy', 'sell', 'hold'][i % 3]
            reward = np.random.random()
            
            self.agent.update_experience(state, action, reward, next_state, False)
            self.agent.train_step()
        
        # Should have updated metrics
        metrics = self.agent.get_training_metrics()
        self.assertGreater(metrics['total_training_steps'], 0)
        self.assertGreater(metrics['buffer_size'], 0)
        self.assertIsNotNone(metrics['current_epsilon'])
    
    def test_save_and_load_model(self):
        """Test model saving and loading."""
        # Train agent briefly
        self.agent.is_trained = True
        
        # Save model
        model_path = os.path.join(self.temp_dir, 'test_dqn')
        success = self.agent.save_model(model_path)
        self.assertTrue(success)
        
        # Check files were created
        self.assertTrue(os.path.exists(f"{model_path}_q_network.keras"))
        self.assertTrue(os.path.exists(f"{model_path}_target_network.keras"))
        self.assertTrue(os.path.exists(f"{model_path}_metadata.json"))
        
        # Create new agent and load model
        new_agent = DQNAgent(state_size=5)
        load_success = new_agent.load_model(model_path)
        self.assertTrue(load_success)
        self.assertTrue(new_agent.is_trained)
    
    def test_save_untrained_model(self):
        """Test saving untrained model should fail."""
        model_path = os.path.join(self.temp_dir, 'untrained_dqn')
        success = self.agent.save_model(model_path)
        self.assertFalse(success)
    
    def test_predict_untrained_agent(self):
        """Test prediction with untrained agent should raise error."""
        data = pd.DataFrame({'close': [100, 101, 102]})
        
        with self.assertRaises(ValueError):
            self.agent.predict(data)
    
    def test_predict_trained_agent(self):
        """Test prediction with trained agent."""
        self.agent.is_trained = True
        data = pd.DataFrame({'close': [100, 101, 102]})
        
        predictions = self.agent.predict(data)
        
        self.assertEqual(len(predictions), len(data))
        self.assertTrue(all(pred in [0, 1, 2] for pred in predictions))
    
    def test_reset_epsilon(self):
        """Test epsilon reset functionality."""
        self.agent.epsilon = 0.1
        self.agent.reset_epsilon(0.8)
        self.assertEqual(self.agent.epsilon, 0.8)
    
    def test_action_mapping_consistency(self):
        """Test action mapping consistency."""
        # Test forward and reverse mapping
        for action in self.agent.actions:
            index = self.agent.action_to_index[action]
            reverse_action = self.agent.index_to_action[index]
            self.assertEqual(action, reverse_action)
        
        # Test all indices are covered
        self.assertEqual(len(self.agent.action_to_index), 3)
        self.assertEqual(len(self.agent.index_to_action), 3)
        self.assertEqual(set(self.agent.action_to_index.values()), {0, 1, 2})
    
    def test_error_handling_invalid_state(self):
        """Test error handling with invalid state."""
        # Test with None state
        action = self.agent.select_action(None)
        self.assertEqual(action, 'hold')  # Should default to hold
        
        q_values = self.agent.get_q_values(None)
        self.assertEqual(q_values, {'buy': 0.0, 'sell': 0.0, 'hold': 0.0})


if __name__ == '__main__':
    # Set up logging to reduce noise during testing
    import logging
    logging.getLogger().setLevel(logging.CRITICAL)
    
    unittest.main()