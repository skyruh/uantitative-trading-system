"""
Unit tests for LSTM model architecture and functionality.
Tests model creation, training, prediction, and persistence.
"""

import pytest
import numpy as np
import pandas as pd
import tensorflow as tf
import tempfile
import os
from unittest.mock import patch, MagicMock

from src.models.lstm_model import LSTMModel


class TestLSTMModel:
    """Test cases for LSTM model."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        n_samples = 200
        
        data = pd.DataFrame({
            'close': np.random.randn(n_samples).cumsum() + 100,
            'high': np.random.randn(n_samples).cumsum() + 105,
            'low': np.random.randn(n_samples).cumsum() + 95,
            'volume': np.random.randint(1000, 10000, n_samples),
            'rsi_14': np.random.uniform(20, 80, n_samples),
            'sma_50': np.random.randn(n_samples).cumsum() + 100,
            'bb_upper': np.random.randn(n_samples).cumsum() + 110,
            'bb_middle': np.random.randn(n_samples).cumsum() + 100,
            'bb_lower': np.random.randn(n_samples).cumsum() + 90,
            'sentiment_score': np.random.uniform(-1, 1, n_samples)
        })
        
        return data
    
    @pytest.fixture
    def lstm_model(self):
        """Create LSTM model instance for testing."""
        return LSTMModel(sequence_length=10, features=10, learning_rate=0.001)
    
    def test_model_initialization(self, lstm_model):
        """Test LSTM model initialization."""
        assert lstm_model.sequence_length == 10
        assert lstm_model.features == 10
        assert lstm_model.learning_rate == 0.001
        assert lstm_model.random_seed == 42
        assert lstm_model.model is not None
        assert not lstm_model.is_trained
        assert lstm_model.training_history is None
    
    def test_model_architecture(self, lstm_model):
        """Test LSTM model architecture specifications."""
        model = lstm_model.model
        
        # Check model structure
        assert len(model.layers) == 5  # 2 LSTM + 2 Dropout + 1 Dense
        
        # Check LSTM layers
        lstm_layers = [layer for layer in model.layers if 'lstm' in layer.name]
        assert len(lstm_layers) == 2
        
        # Check first LSTM layer
        first_lstm = lstm_layers[0]
        assert first_lstm.units == 50
        assert first_lstm.return_sequences == True
        
        # Check second LSTM layer
        second_lstm = lstm_layers[1]
        assert second_lstm.units == 50
        assert second_lstm.return_sequences == False
        
        # Check dropout layers
        dropout_layers = [layer for layer in model.layers if 'dropout' in layer.name]
        assert len(dropout_layers) == 2
        for dropout_layer in dropout_layers:
            assert dropout_layer.rate == 0.2
        
        # Check output layer
        output_layer = model.layers[-1]
        assert output_layer.units == 1
        assert output_layer.activation.__name__ == 'sigmoid'
    
    def test_model_compilation(self, lstm_model):
        """Test model compilation settings."""
        model = lstm_model.model
        
        # Check optimizer
        assert isinstance(model.optimizer, tf.keras.optimizers.Adam)
        assert abs(model.optimizer.learning_rate.numpy() - 0.001) < 1e-6
        
        # Check loss function
        assert model.loss == 'binary_crossentropy'
        
        # Check that model is compiled (has metrics)
        assert hasattr(model, 'compiled_metrics')
        assert model.compiled_metrics is not None
    
    def test_get_model_summary(self, lstm_model):
        """Test model summary generation."""
        summary = lstm_model.get_model_summary()
        assert isinstance(summary, str)
        assert len(summary) > 0
        assert 'lstm_layer_1' in summary
        assert 'lstm_layer_2' in summary
        assert 'dropout' in summary
        assert 'output_layer' in summary
    
    def test_prepare_sequences(self, lstm_model, sample_data):
        """Test sequence preparation for training."""
        X, y = lstm_model._prepare_sequences(sample_data)
        
        # Check shapes
        expected_samples = len(sample_data) - lstm_model.sequence_length - 1
        assert X.shape == (expected_samples, lstm_model.sequence_length, len(sample_data.columns))
        assert y.shape == (expected_samples,)
        
        # Check data types
        assert X.dtype == np.float64
        assert y.dtype in [np.int32, np.int64]
        
        # Check target values are binary
        assert set(np.unique(y)).issubset({0, 1})
    
    def test_prepare_sequences_insufficient_features(self, lstm_model):
        """Test sequence preparation with insufficient features."""
        # Create data with only 2 columns (less than required minimum of 3)
        insufficient_data = pd.DataFrame({
            'close': np.random.randn(100),
            'volume': np.random.randint(1000, 10000, 100)
        })
        
        with pytest.raises(ValueError, match="Insufficient features"):
            lstm_model._prepare_sequences(insufficient_data)
    
    def test_train_success(self, lstm_model, sample_data):
        """Test successful model training."""
        # Mock the fit method to avoid actual training
        with patch.object(lstm_model.model, 'fit') as mock_fit:
            mock_history = MagicMock()
            mock_history.history = {
                'loss': [0.7, 0.6, 0.5],
                'val_loss': [0.8, 0.7, 0.6],
                'accuracy': [0.5, 0.6, 0.7],
                'val_accuracy': [0.4, 0.5, 0.6]
            }
            mock_fit.return_value = mock_history
            
            result = lstm_model.train(sample_data, epochs=3, batch_size=16)
            
            assert result == True
            assert lstm_model.is_trained == True
            assert lstm_model.training_history is not None
            mock_fit.assert_called_once()
    
    def test_train_insufficient_data(self, lstm_model):
        """Test training with insufficient data."""
        # Create very small dataset
        small_data = pd.DataFrame({
            'close': np.random.randn(20),
            'rsi_14': np.random.uniform(20, 80, 20),
            'sma_50': np.random.randn(20),
            'sentiment_score': np.random.uniform(-1, 1, 20)
        })
        
        result = lstm_model.train(small_data)
        assert result == False
        assert lstm_model.is_trained == False
    
    def test_predict_untrained_model(self, lstm_model, sample_data):
        """Test prediction with untrained model."""
        with pytest.raises(ValueError, match="Model must be trained"):
            lstm_model.predict(sample_data)
    
    def test_predict_trained_model(self, lstm_model, sample_data):
        """Test prediction with trained model."""
        # Mock training
        lstm_model.is_trained = True
        
        with patch.object(lstm_model.model, 'predict') as mock_predict:
            mock_predict.return_value = np.array([[0.6], [0.7], [0.4]])
            
            predictions = lstm_model.predict(sample_data)
            
            assert isinstance(predictions, np.ndarray)
            assert len(predictions) == 3
            assert all(0 <= p <= 1 for p in predictions)
    
    def test_predict_price_movement(self, lstm_model):
        """Test price movement prediction."""
        lstm_model.is_trained = True
        
        # Create sample sequence data
        sequence_data = np.random.randn(10, 10)  # (sequence_length, features)
        
        with patch.object(lstm_model.model, 'predict') as mock_predict:
            mock_predict.return_value = np.array([[0.75]])
            
            probability = lstm_model.predict_price_movement(sequence_data)
            
            assert isinstance(probability, float)
            assert 0 <= probability <= 1
            assert probability == 0.75
    
    def test_predict_price_movement_untrained(self, lstm_model):
        """Test price movement prediction with untrained model."""
        sequence_data = np.random.randn(10, 10)
        
        with pytest.raises(ValueError, match="Model must be trained"):
            lstm_model.predict_price_movement(sequence_data)
    
    def test_get_model_confidence_untrained(self, lstm_model):
        """Test confidence calculation for untrained model."""
        confidence = lstm_model.get_model_confidence()
        assert confidence == 0.0
    
    def test_get_model_confidence_trained(self, lstm_model):
        """Test confidence calculation for trained model."""
        # Mock training history
        lstm_model.is_trained = True
        mock_history = MagicMock()
        mock_history.history = {'val_accuracy': [0.5, 0.6, 0.8, 0.7]}
        lstm_model.training_history = mock_history
        
        confidence = lstm_model.get_model_confidence()
        assert confidence == 0.8  # Maximum validation accuracy
    
    def test_save_model_untrained(self, lstm_model):
        """Test saving untrained model."""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, 'test_model')
            result = lstm_model.save_model(model_path)
            assert result == False
    
    def test_save_load_model_trained(self, lstm_model, sample_data):
        """Test saving and loading trained model."""
        # Mock training
        lstm_model.is_trained = True
        
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, 'test_model')
            
            # Mock save method
            with patch.object(lstm_model.model, 'save') as mock_save:
                result = lstm_model.save_model(model_path)
                assert result == True
                mock_save.assert_called_once_with(model_path)
                
                # Check metadata file creation
                metadata_path = f"{model_path}_metadata.json"
                assert os.path.exists(metadata_path)
    
    def test_load_model(self, lstm_model):
        """Test loading model from disk."""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, 'test_model')
            
            # Create mock model file
            with patch('tensorflow.keras.models.load_model') as mock_load:
                mock_model = MagicMock()
                mock_load.return_value = mock_model
                
                result = lstm_model.load_model(model_path)
                
                assert result == True
                assert lstm_model.model == mock_model
                assert lstm_model.is_trained == True
                mock_load.assert_called_once_with(model_path)
    
    def test_get_training_metrics_no_history(self, lstm_model):
        """Test getting training metrics without training history."""
        metrics = lstm_model.get_training_metrics()
        assert metrics == {}
    
    def test_get_training_metrics_with_history(self, lstm_model):
        """Test getting training metrics with training history."""
        # Mock training history
        mock_history = MagicMock()
        mock_history.history = {
            'loss': [0.8, 0.6, 0.4],
            'val_loss': [0.9, 0.7, 0.5],
            'accuracy': [0.5, 0.7, 0.8],
            'val_accuracy': [0.4, 0.6, 0.75]
        }
        lstm_model.training_history = mock_history
        
        metrics = lstm_model.get_training_metrics()
        
        assert metrics['final_loss'] == 0.4
        assert metrics['final_val_loss'] == 0.5
        assert metrics['final_accuracy'] == 0.8
        assert metrics['final_val_accuracy'] == 0.75
        assert metrics['epochs_trained'] == 3
        assert metrics['best_val_accuracy'] == 0.75
    
    def test_model_reproducibility(self):
        """Test model reproducibility with same random seed."""
        # Test that models with same seed have consistent behavior
        model1 = LSTMModel(random_seed=123)
        model2 = LSTMModel(random_seed=123)
        
        # Both models should have the same architecture
        assert model1.sequence_length == model2.sequence_length
        assert model1.features == model2.features
        assert model1.learning_rate == model2.learning_rate
        assert model1.random_seed == model2.random_seed
    
    def test_different_sequence_lengths(self):
        """Test model with different sequence lengths."""
        model_short = LSTMModel(sequence_length=5, features=8)
        model_long = LSTMModel(sequence_length=30, features=8)
        
        assert model_short.model.input_shape == (None, 5, 8)
        assert model_long.model.input_shape == (None, 30, 8)
    
    def test_error_handling_in_training(self, lstm_model, sample_data):
        """Test error handling during training."""
        # Mock fit to raise an exception
        with patch.object(lstm_model.model, 'fit', side_effect=Exception("Training error")):
            result = lstm_model.train(sample_data)
            assert result == False
            assert lstm_model.is_trained == False


if __name__ == '__main__':
    pytest.main([__file__])