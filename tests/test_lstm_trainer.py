"""
Integration tests for LSTM training system.
Tests complete LSTM training workflow including data preparation, training, and prediction.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
import json
from unittest.mock import patch, MagicMock
from sklearn.preprocessing import MinMaxScaler

from src.models.lstm_model import LSTMModel
from src.models.lstm_trainer import LSTMTrainer, LSTMModelEvaluator


class TestLSTMTrainer:
    """Test cases for LSTM trainer."""
    
    @pytest.fixture
    def sample_training_data(self):
        """Create comprehensive sample data for training."""
        np.random.seed(42)
        n_samples = 300
        
        # Create realistic price data with trend
        base_price = 100
        price_changes = np.random.randn(n_samples) * 0.02  # 2% daily volatility
        prices = [base_price]
        
        for change in price_changes[1:]:
            prices.append(prices[-1] * (1 + change))
        
        data = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=n_samples, freq='D'),
            'close': prices,
            'high': [p * (1 + abs(np.random.randn() * 0.01)) for p in prices],
            'low': [p * (1 - abs(np.random.randn() * 0.01)) for p in prices],
            'volume': np.random.randint(10000, 100000, n_samples),
            'rsi_14': np.random.uniform(20, 80, n_samples),
            'sma_50': [p * (1 + np.random.randn() * 0.005) for p in prices],
            'bb_upper': [p * 1.02 for p in prices],
            'bb_middle': prices,
            'bb_lower': [p * 0.98 for p in prices],
            'sentiment_score': np.random.uniform(-1, 1, n_samples)
        })
        
        return data
    
    @pytest.fixture
    def lstm_model(self):
        """Create LSTM model for testing."""
        return LSTMModel(sequence_length=20, features=10, learning_rate=0.001)
    
    @pytest.fixture
    def lstm_trainer(self, lstm_model):
        """Create LSTM trainer for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            trainer = LSTMTrainer(lstm_model, checkpoint_dir=temp_dir)
            yield trainer
    
    def test_trainer_initialization(self, lstm_trainer, lstm_model):
        """Test LSTM trainer initialization."""
        assert lstm_trainer.model == lstm_model
        assert isinstance(lstm_trainer.scaler, MinMaxScaler)
        assert lstm_trainer.feature_scalers == {}
        assert not lstm_trainer.is_fitted
        assert lstm_trainer.training_metrics == {}
        assert lstm_trainer.best_model_path is None
    
    def test_prepare_training_data(self, lstm_trainer, sample_training_data):
        """Test training data preparation."""
        X_train, X_test, y_train, y_test = lstm_trainer.prepare_training_data(
            sample_training_data, test_size=0.2
        )
        
        # Check shapes
        total_sequences = len(sample_training_data) - lstm_trainer.model.sequence_length - 1
        train_size = int(total_sequences * 0.8)
        test_size = total_sequences - train_size
        
        assert X_train.shape[0] == train_size
        assert X_test.shape[0] == test_size
        assert X_train.shape[1] == lstm_trainer.model.sequence_length
        assert X_train.shape[2] == len(lstm_trainer.feature_scalers)
        
        # Check target values are binary
        assert set(np.unique(y_train)).issubset({0, 1})
        assert set(np.unique(y_test)).issubset({0, 1})
        
        # Check scalers were fitted
        assert len(lstm_trainer.feature_scalers) > 0
        for scaler in lstm_trainer.feature_scalers.values():
            assert hasattr(scaler, 'scale_')
    
    def test_prepare_training_data_insufficient_features(self, lstm_trainer):
        """Test data preparation with insufficient features."""
        insufficient_data = pd.DataFrame({
            'close': np.random.randn(100),
            'volume': np.random.randint(1000, 10000, 100)
        })
        
        with pytest.raises(ValueError, match="Insufficient features"):
            lstm_trainer.prepare_training_data(insufficient_data)
    
    def test_create_sequences(self, lstm_trainer):
        """Test sequence creation."""
        feature_data = np.random.randn(100, 5)
        target_data = np.random.randint(0, 2, 100)
        
        X, y = lstm_trainer._create_sequences(feature_data, target_data)
        
        expected_samples = len(feature_data) - lstm_trainer.model.sequence_length
        assert X.shape == (expected_samples, lstm_trainer.model.sequence_length, 5)
        assert y.shape == (expected_samples,)
    
    def test_create_callbacks(self, lstm_trainer):
        """Test callback creation."""
        checkpoint_path = "test_checkpoint.keras"
        callbacks = lstm_trainer._create_callbacks(checkpoint_path, patience=10, save_best_only=True)
        
        assert len(callbacks) == 3  # EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
        
        # Check callback types
        callback_types = [type(cb).__name__ for cb in callbacks]
        assert 'EarlyStopping' in callback_types
        assert 'ModelCheckpoint' in callback_types
        assert 'ReduceLROnPlateau' in callback_types
        
        # Check that ModelCheckpoint uses correct weights path
        checkpoint_callback = next(cb for cb in callbacks if type(cb).__name__ == 'ModelCheckpoint')
        assert checkpoint_callback.filepath.endswith('.weights.h5')
    
    def test_evaluate_model(self, lstm_trainer, sample_training_data):
        """Test model evaluation."""
        # Prepare test data
        X_train, X_test, y_train, y_test = lstm_trainer.prepare_training_data(sample_training_data)
        
        # Mock model predictions
        with patch.object(lstm_trainer.model.model, 'predict') as mock_predict:
            mock_predict.return_value = np.random.uniform(0, 1, (len(y_test), 1))
            
            metrics = lstm_trainer._evaluate_model(X_test, y_test)
            
            # Check that all expected metrics are present
            expected_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_score']
            for metric in expected_metrics:
                assert metric in metrics
                assert isinstance(metrics[metric], float)
                assert 0 <= metrics[metric] <= 1
    
    def test_train_model_success(self, lstm_trainer, sample_training_data):
        """Test successful model training."""
        # Mock the model training to avoid actual training
        with patch.object(lstm_trainer.model.model, 'fit') as mock_fit, \
             patch.object(lstm_trainer.model.model, 'load_weights') as mock_load_weights, \
             patch.object(lstm_trainer, '_evaluate_model') as mock_evaluate:
            
            # Setup mocks
            mock_history = MagicMock()
            mock_history.history = {
                'loss': [0.8, 0.6, 0.4],
                'val_loss': [0.9, 0.7, 0.5],
                'accuracy': [0.5, 0.7, 0.8],
                'val_accuracy': [0.4, 0.6, 0.75]
            }
            mock_fit.return_value = mock_history
            mock_evaluate.return_value = {'accuracy': 0.75, 'precision': 0.7}
            
            # Train model
            results = lstm_trainer.train_model(
                sample_training_data,
                epochs=3,
                batch_size=16,
                patience=5
            )
            
            # Check results
            assert isinstance(results, dict)
            assert 'training_history' in results
            assert 'test_metrics' in results
            assert 'model_path' in results
            assert 'epochs_completed' in results
            assert results['epochs_completed'] == 3
            assert lstm_trainer.is_fitted
            assert lstm_trainer.model.is_trained
            
            # Verify mock calls
            mock_fit.assert_called_once()
            mock_evaluate.assert_called_once()
    
    def test_predict_price_movement_probability_untrained(self, lstm_trainer, sample_training_data):
        """Test prediction with untrained model."""
        with pytest.raises(ValueError, match="Model must be trained"):
            lstm_trainer.predict_price_movement_probability(sample_training_data)
    
    def test_predict_price_movement_probability_trained(self, lstm_trainer, sample_training_data):
        """Test prediction with trained model."""
        # Setup trained state
        lstm_trainer.is_fitted = True
        lstm_trainer.feature_scalers = {
            'close': MinMaxScaler().fit([[90], [110]]),
            'volume': MinMaxScaler().fit([[1000], [100000]]),
            'rsi_14': MinMaxScaler().fit([[20], [80]]),
            'sentiment_score': MinMaxScaler().fit([[-1], [1]])
        }
        
        # Mock model prediction
        with patch.object(lstm_trainer.model, 'predict_price_movement') as mock_predict, \
             patch.object(lstm_trainer.model, 'get_model_confidence') as mock_confidence:
            
            mock_predict.return_value = 0.75
            mock_confidence.return_value = 0.85
            
            result = lstm_trainer.predict_price_movement_probability(
                sample_training_data.tail(30), symbol='TEST'
            )
            
            # Check result structure
            assert isinstance(result, dict)
            assert result['symbol'] == 'TEST'
            assert result['probability'] == 0.75
            assert result['confidence'] == 0.85
            assert result['prediction'] == 'UP'
            assert result['strength'] in ['LOW', 'MEDIUM', 'HIGH']
            assert 'timestamp' in result
    
    def test_predict_insufficient_data(self, lstm_trainer):
        """Test prediction with insufficient data."""
        lstm_trainer.is_fitted = True
        lstm_trainer.feature_scalers = {'close': MinMaxScaler().fit([[90], [110]])}
        
        # Create data with insufficient length
        short_data = pd.DataFrame({
            'close': np.random.randn(5)  # Less than sequence_length
        })
        
        with pytest.raises(ValueError, match="Insufficient data for prediction"):
            lstm_trainer.predict_price_movement_probability(short_data)
    
    def test_batch_predict(self, lstm_trainer, sample_training_data):
        """Test batch prediction for multiple stocks."""
        lstm_trainer.is_fitted = True
        lstm_trainer.feature_scalers = {
            'close': MinMaxScaler().fit([[90], [110]]),
            'volume': MinMaxScaler().fit([[1000], [100000]])
        }
        
        # Create data for multiple stocks
        data_dict = {
            'STOCK1': sample_training_data.tail(30),
            'STOCK2': sample_training_data.tail(25)
        }
        
        with patch.object(lstm_trainer, 'predict_price_movement_probability') as mock_predict:
            mock_predict.side_effect = [
                {'symbol': 'STOCK1', 'probability': 0.6},
                {'symbol': 'STOCK2', 'probability': 0.4}
            ]
            
            results = lstm_trainer.batch_predict(data_dict)
            
            assert len(results) == 2
            assert 'STOCK1' in results
            assert 'STOCK2' in results
            assert results['STOCK1']['probability'] == 0.6
            assert results['STOCK2']['probability'] == 0.4
    
    def test_batch_predict_with_errors(self, lstm_trainer, sample_training_data):
        """Test batch prediction with some errors."""
        lstm_trainer.is_fitted = True
        
        data_dict = {
            'GOOD_STOCK': sample_training_data.tail(30),
            'BAD_STOCK': pd.DataFrame({'close': [1, 2]})  # Insufficient data
        }
        
        with patch.object(lstm_trainer, 'predict_price_movement_probability') as mock_predict:
            mock_predict.side_effect = [
                {'symbol': 'GOOD_STOCK', 'probability': 0.6},
                ValueError("Insufficient data")
            ]
            
            results = lstm_trainer.batch_predict(data_dict)
            
            assert len(results) == 2
            assert 'probability' in results['GOOD_STOCK']
            assert 'error' in results['BAD_STOCK']
    
    def test_save_load_training_state(self, lstm_trainer):
        """Test saving and loading training state."""
        # Setup training state
        lstm_trainer.is_fitted = True
        lstm_trainer.training_metrics = {'accuracy': 0.8}
        lstm_trainer.best_model_path = 'test_model.keras'
        lstm_trainer.feature_scalers = {
            'close': MinMaxScaler().fit([[90], [110]])
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filepath = f.name
        
        try:
            # Save state
            result = lstm_trainer.save_training_state(filepath)
            assert result == True
            assert os.path.exists(filepath)
            
            # Create new trainer and load state
            new_model = LSTMModel(sequence_length=20, features=10)
            new_trainer = LSTMTrainer(new_model)
            
            load_result = new_trainer.load_training_state(filepath)
            assert load_result == True
            assert new_trainer.is_fitted == True
            assert new_trainer.training_metrics['accuracy'] == 0.8
            assert new_trainer.best_model_path == 'test_model.keras'
            assert 'close' in new_trainer.feature_scalers
            
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)
    
    def test_get_training_summary_untrained(self, lstm_trainer):
        """Test training summary for untrained model."""
        summary = lstm_trainer.get_training_summary()
        assert summary['status'] == 'not_trained'
    
    def test_get_training_summary_trained(self, lstm_trainer):
        """Test training summary for trained model."""
        # Setup trained state
        lstm_trainer.is_fitted = True
        lstm_trainer.training_metrics = {
            'test_metrics': {'accuracy': 0.8},
            'epochs_completed': 50
        }
        lstm_trainer.feature_scalers = {'close': MinMaxScaler(), 'volume': MinMaxScaler()}
        lstm_trainer.best_model_path = 'best_model.keras'
        
        summary = lstm_trainer.get_training_summary()
        
        assert summary['status'] == 'trained'
        assert 'model_architecture' in summary
        assert 'training_results' in summary
        assert summary['feature_scalers'] == ['close', 'volume']
        assert summary['best_model_path'] == 'best_model.keras'


class TestLSTMModelEvaluator:
    """Test cases for LSTM model evaluator."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for evaluation."""
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
    def evaluator(self):
        """Create model evaluator."""
        return LSTMModelEvaluator()
    
    def test_evaluator_initialization(self, evaluator):
        """Test evaluator initialization."""
        assert isinstance(evaluator, LSTMModelEvaluator)
        assert hasattr(evaluator, 'logger')
    
    def test_evaluate_lstm_performance_untrained(self, evaluator, sample_data):
        """Test evaluation with untrained model."""
        model = LSTMModel()
        
        with pytest.raises(ValueError, match="Model must be trained"):
            evaluator.evaluate_lstm_performance(model, sample_data)
    
    def test_evaluate_lstm_performance_trained(self, evaluator, sample_data):
        """Test evaluation with trained model."""
        model = LSTMModel()
        model.is_trained = True
        
        # Mock the prediction and data preparation
        with patch.object(model, 'predict') as mock_predict, \
             patch.object(model, 'get_model_confidence') as mock_confidence, \
             patch('src.models.lstm_trainer.LSTMTrainer.prepare_training_data') as mock_prepare:
            
            # Setup mocks
            n_samples = 50
            mock_predict.return_value = np.random.uniform(0, 1, n_samples)
            mock_confidence.return_value = 0.85
            mock_prepare.return_value = (
                np.random.randn(n_samples, 20, 10),  # X_test
                None,  # X_train (not used)
                np.random.randint(0, 2, n_samples),  # y_test
                None   # y_train (not used)
            )
            
            metrics = evaluator.evaluate_lstm_performance(model, sample_data)
            
            # Check that all expected metrics are present
            expected_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_score', 'model_confidence']
            for metric in expected_metrics:
                assert metric in metrics
                assert isinstance(metrics[metric], float)
            
            assert 'confusion_matrix' in metrics
            assert isinstance(metrics['confusion_matrix'], list)
    
    def test_evaluate_dqn_performance_placeholder(self, evaluator, sample_data):
        """Test DQN evaluation placeholder."""
        result = evaluator.evaluate_dqn_performance(None, sample_data)
        assert result['status'] == 'not_implemented'


if __name__ == '__main__':
    pytest.main([__file__])