"""
LSTM Training System for the quantitative trading system.
Implements model training pipeline with checkpointing, early stopping, and prediction functions.
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple, Optional, Any
import logging
import json
from datetime import datetime

from .lstm_model import LSTMModel
from ..interfaces.model_interfaces import IModelEvaluator


class LSTMTrainer:
    """
    LSTM training system with comprehensive training pipeline.
    
    Features:
    - Sequential price data preparation
    - Model checkpointing and early stopping
    - Prediction functions for next-day price movement probability
    - Performance evaluation and metrics tracking
    """
    
    def __init__(self, model: LSTMModel, checkpoint_dir: str = "models/checkpoints"):
        """
        Initialize LSTM trainer.
        
        Args:
            model: LSTMModel instance to train
            checkpoint_dir: Directory to save model checkpoints
        """
        self.model = model
        self.checkpoint_dir = checkpoint_dir
        self.scaler = MinMaxScaler()
        self.feature_scalers = {}
        self.logger = logging.getLogger(__name__)
        
        # Training state
        self.is_fitted = False
        self.training_metrics = {}
        self.best_model_path = None
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def prepare_training_data(self, data: pd.DataFrame, 
                            target_column: str = 'close',
                            test_size: float = 0.2,
                            random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare sequential training data with proper scaling and splitting.
        
        Args:
            data: Input DataFrame with price and feature data
            target_column: Column to use for target variable creation
            test_size: Fraction of data for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        try:
            self.logger.info("Preparing training data...")
            
            # Ensure data is sorted by date if date column exists
            if 'date' in data.columns:
                data = data.sort_values('date').reset_index(drop=True)
            
            # Define feature columns
            feature_columns = [
                'close', 'high', 'low', 'volume', 'rsi_14', 'sma_50',
                'bb_upper', 'bb_middle', 'bb_lower', 'sentiment_score'
            ]
            
            # Use available columns
            available_features = [col for col in feature_columns if col in data.columns]
            if len(available_features) < 3:
                raise ValueError(f"Insufficient features. Need at least 3, got {len(available_features)}")
            
            # Create target variable (1 if next day price > today price, 0 otherwise)
            data = data.copy()
            data['target'] = (data[target_column].shift(-1) > data[target_column]).astype(int)
            
            # Remove last row (no target available)
            data = data[:-1]
            
            # Scale features
            feature_data = data[available_features].copy()
            
            # Apply scaling to each feature separately
            scaled_features = {}
            for col in available_features:
                scaler = MinMaxScaler()
                scaled_features[col] = scaler.fit_transform(feature_data[[col]]).flatten()
                self.feature_scalers[col] = scaler
            
            scaled_df = pd.DataFrame(scaled_features)
            
            # Create sequences
            X, y = self._create_sequences(scaled_df.values, data['target'].values)
            
            # Split data temporally (earlier data for training, later for testing)
            split_idx = int(len(X) * (1 - test_size))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            self.logger.info(f"Training data prepared: {X_train.shape[0]} train, {X_test.shape[0]} test samples")
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            self.logger.error(f"Error preparing training data: {str(e)}")
            raise
    
    def _create_sequences(self, feature_data: np.ndarray, target_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequential data for LSTM training.
        
        Args:
            feature_data: Scaled feature data
            target_data: Target values
            
        Returns:
            Tuple of (X, y) sequences
        """
        X, y = [], []
        sequence_length = self.model.sequence_length
        
        for i in range(sequence_length, len(feature_data)):
            X.append(feature_data[i-sequence_length:i])
            y.append(target_data[i])
        
        return np.array(X), np.array(y)
    
    def train_model(self, data: pd.DataFrame,
                   epochs: int = 100,
                   batch_size: int = 32,
                   validation_split: float = 0.2,
                   patience: int = 15,
                   save_best_only: bool = True) -> Dict[str, Any]:
        """
        Train the LSTM model with comprehensive training pipeline.
        
        Args:
            data: Training data
            epochs: Maximum number of training epochs
            batch_size: Training batch size
            validation_split: Fraction for validation
            patience: Early stopping patience
            save_best_only: Whether to save only the best model
            
        Returns:
            Dictionary with training results and metrics
        """
        try:
            self.logger.info("Starting LSTM model training...")
            
            # Prepare training data
            X_train, X_test, y_train, y_test = self.prepare_training_data(data)
            
            # Create checkpoint path
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_path = os.path.join(self.checkpoint_dir, f"lstm_model_{timestamp}.keras")
            self.best_model_path = checkpoint_path
            
            # Configure callbacks
            callbacks = self._create_callbacks(checkpoint_path, patience, save_best_only)
            
            # Train the model
            self.logger.info(f"Training with {X_train.shape[0]} samples...")
            history = self.model.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                callbacks=callbacks,
                verbose=1,
                shuffle=False  # Don't shuffle to maintain temporal order
            )
            
            # Load best model if checkpointing was used
            weights_path = checkpoint_path.replace('.keras', '.weights.h5')
            if save_best_only and os.path.exists(weights_path):
                self.model.model.load_weights(weights_path)
                self.logger.info("Loaded best model weights from checkpoint")
            
            # Evaluate on test set
            test_metrics = self._evaluate_model(X_test, y_test)
            
            # Store training results
            self.model.training_history = history
            self.model.is_trained = True
            self.is_fitted = True
            
            # Compile training metrics
            training_results = {
                'training_history': history.history,
                'test_metrics': test_metrics,
                'model_path': checkpoint_path,
                'training_samples': X_train.shape[0],
                'test_samples': X_test.shape[0],
                'epochs_completed': len(history.history['loss']),
                'best_val_accuracy': max(history.history.get('val_accuracy', [0])),
                'final_test_accuracy': test_metrics.get('accuracy', 0)
            }
            
            self.training_metrics = training_results
            self.logger.info("LSTM model training completed successfully")
            
            return training_results
            
        except Exception as e:
            self.logger.error(f"Error training LSTM model: {str(e)}")
            raise
    
    def _create_callbacks(self, checkpoint_path: str, patience: int, save_best_only: bool) -> List:
        """
        Create training callbacks for checkpointing and early stopping.
        
        Args:
            checkpoint_path: Path to save model checkpoints
            patience: Early stopping patience
            save_best_only: Whether to save only the best model
            
        Returns:
            List of Keras callbacks
        """
        callbacks = []
        
        # Early stopping removed - training for full epochs
        # early_stopping = EarlyStopping(
        #     monitor='val_loss',
        #     patience=patience,
        #     restore_best_weights=True,
        #     verbose=1,
        #     mode='min'
        # )
        # callbacks.append(early_stopping)
        
        # Model checkpointing
        if save_best_only:
            # Use .weights.h5 extension for weights-only saving
            weights_path = checkpoint_path.replace('.keras', '.weights.h5')
            checkpoint = ModelCheckpoint(
                filepath=weights_path,
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=True,
                verbose=1,
                mode='max'
            )
            callbacks.append(checkpoint)
        
        # Learning rate reduction
        lr_scheduler = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=patience // 2,
            min_lr=1e-7,
            verbose=1,
            mode='min'
        )
        callbacks.append(lr_scheduler)
        
        return callbacks
    
    def _evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance on test data.
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary with evaluation metrics
        """
        try:
            # Get predictions
            y_pred_prob = self.model.model.predict(X_test, verbose=0)
            y_pred = (y_pred_prob > 0.5).astype(int).flatten()
            
            # Calculate metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
            
            metrics = {
                'accuracy': float(accuracy_score(y_test, y_pred)),
                'precision': float(precision_score(y_test, y_pred, zero_division=0)),
                'recall': float(recall_score(y_test, y_pred, zero_division=0)),
                'f1_score': float(f1_score(y_test, y_pred, zero_division=0)),
                'auc_score': float(roc_auc_score(y_test, y_pred_prob.flatten()))
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error evaluating model: {str(e)}")
            return {}
    
    def predict_price_movement_probability(self, data: pd.DataFrame, 
                                         symbol: str = None) -> Dict[str, Any]:
        """
        Predict next-day price movement probability for given data.
        
        Args:
            data: Input data for prediction
            symbol: Stock symbol (optional, for logging)
            
        Returns:
            Dictionary with prediction results
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")
        
        try:
            # Prepare data for prediction
            feature_columns = list(self.feature_scalers.keys())
            available_features = [col for col in feature_columns if col in data.columns]
            
            if len(available_features) < len(feature_columns):
                self.logger.warning(f"Missing features for prediction: {set(feature_columns) - set(available_features)}")
            
            # Scale features using fitted scalers
            scaled_data = {}
            for col in available_features:
                if col in self.feature_scalers:
                    scaled_data[col] = self.feature_scalers[col].transform(data[[col]]).flatten()
                else:
                    scaled_data[col] = data[col].values
            
            scaled_df = pd.DataFrame(scaled_data)
            
            # Create sequences
            if len(scaled_df) < self.model.sequence_length:
                raise ValueError(f"Insufficient data for prediction. Need at least {self.model.sequence_length} samples")
            
            # Get the last sequence for prediction
            last_sequence = scaled_df.values[-self.model.sequence_length:]
            last_sequence = last_sequence.reshape(1, *last_sequence.shape)
            
            # Make prediction
            probability = self.model.predict_price_movement(last_sequence)
            confidence = self.model.get_model_confidence()
            
            prediction_result = {
                'symbol': symbol,
                'probability': float(probability),
                'confidence': float(confidence),
                'prediction': 'UP' if probability > 0.5 else 'DOWN',
                'strength': 'HIGH' if abs(probability - 0.5) > 0.2 else 'MEDIUM' if abs(probability - 0.5) > 0.1 else 'LOW',
                'timestamp': datetime.now().isoformat()
            }
            
            return prediction_result
            
        except Exception as e:
            self.logger.error(f"Error predicting price movement: {str(e)}")
            raise
    
    def batch_predict(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
        """
        Make batch predictions for multiple stocks.
        
        Args:
            data_dict: Dictionary with symbol as key and DataFrame as value
            
        Returns:
            Dictionary with predictions for each symbol
        """
        predictions = {}
        
        for symbol, data in data_dict.items():
            try:
                prediction = self.predict_price_movement_probability(data, symbol)
                predictions[symbol] = prediction
            except Exception as e:
                self.logger.error(f"Error predicting for {symbol}: {str(e)}")
                predictions[symbol] = {
                    'symbol': symbol,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
        
        return predictions
    
    def save_training_state(self, filepath: str) -> bool:
        """
        Save training state and scalers.
        
        Args:
            filepath: Path to save training state
            
        Returns:
            True if successful, False otherwise
        """
        try:
            training_state = {
                'is_fitted': self.is_fitted,
                'training_metrics': self.training_metrics,
                'best_model_path': self.best_model_path,
                'feature_scalers': {}
            }
            
            # Save scaler parameters
            for col, scaler in self.feature_scalers.items():
                training_state['feature_scalers'][col] = {
                    'min_': scaler.min_.tolist() if hasattr(scaler, 'min_') else None,
                    'scale_': scaler.scale_.tolist() if hasattr(scaler, 'scale_') else None,
                    'data_min_': scaler.data_min_.tolist() if hasattr(scaler, 'data_min_') else None,
                    'data_max_': scaler.data_max_.tolist() if hasattr(scaler, 'data_max_') else None,
                    'data_range_': scaler.data_range_.tolist() if hasattr(scaler, 'data_range_') else None
                }
            
            with open(filepath, 'w') as f:
                json.dump(training_state, f, indent=2)
            
            self.logger.info(f"Training state saved to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving training state: {str(e)}")
            return False
    
    def load_training_state(self, filepath: str) -> bool:
        """
        Load training state and scalers.
        
        Args:
            filepath: Path to load training state from
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(filepath, 'r') as f:
                training_state = json.load(f)
            
            self.is_fitted = training_state.get('is_fitted', False)
            self.training_metrics = training_state.get('training_metrics', {})
            self.best_model_path = training_state.get('best_model_path')
            
            # Restore scalers
            scaler_data = training_state.get('feature_scalers', {})
            for col, params in scaler_data.items():
                scaler = MinMaxScaler()
                if params.get('min_') is not None:
                    scaler.min_ = np.array(params['min_'])
                    scaler.scale_ = np.array(params['scale_'])
                    scaler.data_min_ = np.array(params['data_min_'])
                    scaler.data_max_ = np.array(params['data_max_'])
                    scaler.data_range_ = np.array(params['data_range_'])
                self.feature_scalers[col] = scaler
            
            self.logger.info(f"Training state loaded from {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading training state: {str(e)}")
            return False
    
    def get_training_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive training summary.
        
        Returns:
            Dictionary with training summary
        """
        if not self.training_metrics:
            return {'status': 'not_trained'}
        
        summary = {
            'status': 'trained' if self.is_fitted else 'not_fitted',
            'model_architecture': {
                'sequence_length': self.model.sequence_length,
                'features': self.model.features,
                'learning_rate': self.model.learning_rate
            },
            'training_results': self.training_metrics,
            'feature_scalers': list(self.feature_scalers.keys()),
            'best_model_path': self.best_model_path
        }
        
        return summary
        
    def train_models_for_all_stocks(self, symbols: List[str]) -> bool:
        """
        Train LSTM models for all stocks in the provided list using GPU acceleration.
        
        Args:
            symbols: List of stock symbols
            
        Returns:
            True if successful for at least one symbol, False otherwise
        """
        try:
            # Check if GPU is available
            gpus = tf.config.list_physical_devices('GPU')
            gpu_available = len(gpus) > 0
            
            if gpu_available:
                self.logger.info(f"Training LSTM models for {len(symbols)} stocks using GPU acceleration")
                # Configure GPU memory growth to avoid OOM errors
                for gpu in gpus:
                    try:
                        tf.config.experimental.set_memory_growth(gpu, True)
                        gpu_details = tf.config.experimental.get_device_details(gpu)
                        self.logger.info(f"Using GPU: {gpu_details.get('device_name', 'Unknown')}")
                    except RuntimeError as e:
                        self.logger.warning(f"Error configuring GPU: {str(e)}")
            else:
                self.logger.info(f"Training LSTM models for {len(symbols)} stocks using CPU (no GPU detected)")
            
            success_count = 0
            failed_count = 0
            results = {}
            
            # Get list of available processed files
            processed_files = []
            if os.path.exists("data/processed"):
                processed_files = [f for f in os.listdir("data/processed") if f.endswith("_processed.csv")]
            
            # Set batch size based on GPU availability
            batch_size = 96 if gpu_available else 32
            
            # Process symbols in batches to avoid memory issues
            for symbol in symbols:
                try:
                    # Clear TensorFlow session to free memory
                    tf.keras.backend.clear_session()
                    
                    # Try different naming conventions for the processed file
                    base_symbol = symbol.split('.')[0] if '.' in symbol else symbol
                    
                    # Check for file with exact symbol name
                    data_path = f"data/processed/{symbol.replace('.', '_')}_processed.csv"
                    
                    # If not found, try with base symbol (without .NS)
                    if not os.path.exists(data_path):
                        data_path = f"data/processed/{base_symbol}_processed.csv"
                    
                    # If still not found, look for any matching file
                    if not os.path.exists(data_path):
                        matching_files = [f for f in processed_files if f.startswith(f"{base_symbol}_")]
                        if matching_files:
                            data_path = f"data/processed/{matching_files[0]}"
                        else:
                            self.logger.warning(f"No processed data found for {symbol}")
                            continue
                    
                    # Load data
                    data = pd.read_csv(data_path)
                    if data.empty or len(data) < 60 + 50:  # Need enough data for training (sequence_length + buffer)
                        self.logger.warning(f"Insufficient data for {symbol}")
                        continue
                    
                    # Get all numeric columns for features
                    numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
                    
                    # Exclude index column if present
                    if 'Unnamed: 0' in numeric_cols:
                        numeric_cols.remove('Unnamed: 0')
                    
                    # Ensure we have enough features
                    if len(numeric_cols) < 3:  # Need at least 3 features
                        self.logger.warning(f"Insufficient features for {symbol}, need at least 3 numeric columns")
                        continue
                    
                    # Create a copy of the data with only numeric columns
                    processed_data = data.copy()
                    
                    # Add target column if not present
                    if 'target' not in processed_data.columns and 'close' in processed_data.columns:
                        processed_data['target'] = (processed_data['close'].shift(-1) > processed_data['close']).astype(int)
                        # Remove last row (no target available)
                        processed_data = processed_data[:-1]
                    
                    # Count actual features (excluding target)
                    feature_cols = [col for col in numeric_cols if col != 'target']
                    feature_count = len(feature_cols)
                    
                    self.logger.info(f"Using {feature_count} features for {symbol}: {feature_cols}")
                    
                    # Create a new model with the correct number of features
                    from src.models.lstm_model import LSTMModel
                    custom_model = LSTMModel(sequence_length=60, features=feature_count)
                    
                    # Train model directly using the model's train method with GPU-optimized batch size
                    self.logger.info(f"Training model for {symbol} using data from {data_path}...")
                    training_success = custom_model.train(processed_data, batch_size=batch_size)
                    
                    if not training_success:
                        self.logger.error(f"Training failed for {symbol}")
                        failed_count += 1
                        continue
                    
                    # Save model
                    model_dir = f"models/{base_symbol}"
                    os.makedirs(model_dir, exist_ok=True)
                    model_path = f"{model_dir}/lstm_model.keras"
                    
                    # Save model
                    if custom_model.save_model(model_path):
                        results[symbol] = {
                            'success': True,
                            'model_path': model_path
                        }
                        
                        success_count += 1
                        self.logger.info(f"Successfully trained model for {symbol}")
                    else:
                        self.logger.error(f"Failed to save model for {symbol}")
                        failed_count += 1
                    
                except Exception as e:
                    self.logger.error(f"Error training model for {symbol}: {str(e)}")
                    results[symbol] = {
                        'success': False,
                        'error': str(e)
                    }
                    failed_count += 1
            
            # Log summary
            self.logger.info(f"LSTM training completed: {success_count} successful, {failed_count} failed")
            
            # Update system state if at least one model was trained successfully
            if success_count > 0:
                try:
                    # Update system state file
                    system_state_path = "data/system_state.json"
                    if os.path.exists(system_state_path):
                        with open(system_state_path, 'r') as f:
                            system_state = json.load(f)
                        
                        system_state['system_state']['models_trained'] = True
                        system_state['timestamp'] = datetime.now().isoformat()
                        
                        with open(system_state_path, 'w') as f:
                            json.dump(system_state, f, indent=2)
                        
                        self.logger.info("Updated system state: models_trained = True")
                except Exception as e:
                    self.logger.error(f"Error updating system state: {str(e)}")
            
            return success_count > 0
                
        except Exception as e:
            self.logger.error(f"Error training LSTM models for stocks: {str(e)}")
            return False


class LSTMModelEvaluator(IModelEvaluator):
    """
    Model evaluator for LSTM performance assessment.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def evaluate_lstm_performance(self, model: LSTMModel, test_data: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate LSTM model performance on test data.
        
        Args:
            model: Trained LSTM model
            test_data: Test dataset
            
        Returns:
            Dictionary with performance metrics
        """
        if not model.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        try:
            # Prepare test data
            trainer = LSTMTrainer(model)
            X_test, _, y_test, _ = trainer.prepare_training_data(test_data, test_size=1.0)
            
            # Get predictions
            y_pred_prob = model.predict(test_data)
            y_pred = (y_pred_prob > 0.5).astype(int)
            
            # Calculate comprehensive metrics
            from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                                       f1_score, roc_auc_score, confusion_matrix)
            
            metrics = {
                'accuracy': float(accuracy_score(y_test, y_pred)),
                'precision': float(precision_score(y_test, y_pred, zero_division=0)),
                'recall': float(recall_score(y_test, y_pred, zero_division=0)),
                'f1_score': float(f1_score(y_test, y_pred, zero_division=0)),
                'auc_score': float(roc_auc_score(y_test, y_pred_prob)),
                'model_confidence': model.get_model_confidence()
            }
            
            # Add confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            metrics['confusion_matrix'] = cm.tolist()
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error evaluating LSTM performance: {str(e)}")
            raise
    
    def evaluate_dqn_performance(self, agent, test_data: pd.DataFrame) -> Dict[str, float]:
        """
        Placeholder for DQN evaluation (to be implemented in DQN module).
        
        Args:
            agent: DQN agent
            test_data: Test dataset
            
        Returns:
            Dictionary with performance metrics
        """
        return {'status': 'not_implemented'}