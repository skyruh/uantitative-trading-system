"""
LSTM Model for price prediction in the quantitative trading system.
Implements a 2-layer LSTM with 50 units each and 0.2 dropout rate.
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from typing import Tuple, Optional, Dict, Any
import logging
from datetime import datetime

from src.interfaces.model_interfaces import ILSTMModel


class LSTMModel(ILSTMModel):
    """
    LSTM model for predicting next-day price movement probability.
    
    Architecture:
    - 2 LSTM layers with 50 units each
    - 0.2 dropout rate for regularization
    - Adam optimizer with learning rate scheduling
    - Dense output layer for binary classification (price up/down)
    """
    
    def __init__(self, sequence_length: int = 60, features: int = 10, 
                 learning_rate: float = 0.001, random_seed: int = 42):
        """
        Initialize LSTM model.
        
        Args:
            sequence_length: Number of time steps to look back
            features: Number of input features per time step
            learning_rate: Initial learning rate for Adam optimizer
            random_seed: Random seed for reproducibility
        """
        self.sequence_length = sequence_length
        self.features = features
        self.learning_rate = learning_rate
        self.random_seed = random_seed
        
        # Set random seeds for reproducibility
        tf.random.set_seed(random_seed)
        np.random.seed(random_seed)
        
        self.model = None
        self.is_trained = False
        self.training_history = None
        self.logger = logging.getLogger(__name__)
        
        # Build the model architecture
        self._build_model()
    
    def _build_model(self, feature_count: int = None) -> None:
        """
        Build the LSTM model architecture.
        
        Args:
            feature_count: Optional number of features to use. If provided, updates self.features.
        """
        try:
            # Update features if provided
            if feature_count is not None:
                self.features = feature_count
                self.logger.info(f"Updating model to use {self.features} features")
            
            # Check if GPU is available
            gpu_available = len(tf.config.list_physical_devices('GPU')) > 0
            
            # Create a new Sequential model with GPU optimizations
            self.model = Sequential([
                # First LSTM layer with return sequences for stacking
                # Use CuDNNLSTM implementation when on GPU for better performance
                LSTM(128, return_sequences=True, input_shape=(self.sequence_length, self.features),
                     name='lstm_layer_1', 
                     recurrent_activation='sigmoid',  # For GPU compatibility
                     implementation=2),  # Implementation 2 is more efficient on GPU
                Dropout(0.3, name='dropout_1'),
                
                # Second LSTM layer
                LSTM(64, return_sequences=False, name='lstm_layer_2',
                     recurrent_activation='sigmoid',  # For GPU compatibility
                     implementation=2),  # Implementation 2 is more efficient on GPU
                Dropout(0.3, name='dropout_2'),
                
                # Dense output layer for binary classification (sigmoid activation)
                Dense(1, activation='sigmoid', name='output_layer')
            ])
            
            # Configure Adam optimizer with learning rate scheduling
            optimizer = Adam(learning_rate=self.learning_rate)
            
            # Compile model with GPU-optimized settings
            self.model.compile(
                optimizer=optimizer,
                loss='binary_crossentropy',
                metrics=['accuracy'],
                # Use XLA compilation for faster GPU execution
                jit_compile=gpu_available
            )
            
            # Log device placement
            if gpu_available:
                self.logger.info("Model will run on GPU")
            else:
                self.logger.info("Model will run on CPU")
                
            self.logger.info(f"LSTM model architecture built successfully with {self.features} features")
            self.logger.info(f"Model summary: {self.model.count_params()} parameters")
            
        except Exception as e:
            self.logger.error(f"Error building LSTM model: {str(e)}")
            raise
    
    def get_model_summary(self) -> str:
        """Get model architecture summary."""
        if self.model is None:
            return "Model not built"
        
        import io
        stream = io.StringIO()
        self.model.summary(print_fn=lambda x: stream.write(x + '\n'))
        return stream.getvalue()
    
    def _prepare_sequences(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequential data for LSTM training.
        
        Args:
            data: DataFrame with features and target column
            
        Returns:
            Tuple of (X, y) arrays for training
        """
        try:
            # Use all available numeric columns as features
            numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
            
            # Exclude index column if present
            if 'Unnamed: 0' in numeric_cols:
                numeric_cols.remove('Unnamed: 0')
            
            # Ensure target column is not included in features
            if 'target' in numeric_cols:
                numeric_cols.remove('target')
            
            # Check for sentiment features
            sentiment_features = [col for col in numeric_cols if col.startswith('sentiment_')]
            if sentiment_features:
                self.logger.info(f"Including sentiment features: {sentiment_features}")
            
            # Check if we have enough features
            if len(numeric_cols) < 3:
                raise ValueError(f"Insufficient features. Need at least 3, got {len(numeric_cols)}")
            
            # Ensure we have the target column or create it
            if 'target' not in data.columns and 'close' in data.columns:
                data = data.copy()
                data['target'] = (data['close'].shift(-1) > data['close']).astype(int)
                # Remove last row (no target available)
                data = data[:-1]
            elif 'target' not in data.columns:
                raise ValueError("No 'close' column to create target variable")
            
            # Prepare feature matrix
            feature_data = data[numeric_cols].values
            target_data = data['target'].values
            
            # Create sequences
            X, y = [], []
            for i in range(self.sequence_length, len(feature_data)):
                X.append(feature_data[i-self.sequence_length:i])
                y.append(target_data[i])
            
            X = np.array(X)
            y = np.array(y)
            
            self.logger.info(f"Prepared {len(X)} sequences with shape {X.shape}")
            return X, y
            
        except Exception as e:
            self.logger.error(f"Error preparing sequences: {str(e)}")
            raise
    
    def train(self, data: pd.DataFrame, validation_split: float = 0.2, 
              epochs: int = 100, batch_size: int = 96) -> bool:
        """
        Train the LSTM model with GPU acceleration.
        
        Args:
            data: Training data with features and price information
            validation_split: Fraction of data to use for validation
            epochs: Maximum number of training epochs
            batch_size: Training batch size (default increased for GPU)
            
        Returns:
            True if training successful, False otherwise
        """
        try:
            # Check if GPU is available
            gpu_available = len(tf.config.list_physical_devices('GPU')) > 0
            if gpu_available:
                self.logger.info("Starting LSTM model training with GPU acceleration...")
            else:
                self.logger.info("Starting LSTM model training on CPU...")
            
            # Get numeric columns for features (excluding target)
            numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
            if 'Unnamed: 0' in numeric_cols:
                numeric_cols.remove('Unnamed: 0')
            if 'target' in numeric_cols:
                numeric_cols.remove('target')
                
            # Check if we need to rebuild the model with the correct number of features
            actual_feature_count = len(numeric_cols)
            if actual_feature_count != self.features:
                self.logger.info(f"Rebuilding model: expected {self.features} features, got {actual_feature_count}")
                self._build_model(feature_count=actual_feature_count)
            
            # Prepare sequential data
            X, y = self._prepare_sequences(data)
            
            if len(X) < 100:
                raise ValueError(f"Insufficient training data. Need at least 100 sequences, got {len(X)}")
            
            # Configure callbacks with GPU optimizations
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=15,
                    restore_best_weights=True,
                    verbose=1
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=7,
                    min_lr=1e-7,
                    verbose=1
                )
            ]
            
            # Add ModelCheckpoint callback for saving best model
            model_dir = "models/checkpoints"
            os.makedirs(model_dir, exist_ok=True)
            checkpoint_path = os.path.join(model_dir, f"lstm_checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}.keras")
            
            checkpoint_callback = ModelCheckpoint(
                filepath=checkpoint_path,
                save_best_only=True,
                monitor='val_loss',
                mode='min',
                verbose=1
            )
            callbacks.append(checkpoint_callback)
            
            # Train the model with GPU optimizations
            self.training_history = self.model.fit(
                X, y,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                callbacks=callbacks,
                verbose=1,
                shuffle=True  # Shuffle data for better training
            )
            
            self.is_trained = True
            
            # Log training metrics
            val_accuracy = max(self.training_history.history.get('val_accuracy', [0]))
            final_loss = self.training_history.history.get('loss', [0])[-1]
            epochs_trained = len(self.training_history.history.get('loss', []))
            
            self.logger.info(f"LSTM model training completed successfully in {epochs_trained} epochs")
            self.logger.info(f"Best validation accuracy: {val_accuracy:.4f}, Final loss: {final_loss:.4f}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error training LSTM model: {str(e)}")
            return False
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            data: Input data for prediction
            
        Returns:
            Array of predictions
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        try:
            X, _ = self._prepare_sequences(data)
            predictions = self.model.predict(X, verbose=0)
            return predictions.flatten()
            
        except Exception as e:
            self.logger.error(f"Error making predictions: {str(e)}")
            raise
    
    def predict_price_movement(self, sequence_data: np.ndarray) -> float:
        """
        Predict next-day price movement probability.
        
        Args:
            sequence_data: Sequential input data of shape (sequence_length, features)
            
        Returns:
            Probability of price increase (0-1)
        """
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        try:
            # Ensure correct shape
            if sequence_data.ndim == 2:
                sequence_data = sequence_data.reshape(1, *sequence_data.shape)
            
            prediction = self.model.predict(sequence_data, verbose=0)
            return float(prediction[0][0])
            
        except Exception as e:
            self.logger.error(f"Error predicting price movement: {str(e)}")
            raise
    
    def get_model_confidence(self) -> float:
        """
        Get model prediction confidence score based on training history.
        
        Returns:
            Confidence score (0-1)
        """
        if not self.is_trained or self.training_history is None:
            return 0.0
        
        try:
            # Use validation accuracy as confidence metric
            val_accuracy = self.training_history.history.get('val_accuracy', [0])
            return float(max(val_accuracy)) if val_accuracy else 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating model confidence: {str(e)}")
            return 0.0
    
    def save_model(self, path: str) -> bool:
        """
        Save the trained model to disk.
        
        Args:
            path: Path to save the model
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_trained or self.model is None:
            self.logger.error("Cannot save untrained model")
            return False
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Save model
            self.model.save(path)
            
            # Save additional metadata
            metadata = {
                'sequence_length': self.sequence_length,
                'features': self.features,
                'learning_rate': self.learning_rate,
                'is_trained': self.is_trained
            }
            
            import json
            with open(f"{path}_metadata.json", 'w') as f:
                json.dump(metadata, f)
            
            self.logger.info(f"Model saved successfully to {path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            return False
    
    def load_model(self, path: str) -> bool:
        """
        Load a trained model from disk.
        
        Args:
            path: Path to load the model from
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load model
            self.model = tf.keras.models.load_model(path)
            
            # Load metadata if available
            metadata_path = f"{path}_metadata.json"
            if os.path.exists(metadata_path):
                import json
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                self.sequence_length = metadata.get('sequence_length', self.sequence_length)
                self.features = metadata.get('features', self.features)
                self.learning_rate = metadata.get('learning_rate', self.learning_rate)
                self.is_trained = metadata.get('is_trained', True)
            else:
                self.is_trained = True
            
            self.logger.info(f"Model loaded successfully from {path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            return False
    
    def get_training_metrics(self) -> Dict[str, Any]:
        """
        Get training metrics and history.
        
        Returns:
            Dictionary with training metrics
        """
        if not self.training_history:
            return {}
        
        history = self.training_history.history
        return {
            'final_loss': history['loss'][-1] if 'loss' in history else None,
            'final_val_loss': history['val_loss'][-1] if 'val_loss' in history else None,
            'final_accuracy': history['accuracy'][-1] if 'accuracy' in history else None,
            'final_val_accuracy': history['val_accuracy'][-1] if 'val_accuracy' in history else None,
            'epochs_trained': len(history['loss']) if 'loss' in history else 0,
            'best_val_accuracy': max(history['val_accuracy']) if 'val_accuracy' in history else None
        }