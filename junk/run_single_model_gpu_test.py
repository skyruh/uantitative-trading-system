#!/usr/bin/env python3
"""
Script to test GPU training with a single stock model.
"""

import sys
import os
import logging
import time
import tensorflow as tf
import numpy as np
import pandas as pd
from datetime import datetime

# Add src to Python path
sys.path.append('src')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("GPUTest")

def verify_gpu():
    """Verify that TensorFlow can see and use the GPU."""
    logger.info(f"TensorFlow version: {tf.__version__}")
    
    # Check for GPU devices
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        logger.info(f"Found {len(gpus)} GPU(s):")
        for i, gpu in enumerate(gpus):
            logger.info(f"  GPU {i}: {gpu.name}")
        
        # Configure memory growth
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info("Memory growth set to True for all GPUs")
        except Exception as e:
            logger.warning(f"Error setting memory growth: {e}")
        
        # Test GPU with a simple operation
        logger.info("Testing GPU with a simple matrix multiplication...")
        with tf.device('/GPU:0'):
            a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
            b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
            c = tf.matmul(a, b)
        
        logger.info(f"Result: {c.numpy()}")
        logger.info(f"Tensor device: {c.device}")
        
        if 'GPU' in c.device:
            logger.info("✓ GPU is working correctly!")
            return True
        else:
            logger.warning("✗ Operation executed on CPU despite GPU being available")
            return False
    else:
        logger.warning("No GPU found. TensorFlow will use CPU.")
        return False

def build_model(sequence_length, features):
    """Build LSTM model."""
    model = tf.keras.Sequential([
        # First LSTM layer with return sequences for stacking
        tf.keras.layers.LSTM(128, return_sequences=True, input_shape=(sequence_length, features),
                 name='lstm_layer_1'),
        tf.keras.layers.Dropout(0.3, name='dropout_1'),
        
        # Second LSTM layer
        tf.keras.layers.LSTM(64, return_sequences=False, name='lstm_layer_2'),
        tf.keras.layers.Dropout(0.3, name='dropout_2'),
        
        # Dense output layer for binary classification (sigmoid activation)
        tf.keras.layers.Dense(1, activation='sigmoid', name='output_layer')
    ])
    
    # Configure Adam optimizer with learning rate scheduling
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    # Compile model
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def prepare_sequences(data, sequence_length=60):
    """Prepare sequential data for LSTM training."""
    # Use all available numeric columns as features
    numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
    
    # Exclude index column if present
    if 'Unnamed: 0' in numeric_cols:
        numeric_cols.remove('Unnamed: 0')
    
    # Ensure target column is not included in features
    if 'target' in numeric_cols:
        numeric_cols.remove('target')
    
    # Prepare feature matrix
    feature_data = data[numeric_cols].values
    target_data = data['target'].values
    
    # Create sequences
    X, y = [], []
    for i in range(sequence_length, len(feature_data)):
        X.append(feature_data[i-sequence_length:i])
        y.append(target_data[i])
    
    return np.array(X), np.array(y), len(numeric_cols)

def main():
    """Test GPU training with a single stock model."""
    try:
        logger.info("Starting GPU test with a single stock model...")
        
        # Verify GPU
        gpu_working = verify_gpu()
        
        # Find a stock with sufficient data
        data_dir = "data/processed"
        processed_files = [f for f in os.listdir(data_dir) if f.endswith("_processed.csv")]
        
        if not processed_files:
            logger.error("No processed data files found")
            return 1
        
        # Use RELIANCE as test stock if available, otherwise use the first file
        test_file = None
        for file in processed_files:
            if file.startswith("RELIANCE_"):
                test_file = file
                break
        
        if not test_file:
            test_file = processed_files[0]
        
        logger.info(f"Using {test_file} for GPU test")
        
        # Load data
        data_path = os.path.join(data_dir, test_file)
        data = pd.read_csv(data_path)
        
        # Prepare data
        sequence_length = 60
        X, y, feature_count = prepare_sequences(data, sequence_length)
        
        logger.info(f"Prepared {len(X)} sequences with {feature_count} features")
        
        # Build model
        model = build_model(sequence_length, feature_count)
        
        # Configure callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            )
        ]
        
        # Train model
        logger.info("Starting model training...")
        start_time = time.time()
        
        history = model.fit(
            X, y,
            epochs=10,
            batch_size=64,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=1
        )
        
        end_time = time.time()
        training_time = end_time - start_time
        
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        # Calculate training speed
        epochs_trained = len(history.history['loss'])
        sequences_per_second = (len(X) * epochs_trained) / training_time
        
        logger.info(f"Training speed: {sequences_per_second:.2f} sequences/second")
        logger.info(f"Completed {epochs_trained} epochs")
        
        # Log final metrics
        final_loss = history.history['loss'][-1]
        final_accuracy = history.history['accuracy'][-1]
        
        logger.info(f"Final loss: {final_loss:.4f}")
        logger.info(f"Final accuracy: {final_accuracy:.4f}")
        
        # Save model for verification
        model_dir = "models/gpu_test"
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "test_model.keras")
        model.save(model_path)
        
        logger.info(f"Test model saved to {model_path}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error during GPU test: {str(e)}")
        logger.exception("Full traceback:")
        return 1

if __name__ == "__main__":
    sys.exit(main())