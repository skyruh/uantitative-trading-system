#!/usr/bin/env python3
"""
Script to test improved LSTM model architectures with GPU acceleration.
"""

import sys
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import time
from datetime import datetime
import json
import argparse
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Configure TensorFlow to use GPU
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        # Allow memory growth to avoid allocating all GPU memory at once
        for device in physical_devices:
            tf.config.exper
        print(f"Error configuring GPU: {str(e)}")
    except:
        pass
else:
    print("No GPU found. Using CPU for training.")

def load_data(symbol, data_dir="data/processed"):
    """Load processed data for a symbol."""
    # Try different naming conventions for the processed file
    base_symbol = symbol.split('.')[0] if '.' in symbol else symbol
    
    # Check for file with exact symbol name
    data_path = f"{data_dir}/{symbol.replace('.', '_')}_processed.csv"
    
    # If not found, try with base symbol (without .NS)
    if not os.path.exists(data_path):
        data_path = f"{data_dir}/{base_symbol}_processed.csv"
    
    # If still not found, return None
    if not os.path.exists(data_path):
        print(f"No processed data found for {symbol}")
        return None
    
    # Load data
    try:
        data = pd.read_csv(data_path)
        print(f"Loaded {len(data)} rows for {symbol} from {data_path}")
        return data
    except Exception as e:
        print(f"Error loading data for {symbol}: {str(e)}")
        return None

def prepare_sequences(data, sequence_length=60, test_split=0.2):
    """Prepare sequential data for training and testing."""
    # Use all available numeric columns as features
    numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
    
    # Exclude index column if present
    if 'Unnamed: 0' in numeric_cols:
        numeric_cols.remove('Unnamed: 0')
    
    # Check for sentiment features
    sentiment_features = [col for col in numeric_cols if col.startswith('sentiment_')]
    has_sentiment = len(sentiment_features) > 0
    
    if has_sentiment:
        print(f"Found sentiment features: {sentiment_features}")
    
    # Ensure target column is not included in features
    if 'target' in numeric_cols:
        numeric_cols.remove('target')
    
    # Create target if not present
    if 'target' not in data.columns and 'close' in data.columns:
        data = data.copy()
        data['target'] = (data['close'].shift(-1) > data['close']).astype(int)
        # Remove last row (no target available)
        data = data[:-1]
    
    # Prepare feature matrix
    feature_data = data[numeric_cols].values
    target_data = data['target'].values if 'target' in data.columns else np.zeros(len(data))
    
    # Create sequences
    X, y = [], []
    for i in range(sequence_length, len(feature_data)):
        X.append(feature_data[i-sequence_length:i])
        y.append(target_data[i])
    
    X = np.array(X)
    y = np.array(y)
    
    # Split into train and test sets
    split_idx = int(len(X) * (1 - test_split))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Prepared {len(X_train)} training sequences and {len(X_test)} test sequences")
    print(f"Feature shape: {X_train.shape}")
    
    return X_train, y_train, X_test, y_test, numeric_cols

def build_model(input_shape, use_gpu=True):
    """Build LSTM model optimized for GPU if available."""
    model = tf.keras.Sequential()
    
    # Use CuDNNLSTM if GPU is available for better performance
    if use_gpu and tf.config.list_physical_devices('GPU'):
        # Add LSTM layers with GPU optimization
        model.add(tf.keras.layers.LSTM(
            128, 
            input_shape=input_shape,
            return_sequences=True,
            activation='tanh',
            recurrent_activation='sigmoid',
            recurrent_dropout=0,
            unroll=False,
            use_bias=True
        ))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.LSTM(
            64,
            activation='tanh',
            recurrent_activation='sigmoid',
            recurrent_dropout=0,
            unroll=False,
            use_bias=True
        ))
    else:
        # Standard LSTM layers for CPU
        model.add(tf.keras.layers.LSTM(128, input_shape=input_shape, return_sequences=True))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.LSTM(64))
    
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model(symbol, epochs=10, batch_size=32, sequence_length=60, use_gpu=True):
    """Train LSTM model for a symbol."""
    # Load data
    data = load_data(symbol)
    if data is None:
        return None
    
    # Prepare sequences
    X_train, y_train, X_test, y_test, feature_cols = prepare_sequences(data, sequence_length)
    
    # Build model
    model = build_model((sequence_length, X_train.shape[2]), use_gpu)
    
    # Create model directory
    model_dir = f"models/{symbol.split('.')[0]}"
    os.makedirs(model_dir, exist_ok=True)
    
    # Create callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=f"{model_dir}/lstm_model_checkpoint.keras",
            save_best_only=True,
            monitor='val_accuracy',
            mode='max'
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
    ]
    
    # Train model
    print(f"Training model for {symbol} with {epochs} epochs and batch size {batch_size}")
    start_time = time.time()
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1
    )
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Evaluate model
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    # Save model
    model_path = f"{model_dir}/lstm_model.keras"
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    # Save model with sentiment features if available
    has_sentiment = any(col.startswith('sentiment_') for col in feature_cols)
    if has_sentiment:
        sentiment_model_path = f"{model_dir}/lstm_model_with_sentiment.keras"
        model.save(sentiment_model_path)
        print(f"Model with sentiment features saved to {sentiment_model_path}")
    
    # Save metadata
    metadata = {
        'symbol': symbol,
        'training_date': datetime.now().isoformat(),
        'epochs': epochs,
        'batch_size': batch_size,
        'sequence_length': sequence_length,
        'test_accuracy': float(test_accuracy),
        'training_time_seconds': training_time,
        'feature_columns': feature_cols,
        'has_sentiment_features': has_sentiment,
        'gpu_used': bool(tf.config.list_physical_devices('GPU')) and use_gpu
    }
    
    with open(f"{model_path}_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return {
        'model': model,
        'accuracy': test_accuracy,
        'training_time': training_time,
        'metadata': metadata
    }

def main():
    """Run LSTM model training test."""
    parser = argparse.ArgumentParser(description="Test LSTM model training with GPU acceleration")
    parser.add_argument("--symbol", type=str, default="RELIANCE.NS", help="Stock symbol to train model for")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--sequence-length", type=int, default=60, help="Sequence length for LSTM")
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU acceleration")
    args = parser.parse_args()
    
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    # Train model
    result = train_model(
        args.symbol,
        epochs=args.epochs,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        use_gpu=not args.no_gpu
    )
    
    if result:
        print(f"Training summary for {args.symbol}:")
        print(f"  Accuracy: {result['accuracy']:.4f}")
        print(f"  Training time: {result['training_time']:.2f} seconds")
        print(f"  GPU used: {result['metadata']['gpu_used']}")
        
        # Save summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'symbol': args.symbol,
            'accuracy': float(result['accuracy']),
            'training_time': float(result['training_time']),
            'gpu_used': result['metadata']['gpu_used'],
            'has_sentiment_features': result['metadata']['has_sentiment_features']
        }
        
        with open("lstm_training_test_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())