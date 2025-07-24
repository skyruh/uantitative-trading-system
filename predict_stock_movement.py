#!/usr/bin/env python3
"""
Script to demonstrate how to use trained LSTM models for stock price movement predictions.
"""

import sys
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import json
from datetime import datetime, timedelta

# Add src to Python path
sys.path.append('src')

def load_latest_data(symbol, lookback_days=60):
    """Load the latest data for the given symbol."""
    # Try to find processed data
    symbol_base = symbol.split('.')[0]
    processed_file = f"data/processed/{symbol_base}_processed.csv"
    
    if os.path.exists(processed_file):
        data = pd.read_csv(processed_file)
        # Get the latest data points
        return data.tail(lookback_days + 5)  # Add a few extra days as buffer
    
    return None

def prepare_prediction_data(data, sequence_length=60):
    """Prepare data for prediction."""
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
    
    # Get the latest sequence
    latest_sequence = feature_data[-sequence_length:]
    
    # Reshape for model input
    return np.array([latest_sequence])

def get_prediction_with_confidence(model, sequence_data):
    """Get prediction with confidence score."""
    # Make prediction
    prediction = model.predict(sequence_data, verbose=0)[0][0]
    
    # Calculate confidence based on distance from 0.5
    confidence = abs(prediction - 0.5) * 2  # Scale to 0-1
    
    return prediction, confidence

def format_prediction_result(symbol, prediction, confidence, latest_price, date):
    """Format prediction result for display."""
    direction = "UP" if prediction > 0.5 else "DOWN"
    probability = prediction if direction == "UP" else 1 - prediction
    
    confidence_level = "LOW"
    if confidence > 0.6:
        confidence_level = "HIGH"
    elif confidence > 0.3:
        confidence_level = "MEDIUM"
    
    result = {
        'symbol': symbol,
        'prediction_date': date,
        'direction': direction,
        'probability': float(probability),
        'confidence': float(confidence),
        'confidence_level': confidence_level,
        'latest_price': latest_price,
        'recommendation': "HOLD"
    }
    
    # Generate recommendation
    if direction == "UP" and confidence > 0.3:
        result['recommendation'] = "BUY"
    elif direction == "DOWN" and confidence > 0.3:
        result['recommendation'] = "SELL"
    
    return result

def main():
    """Demonstrate how to use trained models for predictions."""
    # Get list of model directories
    model_dirs = [d for d in os.listdir('models') if os.path.isdir(os.path.join('models', d)) and d != 'checkpoints']
    
    print(f"Found {len(model_dirs)} trained models")
    
    # Select a few models for demonstration
    models_to_demo = model_dirs[:5]  # First 5 models
    
    predictions = []
    
    for model_dir in models_to_demo:
        symbol = f"{model_dir}.NS"
        model_path = os.path.join('models', model_dir, 'lstm_model.keras')
        metadata_path = f"{model_path}_metadata.json"
        
        print(f"\nGenerating prediction for {symbol}...")
        
        if os.path.exists(model_path) and os.path.exists(metadata_path):
            # Load metadata
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            sequence_length = metadata.get('sequence_length', 60)
            
            # Load model
            model = tf.keras.models.load_model(model_path)
            
            # Load latest data
            latest_data = load_latest_data(symbol, sequence_length)
            
            if latest_data is not None:
                # Prepare data for prediction
                sequence_data = prepare_prediction_data(latest_data, sequence_length)
                
                # Get prediction
                prediction, confidence = get_prediction_with_confidence(model, sequence_data)
                
                # Get latest price
                latest_price = latest_data['close'].iloc[-1] if 'close' in latest_data.columns else 0
                
                # Get latest date
                latest_date = latest_data['date'].iloc[-1] if 'date' in latest_data.columns else datetime.now().strftime("%Y-%m-%d")
                
                # Format result
                result = format_prediction_result(symbol, prediction, confidence, latest_price, latest_date)
                
                # Add to predictions
                predictions.append(result)
                
                print(f"Prediction: {result['direction']} with {result['probability']:.4f} probability")
                print(f"Confidence: {result['confidence']:.4f} ({result['confidence_level']})")
                print(f"Recommendation: {result['recommendation']}")
            else:
                print(f"No data found for {symbol}")
        else:
            print(f"No model or metadata found for {symbol}")
    
    # Create a DataFrame for better display
    if predictions:
        df = pd.DataFrame(predictions)
        
        # Sort by confidence
        df = df.sort_values('confidence', ascending=False)
        
        print("\nPrediction Summary:")
        print(df[['symbol', 'direction', 'probability', 'confidence_level', 'recommendation']].to_string(index=False))
        
        # Count recommendations
        buy_count = len(df[df['recommendation'] == 'BUY'])
        sell_count = len(df[df['recommendation'] == 'SELL'])
        hold_count = len(df[df['recommendation'] == 'HOLD'])
        
        print(f"\nRecommendation Summary: {buy_count} BUY, {sell_count} SELL, {hold_count} HOLD")

if __name__ == "__main__":
    main()