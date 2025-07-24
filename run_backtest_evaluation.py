#!/usr/bin/env python3
"""
Script to run a comprehensive backtest evaluation on trained LSTM models.
Compares models with and without sentiment features.
"""

import sys
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
import json
import argparse
from pathlib import Path

# Add src to Python path
sys.path.append('src')

from src.models.lstm_model import LSTMModel

def load_test_data(symbol):
    """Load test data for the given symbol."""
    # Try to find a test data file
    test_file = f"{symbol.split('.')[0]}_test_data.csv"
    if os.path.exists(test_file):
        return pd.read_csv(test_file)
    
    # If not found, use the processed data and split it
    processed_file = f"data/processed/{symbol.split('.')[0]}_processed.csv"
    if os.path.exists(processed_file):
        data = pd.read_csv(processed_file)
        # Use the last 20% as test data
        split_idx = int(len(data) * 0.8)
        return data.iloc[split_idx:]
    
    return None

def prepare_sequences(data, sequence_length=60, feature_columns=None, model_metadata=None):
    """Prepare sequential data for prediction."""
    # If model metadata is provided, use its feature columns
    if model_metadata and 'feature_columns' in model_metadata:
        # Use the exact feature columns from the model metadata
        feature_cols = model_metadata['feature_columns']
        
        # Check if all required features are available
        missing_cols = [col for col in feature_cols if col not in data.columns]
        if missing_cols:
            print(f"Warning: Missing columns in data: {missing_cols}")
            # Add missing columns with zeros
            for col in missing_cols:
                data[col] = 0.0
                
        numeric_cols = feature_cols
        print(f"Using {len(numeric_cols)} features from model metadata")
    # Use specified feature columns or all available numeric columns
    elif feature_columns:
        numeric_cols = [col for col in feature_columns if col in data.columns]
    else:
        numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
    
    # Exclude index column if present
    if 'Unnamed: 0' in numeric_cols:
        numeric_cols.remove('Unnamed: 0')
    
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
    
    return np.array(X), np.array(y), data.iloc[sequence_length:].reset_index(drop=True), numeric_cols

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance."""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    
    # Get predictions
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    # Calculate metrics
    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, zero_division=0)),
        'f1_score': float(f1_score(y_test, y_pred, zero_division=0)),
    }
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    return metrics, cm, y_pred, y_pred_prob

def run_trading_simulation(data, predictions, initial_capital=100000):
    """Run a simple trading simulation."""
    # Add predictions to data
    data = data.copy()
    data['prediction'] = predictions
    
    # Initialize portfolio
    portfolio = pd.DataFrame(index=data.index)
    portfolio['holdings'] = 0
    portfolio['cash'] = initial_capital
    portfolio['position'] = 0
    
    # Trading logic
    for i in range(1, len(data)):
        # Previous position
        prev_position = portfolio.loc[i-1, 'position']
        
        # Current prediction
        prediction = data.loc[i, 'prediction']
        
        # Current price
        price = data.loc[i, 'close']
        
        # Previous cash
        cash = portfolio.loc[i-1, 'cash']
        
        # Trading decision
        if prediction == 1 and prev_position <= 0:  # Buy signal
            # Calculate shares to buy (use 90% of cash)
            shares_to_buy = int((cash * 0.9) / price)
            
            # Update portfolio - explicitly convert to correct dtype
            portfolio.loc[i, 'position'] = int(shares_to_buy)
            portfolio.loc[i, 'cash'] = float(cash - (shares_to_buy * price))
            portfolio.loc[i, 'holdings'] = float(shares_to_buy * price)
            
        elif prediction == 0 and prev_position > 0:  # Sell signal
            # Sell all shares
            shares_to_sell = prev_position
            
            # Update portfolio - explicitly convert to correct dtype
            portfolio.loc[i, 'position'] = 0
            portfolio.loc[i, 'cash'] = float(cash + (shares_to_sell * price))
            portfolio.loc[i, 'holdings'] = 0.0
            
        else:  # Hold position
            portfolio.loc[i, 'position'] = prev_position
            portfolio.loc[i, 'cash'] = cash
            portfolio.loc[i, 'holdings'] = prev_position * price
    
    # Calculate portfolio value
    portfolio['value'] = portfolio['cash'] + portfolio['holdings']
    
    # Calculate returns
    portfolio['returns'] = portfolio['value'].pct_change()
    
    # Calculate metrics
    total_return = (portfolio['value'].iloc[-1] / initial_capital) - 1
    annual_return = total_return / (len(portfolio) / 252)  # Assuming 252 trading days per year
    
    # Calculate Sharpe ratio (assuming risk-free rate of 0)
    sharpe_ratio = np.sqrt(252) * portfolio['returns'].mean() / portfolio['returns'].std() if portfolio['returns'].std() > 0 else 0
    
    # Calculate max drawdown
    portfolio['cumulative_return'] = (1 + portfolio['returns']).cumprod()
    portfolio['cumulative_max'] = portfolio['cumulative_return'].cummax()
    portfolio['drawdown'] = (portfolio['cumulative_return'] / portfolio['cumulative_max']) - 1
    max_drawdown = portfolio['drawdown'].min()
    
    # Calculate win rate
    trades = portfolio['position'].diff().fillna(0) != 0
    num_trades = trades.sum()
    
    # Calculate profit factor
    winning_trades = portfolio[trades & (portfolio['returns'] > 0)]
    losing_trades = portfolio[trades & (portfolio['returns'] < 0)]
    
    profit_factor = abs(winning_trades['returns'].sum() / losing_trades['returns'].sum()) if losing_trades['returns'].sum() != 0 else float('inf')
    
    metrics = {
        'total_return': float(total_return),
        'annual_return': float(annual_return),
        'sharpe_ratio': float(sharpe_ratio),
        'max_drawdown': float(max_drawdown),
        'num_trades': int(num_trades),
        'profit_factor': float(profit_factor),
        'final_value': float(portfolio['value'].iloc[-1])
    }
    
    return metrics, portfolio

def plot_portfolio_performance(portfolio, symbol, model_type, output_dir="results/backtest"):
    """Plot portfolio performance."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot portfolio value
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio['value'])
    plt.title(f"Portfolio Value - {symbol} ({model_type})")
    plt.xlabel("Trading Days")
    plt.ylabel("Portfolio Value ($)")
    plt.grid(True)
    plt.savefig(f"{output_dir}/{symbol}_{model_type}_portfolio_value.png")
    plt.close()
    
    # Plot drawdown
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio['drawdown'])
    plt.title(f"Portfolio Drawdown - {symbol} ({model_type})")
    plt.xlabel("Trading Days")
    plt.ylabel("Drawdown (%)")
    plt.grid(True)
    plt.savefig(f"{output_dir}/{symbol}_{model_type}_drawdown.png")
    plt.close()
    
    # Plot cumulative returns
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio['cumulative_return'])
    plt.title(f"Cumulative Returns - {symbol} ({model_type})")
    plt.xlabel("Trading Days")
    plt.ylabel("Cumulative Returns")
    plt.grid(True)
    plt.savefig(f"{output_dir}/{symbol}_{model_type}_cumulative_returns.png")
    plt.close()

def compare_models(symbol, with_sentiment_metrics, without_sentiment_metrics, output_dir="results/backtest"):
    """Compare models with and without sentiment features."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create comparison table
    comparison = pd.DataFrame({
        'With Sentiment': with_sentiment_metrics,
        'Without Sentiment': without_sentiment_metrics
    })
    
    # Calculate improvement
    comparison['Improvement'] = comparison['With Sentiment'] - comparison['Without Sentiment']
    comparison['Improvement %'] = (comparison['Improvement'] / comparison['Without Sentiment'].abs()) * 100
    
    # Save comparison to CSV
    comparison.to_csv(f"{output_dir}/{symbol}_model_comparison.csv")
    
    # Plot comparison
    metrics_to_plot = ['total_return', 'sharpe_ratio', 'max_drawdown']
    
    for metric in metrics_to_plot:
        plt.figure(figsize=(10, 6))
        bars = plt.bar(['Without Sentiment', 'With Sentiment'], 
                      [without_sentiment_metrics[metric], with_sentiment_metrics[metric]])
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}',
                    ha='center', va='bottom')
        
        plt.title(f"{symbol} - {metric.replace('_', ' ').title()} Comparison")
        plt.ylabel(metric.replace('_', ' ').title())
        plt.grid(axis='y')
        plt.savefig(f"{output_dir}/{symbol}_{metric}_comparison.png")
        plt.close()
    
    return comparison

def evaluate_symbol(symbol, output_dir="results/backtest", initial_capital=100000):
    """Evaluate models for a single symbol."""
    print(f"\nEvaluating models for {symbol}...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Define model paths
    base_model_path = os.path.join('models', symbol.split('.')[0], 'lstm_model.keras')
    sentiment_model_path = os.path.join('models', symbol.split('.')[0], 'lstm_model_with_sentiment.keras')
    
    results = {
        'symbol': symbol,
        'has_base_model': os.path.exists(base_model_path),
        'has_sentiment_model': os.path.exists(sentiment_model_path),
        'base_model_metrics': None,
        'sentiment_model_metrics': None,
        'comparison': None
    }
    
    # Load test data
    test_data = load_test_data(symbol)
    
    if test_data is None:
        print(f"No test data found for {symbol}")
        return results
    
    # Check if test data has sentiment features
    has_sentiment_features = any(col.startswith('sentiment_') for col in test_data.columns)
    results['has_sentiment_features'] = has_sentiment_features
    
    # Evaluate base model (without sentiment)
    if os.path.exists(base_model_path):
        # Load model
        base_model = tf.keras.models.load_model(base_model_path)
        
        # Load metadata
        metadata_path = f"{base_model_path}_metadata.json"
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            sequence_length = metadata.get('sequence_length', 60)
            feature_columns = metadata.get('feature_columns', None)
        else:
            sequence_length = 60
            feature_columns = None
            metadata = None
        
        # Filter out sentiment features for base model
        if feature_columns:
            base_features = [col for col in feature_columns if not col.startswith('sentiment_')]
        else:
            base_features = [col for col in test_data.columns if not col.startswith('sentiment_')]
        
        # Prepare sequences without sentiment features
        X_test, y_test, test_df, used_features = prepare_sequences(
            test_data, sequence_length, base_features, metadata
        )
        
        # Evaluate model
        metrics, cm, y_pred, y_pred_prob = evaluate_model(base_model, X_test, y_test)
        
        print(f"Base model metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")
        
        print(f"Confusion matrix:")
        print(cm)
        
        # Run trading simulation
        trading_metrics, portfolio = run_trading_simulation(test_df, y_pred, initial_capital)
        
        print(f"Base model trading simulation results:")
        for key, value in trading_metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        
        # Plot portfolio performance
        plot_portfolio_performance(portfolio, symbol, "base_model", output_dir)
        
        # Store results
        results['base_model_metrics'] = {
            'model_metrics': metrics,
            'trading_metrics': trading_metrics,
            'features_used': used_features
        }
    
    # Evaluate sentiment model
    if os.path.exists(sentiment_model_path) and has_sentiment_features:
        # Load model
        sentiment_model = tf.keras.models.load_model(sentiment_model_path)
        
        # Load metadata
        metadata_path = f"{sentiment_model_path}_metadata.json"
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            sequence_length = metadata.get('sequence_length', 60)
            feature_columns = metadata.get('feature_columns', None)
        else:
            sequence_length = 60
            feature_columns = None
        
        # Prepare sequences with all features including sentiment
        X_test, y_test, test_df, used_features = prepare_sequences(
            test_data, sequence_length, feature_columns
        )
        
        # Evaluate model
        metrics, cm, y_pred, y_pred_prob = evaluate_model(sentiment_model, X_test, y_test)
        
        print(f"Sentiment model metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")
        
        print(f"Confusion matrix:")
        print(cm)
        
        # Run trading simulation
        trading_metrics, portfolio = run_trading_simulation(test_df, y_pred, initial_capital)
        
        print(f"Sentiment model trading simulation results:")
        for key, value in trading_metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        
        # Plot portfolio performance
        plot_portfolio_performance(portfolio, symbol, "sentiment_model", output_dir)
        
        # Store results
        results['sentiment_model_metrics'] = {
            'model_metrics': metrics,
            'trading_metrics': trading_metrics,
            'features_used': used_features
        }
    
    # Compare models if both are available
    if results['base_model_metrics'] and results['sentiment_model_metrics']:
        comparison = compare_models(
            symbol,
            results['sentiment_model_metrics']['trading_metrics'],
            results['base_model_metrics']['trading_metrics'],
            output_dir
        )
        results['comparison'] = comparison.to_dict()
        
        # Print comparison
        print("\nModel comparison:")
        print(comparison)
    
    # Save results to JSON
    with open(f"{output_dir}/{symbol}_evaluation_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    return results

def main():
    """Run backtest evaluation on trained models."""
    parser = argparse.ArgumentParser(description="Evaluate and compare LSTM models with and without sentiment features")
    parser.add_argument("--symbols", type=str, help="Comma-separated list of symbols to evaluate")
    parser.add_argument("--output-dir", type=str, default="results/backtest", help="Output directory for results")
    parser.add_argument("--capital", type=float, default=100000, help="Initial capital for trading simulation")
    parser.add_argument("--top", type=int, default=0, help="Evaluate only top N models by directory size")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get list of model directories
    model_dirs = [d for d in os.listdir('models') if os.path.isdir(os.path.join('models', d)) and d != 'checkpoints']
    
    print(f"Found {len(model_dirs)} model directories")
    
    # Sort by directory size if top N requested
    if args.top > 0:
        dir_sizes = []
        for d in model_dirs:
            dir_path = os.path.join('models', d)
            size = sum(os.path.getsize(os.path.join(dir_path, f)) for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f)))
            dir_sizes.append((d, size))
        
        # Sort by size (largest first)
        dir_sizes.sort(key=lambda x: x[1], reverse=True)
        model_dirs = [d[0] for d in dir_sizes[:args.top]]
        print(f"Selected top {len(model_dirs)} models by directory size")
    
    # Select specific symbols if provided
    if args.symbols:
        symbols = args.symbols.split(',')
        # Remove .NS suffix if present
        symbols = [s.split('.')[0] for s in symbols]
        # Filter model_dirs to only include specified symbols
        model_dirs = [d for d in model_dirs if d in symbols]
        print(f"Filtered to {len(model_dirs)} specified symbols")
    
    # Convert model directories to symbols
    symbols_to_evaluate = [f"{d}.NS" for d in model_dirs]
    
    # Evaluate each symbol
    all_results = {}
    
    for symbol in symbols_to_evaluate:
        results = evaluate_symbol(symbol, args.output_dir, args.capital)
        all_results[symbol] = results
    
    # Generate summary report
    summary = {
        'total_symbols': len(symbols_to_evaluate),
        'symbols_with_base_model': sum(1 for r in all_results.values() if r['has_base_model']),
        'symbols_with_sentiment_model': sum(1 for r in all_results.values() if r['has_sentiment_model']),
        'symbols_with_both_models': sum(1 for r in all_results.values() if r['has_base_model'] and r['has_sentiment_model']),
        'symbols_with_sentiment_features': sum(1 for r in all_results.values() if r.get('has_sentiment_features', False)),
        'timestamp': datetime.now().isoformat()
    }
    
    # Calculate average improvement
    improvements = []
    for symbol, result in all_results.items():
        if result['base_model_metrics'] and result['sentiment_model_metrics']:
            base_return = result['base_model_metrics']['trading_metrics']['total_return']
            sentiment_return = result['sentiment_model_metrics']['trading_metrics']['total_return']
            improvement = sentiment_return - base_return
            improvements.append(improvement)
    
    if improvements:
        summary['avg_return_improvement'] = float(np.mean(improvements))
        summary['median_return_improvement'] = float(np.median(improvements))
        summary['positive_improvement_count'] = sum(1 for imp in improvements if imp > 0)
        summary['negative_improvement_count'] = sum(1 for imp in improvements if imp < 0)
    
    # Save summary to JSON
    with open(f"{args.output_dir}/evaluation_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print("\nEvaluation Summary:")
    print(f"Total symbols evaluated: {summary['total_symbols']}")
    print(f"Symbols with base model: {summary['symbols_with_base_model']}")
    print(f"Symbols with sentiment model: {summary['symbols_with_sentiment_model']}")
    print(f"Symbols with both models: {summary['symbols_with_both_models']}")
    print(f"Symbols with sentiment features: {summary['symbols_with_sentiment_features']}")
    
    if 'avg_return_improvement' in summary:
        print(f"\nAverage return improvement: {summary['avg_return_improvement']:.4f}")
        print(f"Median return improvement: {summary['median_return_improvement']:.4f}")
        print(f"Symbols with positive improvement: {summary['positive_improvement_count']}")
        print(f"Symbols with negative improvement: {summary['negative_improvement_count']}")
    
    # Generate detailed performance table
    performance_data = []
    for symbol, result in all_results.items():
        if result['base_model_metrics'] and result['sentiment_model_metrics']:
            base_metrics = result['base_model_metrics']['trading_metrics']
            sentiment_metrics = result['sentiment_model_metrics']['trading_metrics']
            
            performance_data.append({
                'symbol': symbol,
                'base_return': base_metrics['total_return'],
                'sentiment_return': sentiment_metrics['total_return'],
                'return_diff': sentiment_metrics['total_return'] - base_metrics['total_return'],
                'base_sharpe': base_metrics['sharpe_ratio'],
                'sentiment_sharpe': sentiment_metrics['sharpe_ratio'],
                'sharpe_diff': sentiment_metrics['sharpe_ratio'] - base_metrics['sharpe_ratio'],
                'base_drawdown': base_metrics['max_drawdown'],
                'sentiment_drawdown': sentiment_metrics['max_drawdown'],
                'drawdown_diff': sentiment_metrics['max_drawdown'] - base_metrics['max_drawdown']
            })
    
    if performance_data:
        performance_df = pd.DataFrame(performance_data)
        performance_df.sort_values('return_diff', ascending=False, inplace=True)
        performance_df.to_csv(f"{args.output_dir}/performance_comparison.csv", index=False)
        
        print("\nTop 5 symbols by return improvement:")
        print(performance_df.head(5)[['symbol', 'base_return', 'sentiment_return', 'return_diff']])
        
        print("\nBottom 5 symbols by return improvement:")
        print(performance_df.tail(5)[['symbol', 'base_return', 'sentiment_return', 'return_diff']])

if __name__ == "__main__":
    main()