#!/usr/bin/env python3
"""
System analysis script for the quantitative trading system.
Analyzes data quality, model performance, and system health.
"""

import sys
import os
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import argparse
from pathlib import Path
import logging

# Add src to Python path
sys.path.append('src')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SystemAnalysis")

def analyze_data_quality(data_dir="data/processed", output_dir="results/analysis"):
    """Analyze the quality of processed data."""
    logger.info("Analyzing data quality...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of processed data files
    data_files = [f for f in os.listdir(data_dir) if f.endswith('_processed.csv')]
    
    if not data_files:
        logger.warning(f"No processed data files found in {data_dir}")
        return None
    
    logger.info(f"Found {len(data_files)} processed data files")
    
    # Initialize results
    results = {
        'total_files': len(data_files),
        'files_with_sentiment': 0,
        'files_with_missing_values': 0,
        'avg_file_size': 0,
        'avg_row_count': 0,
        'date_range_summary': {},
        'sentiment_coverage': {},
        'file_details': []
    }
    
    total_size = 0
    total_rows = 0
    min_date = None
    max_date = None
    
    # Analyze each file
    for file_name in data_files:
        file_path = os.path.join(data_dir, file_name)
        symbol = file_name.replace('_processed.csv', '')
        
        try:
            # Get file size
            file_size = os.path.getsize(file_path)
            total_size += file_size
            
            # Load data
            data = pd.read_csv(file_path)
            total_rows += len(data)
            
            # Check for sentiment columns
            has_sentiment = any(col.startswith('sentiment_') for col in data.columns)
            if has_sentiment:
                results['files_with_sentiment'] += 1
            
            # Check for missing values
            missing_values = data.isnull().sum().sum()
            has_missing = missing_values > 0
            if has_missing:
                results['files_with_missing_values'] += 1
            
            # Get date range
            if 'date' in data.columns:
                dates = pd.to_datetime(data['date'], errors='coerce', utc=True)
                # Filter out NaT values
                valid_dates = dates.dropna()
                if not valid_dates.empty:
                    file_min_date = valid_dates.min()
                    file_max_date = valid_dates.max()
                    
                    if min_date is None or file_min_date < min_date:
                        min_date = file_min_date
                    
                    if max_date is None or file_max_date > max_date:
                        max_date = file_max_date
            
            # Calculate sentiment coverage if available
            sentiment_coverage = 0
            if has_sentiment:
                non_zero_sentiment = (data['sentiment_score'] != 0).sum()
                sentiment_coverage = non_zero_sentiment / len(data) if len(data) > 0 else 0
            
            # Add file details
            date_range = "N/A"
            if 'date' in data.columns and 'file_min_date' in locals() and 'file_max_date' in locals():
                try:
                    date_range = f"{file_min_date.strftime('%Y-%m-%d')} to {file_max_date.strftime('%Y-%m-%d')}"
                except:
                    date_range = "Invalid date format"
            
            results['file_details'].append({
                'symbol': symbol,
                'file_size': file_size,
                'row_count': len(data),
                'column_count': len(data.columns),
                'has_sentiment': has_sentiment,
                'missing_values': int(missing_values),
                'date_range': date_range,
                'sentiment_coverage': sentiment_coverage
            })
            
        except Exception as e:
            logger.error(f"Error analyzing {file_name}: {str(e)}")
            results['file_details'].append({
                'symbol': symbol,
                'error': str(e)
            })
    
    # Calculate averages
    if results['total_files'] > 0:
        results['avg_file_size'] = total_size / results['total_files']
        results['avg_row_count'] = total_rows / results['total_files']
    
    # Add date range summary
    if min_date and max_date:
        results['date_range_summary'] = {
            'min_date': min_date.strftime('%Y-%m-%d'),
            'max_date': max_date.strftime('%Y-%m-%d'),
            'total_days': (max_date - min_date).days
        }
    
    # Calculate sentiment coverage statistics
    sentiment_coverages = [file['sentiment_coverage'] for file in results['file_details'] 
                          if 'sentiment_coverage' in file and file['sentiment_coverage'] > 0]
    
    if sentiment_coverages:
        results['sentiment_coverage'] = {
            'min': min(sentiment_coverages),
            'max': max(sentiment_coverages),
            'mean': np.mean(sentiment_coverages),
            'median': np.median(sentiment_coverages)
        }
    
    # Save results to JSON
    with open(f"{output_dir}/data_quality_analysis.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Generate visualizations
    if results['file_details']:
        # Create DataFrame from file details
        df = pd.DataFrame(results['file_details'])
        
        # Plot sentiment coverage distribution
        if 'sentiment_coverage' in df.columns and df['sentiment_coverage'].sum() > 0:
            plt.figure(figsize=(10, 6))
            sns.histplot(df['sentiment_coverage'], bins=20)
            plt.title('Sentiment Coverage Distribution')
            plt.xlabel('Sentiment Coverage (% of rows with non-zero sentiment)')
            plt.ylabel('Number of Stocks')
            plt.savefig(f"{output_dir}/sentiment_coverage_distribution.png")
            plt.close()
        
        # Plot row count distribution
        if 'row_count' in df.columns:
            plt.figure(figsize=(10, 6))
            sns.histplot(df['row_count'], bins=20)
            plt.title('Data Size Distribution')
            plt.xlabel('Number of Rows')
            plt.ylabel('Number of Stocks')
            plt.savefig(f"{output_dir}/row_count_distribution.png")
            plt.close()
    
    logger.info(f"Data quality analysis completed and saved to {output_dir}/data_quality_analysis.json")
    return results

def analyze_model_performance(models_dir="models", results_dir="results/backtest", output_dir="results/analysis"):
    """Analyze the performance of trained models."""
    logger.info("Analyzing model performance...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of model directories
    model_dirs = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d)) and d != 'checkpoints']
    
    if not model_dirs:
        logger.warning(f"No model directories found in {models_dir}")
        return None
    
    logger.info(f"Found {len(model_dirs)} model directories")
    
    # Initialize results
    results = {
        'total_models': len(model_dirs),
        'models_with_sentiment': 0,
        'models_with_backtest': 0,
        'performance_summary': {},
        'model_details': []
    }
    
    # Check for backtest results
    if os.path.exists(os.path.join(results_dir, 'evaluation_summary.json')):
        try:
            with open(os.path.join(results_dir, 'evaluation_summary.json'), 'r') as f:
                backtest_summary = json.load(f)
            results['backtest_summary'] = backtest_summary
        except Exception as e:
            logger.error(f"Error loading backtest summary: {str(e)}")
    
    # Check for performance comparison
    if os.path.exists(os.path.join(results_dir, 'performance_comparison.csv')):
        try:
            performance_df = pd.read_csv(os.path.join(results_dir, 'performance_comparison.csv'))
            
            # Calculate performance statistics
            if not performance_df.empty:
                results['performance_summary'] = {
                    'avg_base_return': float(performance_df['base_return'].mean()),
                    'avg_sentiment_return': float(performance_df['sentiment_return'].mean()),
                    'avg_return_improvement': float(performance_df['return_diff'].mean()),
                    'median_return_improvement': float(performance_df['return_diff'].median()),
                    'positive_improvements': int((performance_df['return_diff'] > 0).sum()),
                    'negative_improvements': int((performance_df['return_diff'] <= 0).sum()),
                    'avg_sharpe_improvement': float(performance_df['sharpe_diff'].mean()),
                    'avg_drawdown_improvement': float(performance_df['drawdown_diff'].mean())
                }
                
                # Generate visualizations
                plt.figure(figsize=(12, 6))
                sns.histplot(performance_df['return_diff'], bins=20)
                plt.axvline(x=0, color='r', linestyle='--')
                plt.title('Return Improvement Distribution (Sentiment vs Base)')
                plt.xlabel('Return Difference')
                plt.ylabel('Number of Stocks')
                plt.savefig(f"{output_dir}/return_improvement_distribution.png")
                plt.close()
                
                # Scatter plot of base vs sentiment returns
                plt.figure(figsize=(10, 10))
                plt.scatter(performance_df['base_return'], performance_df['sentiment_return'])
                plt.plot([-1, 1], [-1, 1], 'r--')  # Diagonal line
                plt.title('Base Model vs Sentiment Model Returns')
                plt.xlabel('Base Model Return')
                plt.ylabel('Sentiment Model Return')
                plt.grid(True)
                plt.savefig(f"{output_dir}/base_vs_sentiment_returns.png")
                plt.close()
                
                # Top 10 improved stocks
                top_improved = performance_df.sort_values('return_diff', ascending=False).head(10)
                plt.figure(figsize=(12, 8))
                sns.barplot(x='symbol', y='return_diff', data=top_improved)
                plt.title('Top 10 Stocks by Return Improvement')
                plt.xlabel('Stock Symbol')
                plt.ylabel('Return Improvement')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(f"{output_dir}/top_improved_stocks.png")
                plt.close()
                
                # Save performance comparison to output directory
                performance_df.to_csv(f"{output_dir}/model_performance_comparison.csv", index=False)
                
        except Exception as e:
            logger.error(f"Error analyzing performance comparison: {str(e)}")
    
    # Analyze individual model directories
    for model_dir in model_dirs:
        dir_path = os.path.join(models_dir, model_dir)
        
        try:
            # Check for model files
            has_base_model = os.path.exists(os.path.join(dir_path, 'lstm_model.keras'))
            has_sentiment_model = os.path.exists(os.path.join(dir_path, 'lstm_model_with_sentiment.keras'))
            
            if has_sentiment_model:
                results['models_with_sentiment'] += 1
            
            # Check for evaluation results
            eval_file = os.path.join(results_dir, f"{model_dir}.NS_evaluation_results.json")
            has_evaluation = os.path.exists(eval_file)
            
            if has_evaluation:
                results['models_with_backtest'] += 1
                
                # Load evaluation results
                with open(eval_file, 'r') as f:
                    eval_results = json.load(f)
                
                # Add to model details
                results['model_details'].append({
                    'symbol': model_dir,
                    'has_base_model': has_base_model,
                    'has_sentiment_model': has_sentiment_model,
                    'has_evaluation': True,
                    'evaluation_summary': eval_results
                })
            else:
                # Add basic info without evaluation
                results['model_details'].append({
                    'symbol': model_dir,
                    'has_base_model': has_base_model,
                    'has_sentiment_model': has_sentiment_model,
                    'has_evaluation': False
                })
                
        except Exception as e:
            logger.error(f"Error analyzing model directory {model_dir}: {str(e)}")
            results['model_details'].append({
                'symbol': model_dir,
                'error': str(e)
            })
    
    # Save results to JSON
    with open(f"{output_dir}/model_performance_analysis.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Model performance analysis completed and saved to {output_dir}/model_performance_analysis.json")
    return results

def analyze_system_health(logs_dir="logs", output_dir="results/analysis"):
    """Analyze system health based on logs and resource usage."""
    logger.info("Analyzing system health...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize results
    results = {
        'timestamp': datetime.now().isoformat(),
        'log_analysis': {},
        'resource_usage': {},
        'system_status': 'healthy'
    }
    
    # Check if logs directory exists
    if not os.path.exists(logs_dir):
        logger.warning(f"Logs directory {logs_dir} not found")
        results['log_analysis']['status'] = 'logs_not_found'
        results['system_status'] = 'warning'
    else:
        # Get list of log files
        log_files = [f for f in os.listdir(logs_dir) if f.endswith('.log')]
        
        if not log_files:
            logger.warning(f"No log files found in {logs_dir}")
            results['log_analysis']['status'] = 'no_logs_found'
            results['system_status'] = 'warning'
        else:
            # Analyze log files
            error_count = 0
            warning_count = 0
            info_count = 0
            
            for log_file in log_files:
                try:
                    with open(os.path.join(logs_dir, log_file), 'r') as f:
                        for line in f:
                            if ' ERROR ' in line:
                                error_count += 1
                            elif ' WARNING ' in line:
                                warning_count += 1
                            elif ' INFO ' in line:
                                info_count += 1
                except Exception as e:
                    logger.error(f"Error reading log file {log_file}: {str(e)}")
            
            results['log_analysis'] = {
                'log_files': len(log_files),
                'error_count': error_count,
                'warning_count': warning_count,
                'info_count': info_count
            }
            
            # Set system status based on error count
            if error_count > 100:
                results['system_status'] = 'critical'
            elif error_count > 10:
                results['system_status'] = 'warning'
    
    # Check GPU status
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        
        if gpus:
            results['resource_usage']['gpu'] = {
                'available_gpus': len(gpus),
                'gpu_names': [gpu.name for gpu in gpus],
                'tensorflow_gpu_enabled': True
            }
        else:
            results['resource_usage']['gpu'] = {
                'available_gpus': 0,
                'tensorflow_gpu_enabled': False
            }
            logger.warning("No GPUs detected by TensorFlow")
    except Exception as e:
        logger.error(f"Error checking GPU status: {str(e)}")
        results['resource_usage']['gpu'] = {
            'error': str(e)
        }
    
    # Check disk space
    try:
        import shutil
        total, used, free = shutil.disk_usage("/")
        
        results['resource_usage']['disk'] = {
            'total_gb': total // (2**30),
            'used_gb': used // (2**30),
            'free_gb': free // (2**30),
            'usage_percent': (used / total) * 100
        }
        
        # Set system status based on disk space
        if results['resource_usage']['disk']['usage_percent'] > 90:
            results['system_status'] = 'critical'
        elif results['resource_usage']['disk']['usage_percent'] > 80:
            results['system_status'] = 'warning'
    except Exception as e:
        logger.error(f"Error checking disk space: {str(e)}")
        results['resource_usage']['disk'] = {
            'error': str(e)
        }
    
    # Save results to JSON
    with open(f"{output_dir}/system_health_analysis.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"System health analysis completed and saved to {output_dir}/system_health_analysis.json")
    return results

def main():
    """Run system analysis."""
    parser = argparse.ArgumentParser(description="Analyze quantitative trading system")
    parser.add_argument("--output-dir", type=str, default="results/analysis", help="Output directory for results")
    parser.add_argument("--data-dir", type=str, default="data/processed", help="Directory with processed data")
    parser.add_argument("--models-dir", type=str, default="models", help="Directory with trained models")
    parser.add_argument("--results-dir", type=str, default="results/backtest", help="Directory with backtest results")
    parser.add_argument("--logs-dir", type=str, default="logs", help="Directory with system logs")
    parser.add_argument("--skip-data", action="store_true", help="Skip data quality analysis")
    parser.add_argument("--skip-models", action="store_true", help="Skip model performance analysis")
    parser.add_argument("--skip-health", action="store_true", help="Skip system health analysis")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run analyses
    data_results = None
    model_results = None
    health_results = None
    
    if not args.skip_data:
        data_results = analyze_data_quality(args.data_dir, args.output_dir)
    
    if not args.skip_models:
        model_results = analyze_model_performance(args.models_dir, args.results_dir, args.output_dir)
    
    if not args.skip_health:
        health_results = analyze_system_health(args.logs_dir, args.output_dir)
    
    # Generate summary report
    summary = {
        'timestamp': datetime.now().isoformat(),
        'data_quality': data_results is not None,
        'model_performance': model_results is not None,
        'system_health': health_results is not None
    }
    
    # Add key metrics to summary
    if data_results:
        summary['data_summary'] = {
            'total_files': data_results['total_files'],
            'files_with_sentiment': data_results['files_with_sentiment'],
            'sentiment_coverage': data_results.get('sentiment_coverage', {})
        }
    
    if model_results:
        summary['model_summary'] = {
            'total_models': model_results['total_models'],
            'models_with_sentiment': model_results['models_with_sentiment'],
            'performance_summary': model_results.get('performance_summary', {})
        }
    
    if health_results:
        summary['health_summary'] = {
            'system_status': health_results['system_status'],
            'error_count': health_results.get('log_analysis', {}).get('error_count', 0)
        }
    
    # Save summary to JSON
    with open(f"{args.output_dir}/system_analysis_summary.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    # Print summary
    print("\nSystem Analysis Summary:")
    print(f"Timestamp: {summary['timestamp']}")
    
    if data_results:
        print("\nData Quality:")
        print(f"  Total files: {data_results['total_files']}")
        print(f"  Files with sentiment: {data_results['files_with_sentiment']}")
        
        if 'sentiment_coverage' in data_results and data_results['sentiment_coverage']:
            print(f"  Avg sentiment coverage: {data_results['sentiment_coverage'].get('mean', 0):.2%}")
    
    if model_results:
        print("\nModel Performance:")
        print(f"  Total models: {model_results['total_models']}")
        print(f"  Models with sentiment: {model_results['models_with_sentiment']}")
        
        if 'performance_summary' in model_results and model_results['performance_summary']:
            print(f"  Avg base return: {model_results['performance_summary'].get('avg_base_return', 0):.4f}")
            print(f"  Avg sentiment return: {model_results['performance_summary'].get('avg_sentiment_return', 0):.4f}")
            print(f"  Avg return improvement: {model_results['performance_summary'].get('avg_return_improvement', 0):.4f}")
            print(f"  Positive improvements: {model_results['performance_summary'].get('positive_improvements', 0)}")
            print(f"  Negative improvements: {model_results['performance_summary'].get('negative_improvements', 0)}")
    
    if health_results:
        print("\nSystem Health:")
        print(f"  Status: {health_results['system_status']}")
        
        if 'log_analysis' in health_results:
            print(f"  Error count: {health_results['log_analysis'].get('error_count', 0)}")
            print(f"  Warning count: {health_results['log_analysis'].get('warning_count', 0)}")
        
        if 'resource_usage' in health_results and 'gpu' in health_results['resource_usage']:
            print(f"  GPUs available: {health_results['resource_usage']['gpu'].get('available_gpus', 0)}")
        
        if 'resource_usage' in health_results and 'disk' in health_results['resource_usage']:
            print(f"  Disk usage: {health_results['resource_usage']['disk'].get('usage_percent', 0):.1f}%")
    
    print(f"\nDetailed results saved to {args.output_dir}/")

if __name__ == "__main__":
    main()