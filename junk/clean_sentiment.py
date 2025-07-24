#!/usr/bin/env python3
"""
Script to remove sentiment analysis features and files from the system.
"""

import sys
import os
import pandas as pd
import json
import shutil
import argparse
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("CleanSentiment")

def remove_sentiment_features(file_path):
    """Remove sentiment features from a processed data file."""
    try:
        # Load the data
        data = pd.read_csv(file_path)
        
        # Check if file has sentiment features
        sentiment_cols = [col for col in data.columns if col.startswith('sentiment_')]
        
        if not sentiment_cols:
            logger.info(f"No sentiment features found in {file_path}")
            return False
        
        # Remove sentiment columns
        logger.info(f"Removing {len(sentiment_cols)} sentiment features from {file_path}")
        data = data.drop(columns=sentiment_cols)
        
        # Save the file
        data.to_csv(file_path, index=False)
        logger.info(f"Saved cleaned data to {file_path}")
        
        return True
    except Exception as e:
        logger.error(f"Error processing {file_path}: {str(e)}")
        return False

def clean_processed_data(data_dir="data/processed"):
    """Remove sentiment features from all processed data files."""
    logger.info(f"Cleaning sentiment features from processed data in {data_dir}")
    
    # Get list of processed data files
    if not os.path.exists(data_dir):
        logger.warning(f"Directory not found: {data_dir}")
        return 0
    
    files = [f for f in os.listdir(data_dir) if f.endswith('_processed.csv')]
    logger.info(f"Found {len(files)} processed data files")
    
    # Process each file
    cleaned_count = 0
    for file_name in files:
        file_path = os.path.join(data_dir, file_name)
        if remove_sentiment_features(file_path):
            cleaned_count += 1
    
    logger.info(f"Cleaned sentiment features from {cleaned_count} files")
    return cleaned_count

def delete_sentiment_files():
    """Delete sentiment-related files."""
    files_to_delete = [
        "enhanced_sentiment_processing.py",
        "run_enhanced_sentiment.bat",
        "sentiment_enhancement_plan.md",
        "test_alpha_vantage.py",
        "test_web_scraper.py",
        "test_yfinance.py",
        "generate_test_sentiment.py",
        "alpha_vantage_response.json"
    ]
    
    dirs_to_delete = [
        "data/cache/alpha_vantage",
        "data/cache/indian_news"
    ]
    
    # Delete files
    deleted_files = 0
    for file_path in files_to_delete:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info(f"Deleted file: {file_path}")
                deleted_files += 1
            except Exception as e:
                logger.error(f"Error deleting {file_path}: {str(e)}")
    
    # Delete directories
    deleted_dirs = 0
    for dir_path in dirs_to_delete:
        if os.path.exists(dir_path):
            try:
                shutil.rmtree(dir_path)
                logger.info(f"Deleted directory: {dir_path}")
                deleted_dirs += 1
            except Exception as e:
                logger.error(f"Error deleting {dir_path}: {str(e)}")
    
    # Delete news data files
    news_files = [f for f in os.listdir('.') if f.endswith('_news_data.json')]
    for file_path in news_files:
        try:
            os.remove(file_path)
            logger.info(f"Deleted file: {file_path}")
            deleted_files += 1
        except Exception as e:
            logger.error(f"Error deleting {file_path}: {str(e)}")
    
    # Delete sentiment summary files
    summary_files = [os.path.join("data/processed", f) for f in os.listdir("data/processed") 
                    if f.endswith('_sentiment_summary.json')]
    for file_path in summary_files:
        try:
            os.remove(file_path)
            logger.info(f"Deleted file: {file_path}")
            deleted_files += 1
        except Exception as e:
            logger.error(f"Error deleting {file_path}: {str(e)}")
    
    logger.info(f"Deleted {deleted_files} files and {deleted_dirs} directories")
    return deleted_files, deleted_dirs

def clean_models(models_dir="models"):
    """Remove sentiment models and keep only the base models."""
    logger.info(f"Cleaning models in {models_dir}")
    
    if not os.path.exists(models_dir):
        logger.warning(f"Directory not found: {models_dir}")
        return 0
    
    # Get list of model directories
    model_dirs = [d for d in os.listdir(models_dir) 
                 if os.path.isdir(os.path.join(models_dir, d)) and d != 'checkpoints']
    
    logger.info(f"Found {len(model_dirs)} model directories")
    
    # Process each model directory
    cleaned_count = 0
    for model_dir in model_dirs:
        dir_path = os.path.join(models_dir, model_dir)
        
        # Check for sentiment model
        sentiment_model_path = os.path.join(dir_path, "lstm_model_with_sentiment.keras")
        if os.path.exists(sentiment_model_path):
            try:
                os.remove(sentiment_model_path)
                logger.info(f"Deleted sentiment model: {sentiment_model_path}")
                
                # Also delete metadata
                metadata_path = f"{sentiment_model_path}_metadata.json"
                if os.path.exists(metadata_path):
                    os.remove(metadata_path)
                    logger.info(f"Deleted metadata: {metadata_path}")
                
                cleaned_count += 1
            except Exception as e:
                logger.error(f"Error deleting {sentiment_model_path}: {str(e)}")
        
        # Rename base model if needed
        base_model_path = os.path.join(dir_path, "lstm_model_without_sentiment.keras")
        if os.path.exists(base_model_path):
            try:
                # Rename to standard name
                standard_path = os.path.join(dir_path, "lstm_model.keras")
                if os.path.exists(standard_path):
                    os.remove(standard_path)
                    logger.info(f"Removed existing model: {standard_path}")
                
                os.rename(base_model_path, standard_path)
                logger.info(f"Renamed {base_model_path} to {standard_path}")
                
                # Also rename metadata
                base_metadata_path = f"{base_model_path}_metadata.json"
                if os.path.exists(base_metadata_path):
                    standard_metadata_path = f"{standard_path}_metadata.json"
                    if os.path.exists(standard_metadata_path):
                        os.remove(standard_metadata_path)
                    
                    os.rename(base_metadata_path, standard_metadata_path)
                    logger.info(f"Renamed {base_metadata_path} to {standard_metadata_path}")
                
                cleaned_count += 1
            except Exception as e:
                logger.error(f"Error renaming {base_model_path}: {str(e)}")
    
    logger.info(f"Cleaned {cleaned_count} model directories")
    return cleaned_count

def clean_source_code():
    """Remove sentiment-related code from source files."""
    files_to_clean = [
        "src/data/sentiment_analyzer.py",
        "src/data/sentiment_processor.py",
        "src/data/alpha_vantage_client.py",
        "src/data/indian_news_scraper.py"
    ]
    
    deleted_count = 0
    for file_path in files_to_clean:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info(f"Deleted source file: {file_path}")
                deleted_count += 1
            except Exception as e:
                logger.error(f"Error deleting {file_path}: {str(e)}")
    
    logger.info(f"Deleted {deleted_count} source files")
    return deleted_count

def main():
    """Clean up sentiment analysis from the system."""
    parser = argparse.ArgumentParser(description="Remove sentiment analysis from the system")
    parser.add_argument("--skip-data", action="store_true", help="Skip cleaning processed data")
    parser.add_argument("--skip-files", action="store_true", help="Skip deleting sentiment files")
    parser.add_argument("--skip-models", action="store_true", help="Skip cleaning models")
    parser.add_argument("--skip-source", action="store_true", help="Skip cleaning source code")
    args = parser.parse_args()
    
    logger.info("Starting sentiment analysis cleanup")
    
    # Clean processed data
    if not args.skip_data:
        cleaned_data_count = clean_processed_data()
    else:
        logger.info("Skipping processed data cleanup")
        cleaned_data_count = 0
    
    # Delete sentiment files
    if not args.skip_files:
        deleted_files, deleted_dirs = delete_sentiment_files()
    else:
        logger.info("Skipping sentiment files deletion")
        deleted_files, deleted_dirs = 0, 0
    
    # Clean models
    if not args.skip_models:
        cleaned_models_count = clean_models()
    else:
        logger.info("Skipping models cleanup")
        cleaned_models_count = 0
    
    # Clean source code
    if not args.skip_source:
        deleted_source_count = clean_source_code()
    else:
        logger.info("Skipping source code cleanup")
        deleted_source_count = 0
    
    # Print summary
    logger.info("\nCleanup Summary:")
    logger.info(f"Cleaned {cleaned_data_count} processed data files")
    logger.info(f"Deleted {deleted_files} files and {deleted_dirs} directories")
    logger.info(f"Cleaned {cleaned_models_count} model directories")
    logger.info(f"Deleted {deleted_source_count} source files")
    
    logger.info("\nSentiment analysis has been removed from the system")
    return 0

if __name__ == "__main__":
    sys.exit(main())