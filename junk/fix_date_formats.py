#!/usr/bin/env python3
"""
Script to fix date format issues in processed data files.
This addresses the "NaTType does not support strftime" errors in system analysis.
"""

import os
import pandas as pd
import logging
import argparse
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("FixDateFormats")

# List of problematic files identified in the logs
KNOWN_PROBLEM_FILES = [
    "ACMESOLAR_processed.csv", "AFCONS_processed.csv", "ALIVUS_processed.csv",
    "BAJAJHFL_processed.csv", "COHANCE_processed.csv", "ETERNAL_processed.csv",
    "HBLENGINE_processed.csv", "HYUNDAI_processed.csv", "IGIL_processed.csv",
    "IKS_processed.csv", "METAL_processed.csv", "NIVABUPA_processed.csv",
    "NTPCGREEN_processed.csv", "SAGILITY_processed.csv", "SAILIFE_processed.csv",
    "SWIGGY_processed.csv", "VMM_processed.csv", "WAAREEENER_processed.csv"
]

def fix_date_format(file_path):
    """Fix date format issues in a CSV file."""
    try:
        # Read the file
        logger.info(f"Processing {file_path}")
        df = pd.read_csv(file_path)
        
        # Check if file has a date column
        date_col = None
        for col in ['date', 'Date']:
            if col in df.columns:
                date_col = col
                break
        
        if not date_col:
            logger.warning(f"No date column found in {file_path}")
            return False
        
        # Check if date column needs fixing
        try:
            # Try to convert to datetime to see if it works
            pd.to_datetime(df[date_col])
            logger.info(f"Date column in {file_path} is already valid")
            return True
        except Exception:
            # Date column needs fixing
            logger.info(f"Fixing date column in {file_path}")
            
            # Convert to datetime with errors='coerce' to handle invalid dates
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            
            # Drop rows with invalid dates
            original_count = len(df)
            df = df.dropna(subset=[date_col])
            dropped_count = original_count - len(df)
            
            if dropped_count > 0:
                logger.warning(f"Dropped {dropped_count} rows with invalid dates in {file_path}")
            
            # Save back to file
            df.to_csv(file_path, index=False)
            logger.info(f"Fixed {file_path}")
            return True
            
    except Exception as e:
        logger.error(f"Error fixing {file_path}: {str(e)}")
        return False

def fix_all_files(data_dir="data/processed", specific_files=None):
    """Fix date format issues in all processed data files."""
    # Create directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Get list of files to process
    if specific_files:
        files_to_process = [os.path.join(data_dir, f) for f in specific_files]
    else:
        # Process all CSV files in the directory
        files_to_process = [os.path.join(data_dir, f) for f in os.listdir(data_dir) 
                           if f.endswith('_processed.csv')]
    
    # Process each file
    success_count = 0
    failure_count = 0
    
    for file_path in files_to_process:
        if os.path.exists(file_path):
            if fix_date_format(file_path):
                success_count += 1
            else:
                failure_count += 1
        else:
            logger.warning(f"File not found: {file_path}")
            failure_count += 1
    
    logger.info(f"Processed {success_count + failure_count} files: {success_count} successful, {failure_count} failed")
    return success_count, failure_count

def main():
    """Run the date format fixing script."""
    parser = argparse.ArgumentParser(description="Fix date format issues in processed data files")
    parser.add_argument("--data-dir", type=str, default="data/processed", 
                        help="Directory containing processed data files")
    parser.add_argument("--known-only", action="store_true", 
                        help="Process only known problematic files")
    parser.add_argument("--files", type=str, 
                        help="Comma-separated list of specific files to process")
    args = parser.parse_args()
    
    logger.info("Starting date format fixing process...")
    
    # Determine which files to process
    specific_files = None
    if args.known_only:
        specific_files = KNOWN_PROBLEM_FILES
        logger.info(f"Processing {len(specific_files)} known problematic files")
    elif args.files:
        specific_files = args.files.split(',')
        logger.info(f"Processing {len(specific_files)} specified files")
    else:
        logger.info(f"Processing all CSV files in {args.data_dir}")
    
    # Fix files
    success_count, failure_count = fix_all_files(args.data_dir, specific_files)
    
    if failure_count == 0:
        logger.info("All files processed successfully")
        return 0
    else:
        logger.warning(f"{failure_count} files could not be processed")
        return 1

if __name__ == "__main__":
    exit(main())