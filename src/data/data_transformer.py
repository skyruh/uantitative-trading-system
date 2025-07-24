"""
Data transformation utilities for handling different data formats.
"""

import pandas as pd
from typing import Optional
from src.utils.logging_utils import get_logger

logger = get_logger("DataTransformer")

def transform_yfinance_test_data(data_path: str) -> Optional[pd.DataFrame]:
    """
    Transform test data from yfinance test script format to the standard format
    expected by the DataProcessor.
    
    Args:
        data_path: Path to the CSV file containing test data
        
    Returns:
        Transformed DataFrame or None if transformation fails
    """
    try:
        # Read the CSV file
        df = pd.read_csv(data_path)
        
        # Check if this is the test data format (has 'Price', 'Ticker', 'Date' rows)
        if 'Price' in df.columns and 'Ticker' in df.iloc[0].values and 'Date' in df.iloc[1].values:
            logger.info(f"Detected test data format in {data_path}")
            
            # Extract the actual data (skip the first 3 rows which are headers)
            data = df.iloc[3:].copy()
            
            # Set the first column (Date) as the index
            data.set_index(data.columns[0], inplace=True)
            
            # Rename columns to match expected format (lowercase OHLCV)
            column_mapping = {
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume',
                'Price': 'close'  # Map Price to close if needed
            }
            
            # Apply column renaming for columns that exist
            rename_dict = {col: column_mapping[col] for col in data.columns if col in column_mapping}
            data.rename(columns=rename_dict, inplace=True)
            
            # Convert index to datetime
            data.index = pd.to_datetime(data.index)
            
            # Convert numeric columns to float
            for col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
            
            logger.info(f"Successfully transformed test data with shape {data.shape}")
            return data
        else:
            # This is not the test data format, return the original data
            logger.debug(f"Standard data format detected in {data_path}")
            return pd.read_csv(data_path, index_col=0, parse_dates=True)
            
    except Exception as e:
        logger.error(f"Error transforming test data: {str(e)}")
        return None