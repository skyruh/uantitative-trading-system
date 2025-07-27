"""
Validation utilities for the quantitative trading system.
Provides data validation, configuration validation, and system health checks.
"""

import os
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import importlib.util

# Try to import optional dependencies
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

from src.utils.logging_utils import get_logger

logger = get_logger("Validation")


def validate_stock_data(data, symbol: str) -> Tuple[bool, List[str]]:
    """
    Validate stock data for completeness and consistency.
    
    Args:
        data: Stock data DataFrame (requires pandas)
        symbol: Stock symbol for logging
    
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []
    
    if not PANDAS_AVAILABLE:
        issues.append("pandas not available for data validation")
        return False, issues
    
    if not isinstance(data, pd.DataFrame):
        issues.append("Data is not a pandas DataFrame")
        return False, issues
    
    if data.empty:
        issues.append("DataFrame is empty")
        return False, issues
    
    # Check required columns - support both uppercase and lowercase column names
    required_columns_upper = ['Open', 'High', 'Low', 'Close', 'Volume']
    required_columns_lower = ['open', 'high', 'low', 'close', 'volume']
    
    # Check if we have uppercase or lowercase columns
    upper_columns_present = [col for col in required_columns_upper if col in data.columns]
    lower_columns_present = [col for col in required_columns_lower if col in data.columns]
    
    # Use whichever set has more matches
    if len(upper_columns_present) >= len(lower_columns_present):
        required_columns = required_columns_upper
        price_columns = ['Open', 'High', 'Low', 'Close']
        volume_column = 'Volume'
    else:
        required_columns = required_columns_lower
        price_columns = ['open', 'high', 'low', 'close']
        volume_column = 'volume'
    
    # Check for missing columns
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        issues.append(f"Missing required columns: {missing_columns}")
        # If we're missing all columns, return early
        if len(missing_columns) == len(required_columns):
            return False, issues
    
    # Get the columns that are actually present
    available_columns = [col for col in required_columns if col in data.columns]
    
    # Check for missing values in available columns
    if available_columns:
        missing_data = data[available_columns].isnull().sum()
        if missing_data.any():
            issues.append(f"Missing values found: {missing_data.to_dict()}")
    
    # Check for negative prices in available price columns
    available_price_columns = [col for col in price_columns if col in data.columns]
    for col in available_price_columns:
        if (data[col] <= 0).any():
            issues.append(f"Non-positive values found in {col}")
    
    # Check for negative volume
    if volume_column in data.columns and (data[volume_column] < 0).any():
        issues.append("Negative volume values found")
    
    # Check OHLC consistency if all price columns are available
    if all(col in data.columns for col in price_columns):
        # High should be >= Open, Close, Low
        high_col = price_columns[1]  # 'High' or 'high'
        if (data[high_col] < data[[price_columns[0], price_columns[2], price_columns[3]]].max(axis=1)).any():
            issues.append("High price inconsistency detected")
        
        # Low should be <= Open, Close, High
        low_col = price_columns[2]  # 'Low' or 'low'
        if (data[low_col] > data[[price_columns[0], price_columns[1], price_columns[3]]].min(axis=1)).any():
            issues.append("Low price inconsistency detected")
    
    # Check for extreme outliers (more than 50% daily change)
    close_col = price_columns[3]  # 'Close' or 'close'
    if close_col in data.columns and len(data) > 1:
        daily_returns = data[close_col].pct_change().dropna()
        extreme_moves = (abs(daily_returns) > 0.5).sum()
        if extreme_moves > 0:
            issues.append(f"Extreme price movements detected: {extreme_moves} days")
    
    # Check date index
    if not isinstance(data.index, pd.DatetimeIndex):
        issues.append("Index is not DatetimeIndex")
    
    is_valid = len(issues) == 0
    
    if not is_valid:
        logger.warning(f"Data validation failed for {symbol}: {issues}")
    else:
        logger.debug(f"Data validation passed for {symbol}")
    
    return is_valid, issues


def validate_technical_indicators(data, symbol: str) -> Tuple[bool, List[str]]:
    """
    Validate technical indicators for reasonable values.
    
    Args:
        data: DataFrame with technical indicators
        symbol: Stock symbol for logging
    
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []
    
    # Check RSI values (should be between 0 and 100)
    if 'rsi_14' in data.columns:
        rsi_data = data['rsi_14'].dropna()
        if not rsi_data.empty:
            if (rsi_data < 0).any() or (rsi_data > 100).any():
                issues.append("RSI values outside valid range (0-100)")
    
    # Check Bollinger Bands consistency
    bb_columns = ['bb_upper', 'bb_middle', 'bb_lower']
    if all(col in data.columns for col in bb_columns):
        bb_data = data[bb_columns].dropna()
        if not bb_data.empty:
            # Upper should be >= Middle >= Lower
            if (bb_data['bb_upper'] < bb_data['bb_middle']).any():
                issues.append("Bollinger Bands: Upper < Middle detected")
            if (bb_data['bb_middle'] < bb_data['bb_lower']).any():
                issues.append("Bollinger Bands: Middle < Lower detected")
    
    # Check SMA values (should be positive)
    if 'sma_50' in data.columns:
        sma_data = data['sma_50'].dropna()
        if not sma_data.empty and (sma_data <= 0).any():
            issues.append("Non-positive SMA values detected")
    
    is_valid = len(issues) == 0
    
    if not is_valid:
        logger.warning(f"Technical indicator validation failed for {symbol}: {issues}")
    
    return is_valid, issues


# Sentiment validation removed - no longer using sentiment data


def validate_trading_signal(signal: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate trading signal structure and values.
    
    Args:
        signal: Trading signal dictionary
    
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []
    
    # Check required fields
    required_fields = ['symbol', 'timestamp', 'action', 'confidence', 
                      'lstm_prediction', 'dqn_q_values', 
                      'risk_adjusted_size']
    
    for field in required_fields:
        if field not in signal:
            issues.append(f"Missing required field: {field}")
    
    # Validate action
    if 'action' in signal and signal['action'] not in ['buy', 'sell', 'hold']:
        issues.append(f"Invalid action: {signal['action']}")
    
    # Validate confidence (0 to 1)
    if 'confidence' in signal:
        conf = signal['confidence']
        if not isinstance(conf, (int, float)) or conf < 0 or conf > 1:
            issues.append(f"Invalid confidence value: {conf}")
    
    # Validate sentiment score (-1 to 1)
    if 'sentiment_score' in signal:
        sent = signal['sentiment_score']
        if not isinstance(sent, (int, float)) or sent < -1 or sent > 1:
            issues.append(f"Invalid sentiment score: {sent}")
    
    # Validate position size (should be positive)
    if 'risk_adjusted_size' in signal:
        size = signal['risk_adjusted_size']
        if not isinstance(size, (int, float)) or size < 0:
            issues.append(f"Invalid position size: {size}")
    
    is_valid = len(issues) == 0
    return is_valid, issues


def validate_system_dependencies() -> Tuple[bool, List[str]]:
    """
    Validate system dependencies and environment setup.
    
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []
    
    # Check required Python packages
    required_packages = [
        'pandas', 'numpy', 'yfinance', 'tensorflow', 
        'transformers', 'sklearn', 'matplotlib'
    ]
    
    for package in required_packages:
        if importlib.util.find_spec(package) is None:
            issues.append(f"Required package not found: {package}")
    
    # Check directory structure
    required_dirs = ['data', 'models', 'logs', 'config']
    for directory in required_dirs:
        if not os.path.exists(directory):
            try:
                os.makedirs(directory, exist_ok=True)
                logger.info(f"Created missing directory: {directory}")
            except Exception as e:
                issues.append(f"Cannot create directory {directory}: {str(e)}")
    
    # Check write permissions
    test_dirs = ['data', 'models', 'logs']
    for directory in test_dirs:
        if os.path.exists(directory):
            test_file = os.path.join(directory, 'test_write.tmp')
            try:
                with open(test_file, 'w') as f:
                    f.write('test')
                os.remove(test_file)
            except Exception as e:
                issues.append(f"No write permission for directory {directory}: {str(e)}")
    
    is_valid = len(issues) == 0
    
    if is_valid:
        logger.info("System dependency validation passed")
    else:
        logger.error(f"System dependency validation failed: {issues}")
    
    return is_valid, issues


def validate_date_range(start_date: str, end_date: str) -> Tuple[bool, List[str]]:
    """
    Validate date range for data collection.
    
    Args:
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD)
    
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []
    
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        
        # Check if start date is before end date
        if start_dt >= end_dt:
            issues.append("Start date must be before end date")
        
        # Check if dates are not too far in the future
        today = datetime.now()
        if start_dt > today:
            issues.append("Start date cannot be in the future")
        
        if end_dt > today + timedelta(days=1):
            issues.append("End date cannot be more than 1 day in the future")
        
        # Check if date range is reasonable (not too long)
        date_diff = end_dt - start_dt
        if date_diff.days > 365 * 50:  # 50 years
            issues.append("Date range too large (maximum 50 years)")
        
        if date_diff.days < 1:
            issues.append("Date range too small (minimum 1 day)")
            
    except ValueError as e:
        issues.append(f"Invalid date format: {str(e)}")
    
    is_valid = len(issues) == 0
    return is_valid, issues


def validate_model_inputs(data, model_type: str) -> Tuple[bool, List[str]]:
    """
    Validate model input data.
    
    Args:
        data: Input data array (requires numpy)
        model_type: Type of model ('lstm' or 'dqn')
    
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []
    
    if not NUMPY_AVAILABLE:
        issues.append("numpy not available for model input validation")
        return False, issues
    
    if data is None:
        issues.append("Input data is None")
        return False, issues
    
    if not isinstance(data, np.ndarray):
        issues.append("Input data is not numpy array")
        return False, issues
    
    if data.size == 0:
        issues.append("Input data is empty")
        return False, issues
    
    # Check for NaN or infinite values
    if np.isnan(data).any():
        issues.append("Input data contains NaN values")
    
    if np.isinf(data).any():
        issues.append("Input data contains infinite values")
    
    # Model-specific validation
    if model_type.lower() == 'lstm':
        if len(data.shape) < 2:
            issues.append("LSTM input data must be at least 2-dimensional")
        if data.shape[0] < 1:
            issues.append("LSTM input data must have at least 1 sample")
    
    elif model_type.lower() == 'dqn':
        if len(data.shape) != 1:
            issues.append("DQN input data must be 1-dimensional")
        if data.shape[0] < 1:
            issues.append("DQN input data must have at least 1 feature")
    
    is_valid = len(issues) == 0
    return is_valid, issues


def validate_portfolio_state(positions: List[Dict], total_capital: float) -> Tuple[bool, List[str]]:
    """
    Validate portfolio state for consistency.
    
    Args:
        positions: List of position dictionaries
        total_capital: Total portfolio capital
    
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []
    
    if total_capital <= 0:
        issues.append("Total capital must be positive")
    
    if not positions:
        return len(issues) == 0, issues
    
    # Check position structure
    required_position_fields = ['symbol', 'quantity', 'entry_price', 'current_value']
    for i, position in enumerate(positions):
        for field in required_position_fields:
            if field not in position:
                issues.append(f"Position {i} missing field: {field}")
        
        # Validate position values
        if 'quantity' in position and position['quantity'] <= 0:
            issues.append(f"Position {i} has non-positive quantity")
        
        if 'entry_price' in position and position['entry_price'] <= 0:
            issues.append(f"Position {i} has non-positive entry price")
    
    # Check for duplicate symbols
    symbols = [pos.get('symbol') for pos in positions if 'symbol' in pos]
    if len(symbols) != len(set(symbols)):
        issues.append("Duplicate symbols found in portfolio")
    
    # Check total portfolio value
    total_position_value = sum(pos.get('current_value', 0) for pos in positions)
    if total_position_value > total_capital * 1.1:  # Allow 10% buffer for price movements
        issues.append("Total position value exceeds capital by more than 10%")
    
    is_valid = len(issues) == 0
    return is_valid, issues
