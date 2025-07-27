"""
Data processing utilities for the quantitative trading system.
Provides data cleaning, normalization, and feature engineering capabilities.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any, Optional
from datetime import datetime
import warnings
import os

from src.utils.logging_utils import get_logger
from src.utils.validation_utils import validate_stock_data
from src.data.data_transformer import transform_yfinance_test_data

logger = get_logger("DataProcessor")


class DataCleaner:
    """
    Data cleaning utilities for stock market data.
    Handles missing values, outliers, and data validation.
    """
    
    def __init__(self, outlier_std_threshold: float = 3.0):
        """
        Initialize DataCleaner.
        
        Args:
            outlier_std_threshold: Number of standard deviations for outlier detection
        """
        self.outlier_std_threshold = outlier_std_threshold
        self.cleaning_stats = {}
    
    def clean_stock_data(self, data: pd.DataFrame, symbol: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Clean stock data by removing missing values and handling outliers.
        
        Args:
            data: Raw stock data DataFrame
            symbol: Stock symbol for logging
            
        Returns:
            Tuple of (cleaned_data, cleaning_statistics)
        """
        logger.info(f"Starting data cleaning for {symbol}")
        
        if data.empty:
            logger.warning(f"Empty data provided for {symbol}")
            return data, {"error": "Empty data"}
        
        original_rows = len(data)
        stats = {
            "symbol": symbol,
            "original_rows": original_rows,
            "missing_values_removed": 0,
            "outliers_capped": 0,
            "final_rows": 0,
            "data_quality_score": 0.0
        }
        
        # Make a copy to avoid modifying original data
        cleaned_data = data.copy()
        
        # Step 1: Normalize column names to lowercase
        cleaned_data.columns = [col.lower() for col in cleaned_data.columns]
        
        # Step 2: Remove rows with missing OHLCV values
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        available_columns = [col for col in required_columns if col in cleaned_data.columns]
        
        if not available_columns:
            logger.error(f"No required OHLCV columns found for {symbol}")
            return cleaned_data, {"error": "No OHLCV columns"}
        
        # Count missing values before cleaning
        missing_before = cleaned_data[available_columns].isnull().sum().sum()
        
        # Remove rows with any missing values in OHLCV columns
        cleaned_data = cleaned_data.dropna(subset=available_columns)
        rows_after_missing = len(cleaned_data)
        stats["missing_values_removed"] = original_rows - rows_after_missing
        
        if cleaned_data.empty:
            logger.warning(f"All data removed due to missing values for {symbol}")
            return cleaned_data, stats
        
        # Step 2: Handle outliers by capping at 3 standard deviations
        outliers_capped = 0
        price_columns = ['Open', 'High', 'Low', 'Close']
        
        for column in price_columns:
            if column in cleaned_data.columns:
                outliers_capped += self._cap_outliers(cleaned_data, column)
        
        # Also handle volume outliers
        if 'Volume' in cleaned_data.columns:
            outliers_capped += self._cap_outliers(cleaned_data, 'Volume')
        
        stats["outliers_capped"] = outliers_capped
        stats["final_rows"] = len(cleaned_data)
        
        # Step 3: Validate data consistency
        is_valid, validation_issues = validate_stock_data(cleaned_data, symbol)
        
        if not is_valid:
            logger.warning(f"Data validation issues for {symbol}: {validation_issues}")
            # Try to fix common issues
            cleaned_data = self._fix_data_inconsistencies(cleaned_data, symbol)
        
        # Calculate data quality score
        stats["data_quality_score"] = self._calculate_data_quality_score(
            original_rows, stats["final_rows"], missing_before, outliers_capped
        )
        
        # Store stats for this symbol
        self.cleaning_stats[symbol] = stats
        
        logger.info(f"Data cleaning completed for {symbol}: "
                   f"{stats['final_rows']}/{original_rows} rows retained, "
                   f"quality score: {stats['data_quality_score']:.2f}")
        
        return cleaned_data, stats
    
    def _cap_outliers(self, data: pd.DataFrame, column: str) -> int:
        """
        Cap outliers using Interquartile Range (IQR) method, which is more robust
        than standard deviation for outlier detection.
        
        Args:
            data: DataFrame to modify in-place
            column: Column name to process
            
        Returns:
            Number of values capped
        """
        if column not in data.columns or data[column].empty:
            return 0
        
        # Calculate statistics before any modifications
        original_values = data[column].copy()
        
        # Use IQR method for outlier detection (more robust than std dev)
        Q1 = original_values.quantile(0.25)
        Q3 = original_values.quantile(0.75)
        IQR = Q3 - Q1
        
        if IQR == 0:  # No variation in data
            return 0
        
        # Define outlier bounds using IQR method
        # Standard multiplier is 1.5, but we use 2.0 for less aggressive outlier removal
        multiplier = 2.0
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        # Count outliers before capping
        outliers_mask = (original_values < lower_bound) | (original_values > upper_bound)
        outliers_count = outliers_mask.sum()
        
        if outliers_count > 0:
            # Cap outliers
            data.loc[data[column] < lower_bound, column] = lower_bound
            data.loc[data[column] > upper_bound, column] = upper_bound
            
            logger.debug(f"Capped {outliers_count} outliers in {column} "
                        f"(IQR bounds: {lower_bound:.2f} - {upper_bound:.2f})")
        
        return outliers_count
    
    def _fix_data_inconsistencies(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Fix common data inconsistencies in OHLC data.
        
        Args:
            data: DataFrame to fix
            symbol: Stock symbol for logging
            
        Returns:
            Fixed DataFrame
        """
        fixed_data = data.copy()
        fixes_applied = 0
        
        price_columns = ['Open', 'High', 'Low', 'Close']
        if not all(col in fixed_data.columns for col in price_columns):
            return fixed_data
        
        for idx in fixed_data.index:
            row = fixed_data.loc[idx]
            
            # Fix: High should be >= max(Open, Close, Low)
            max_price = max(row['Open'], row['Close'], row['Low'])
            if row['High'] < max_price:
                fixed_data.loc[idx, 'High'] = max_price
                fixes_applied += 1
            
            # Fix: Low should be <= min(Open, Close, High)
            min_price = min(row['Open'], row['Close'], row['High'])
            if row['Low'] > min_price:
                fixed_data.loc[idx, 'Low'] = min_price
                fixes_applied += 1
        
        if fixes_applied > 0:
            logger.info(f"Applied {fixes_applied} OHLC consistency fixes for {symbol}")
        
        return fixed_data
    
    def _calculate_data_quality_score(self, original_rows: int, final_rows: int, 
                                    missing_values: int, outliers_capped: int) -> float:
        """
        Calculate a data quality score between 0 and 1.
        
        Args:
            original_rows: Original number of rows
            final_rows: Final number of rows after cleaning
            missing_values: Number of missing values found
            outliers_capped: Number of outliers capped
            
        Returns:
            Quality score between 0 and 1
        """
        if original_rows == 0:
            return 0.0
        
        # Base score from data retention
        retention_score = final_rows / original_rows
        
        # Penalty for missing values (normalized by total data points)
        missing_penalty = min(missing_values / (original_rows * 5), 0.3)  # Assume 5 columns
        
        # Penalty for outliers (normalized by final rows)
        outlier_penalty = min(outliers_capped / max(final_rows, 1), 0.2) if final_rows > 0 else 0
        
        # Calculate final score
        quality_score = max(0.0, retention_score - missing_penalty - outlier_penalty)
        
        return min(1.0, quality_score)
    
    def get_cleaning_summary(self) -> Dict[str, Any]:
        """
        Get summary of all cleaning operations performed.
        
        Returns:
            Dictionary with cleaning statistics
        """
        if not self.cleaning_stats:
            return {"message": "No cleaning operations performed yet"}
        
        total_symbols = len(self.cleaning_stats)
        total_original_rows = sum(stats["original_rows"] for stats in self.cleaning_stats.values())
        total_final_rows = sum(stats["final_rows"] for stats in self.cleaning_stats.values())
        total_missing_removed = sum(stats["missing_values_removed"] for stats in self.cleaning_stats.values())
        total_outliers_capped = sum(stats["outliers_capped"] for stats in self.cleaning_stats.values())
        
        avg_quality_score = np.mean([stats["data_quality_score"] for stats in self.cleaning_stats.values()])
        
        return {
            "total_symbols_processed": total_symbols,
            "total_original_rows": total_original_rows,
            "total_final_rows": total_final_rows,
            "total_missing_values_removed": total_missing_removed,
            "total_outliers_capped": total_outliers_capped,
            "overall_retention_rate": total_final_rows / max(total_original_rows, 1),
            "average_quality_score": avg_quality_score,
            "individual_stats": self.cleaning_stats
        }
    
    def validate_cleaned_data(self, data: pd.DataFrame, symbol: str) -> Tuple[bool, List[str]]:
        """
        Validate cleaned data meets quality standards.
        
        Args:
            data: Cleaned data to validate
            symbol: Stock symbol for logging
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check minimum data requirements
        if len(data) < 30:  # Need at least 30 days of data
            issues.append(f"Insufficient data: {len(data)} rows (minimum 30 required)")
        
        # Check data completeness - support both uppercase and lowercase column names
        required_columns_upper = ['Open', 'High', 'Low', 'Close', 'Volume']
        required_columns_lower = ['open', 'high', 'low', 'close', 'volume']
        
        # Check for uppercase columns
        available_columns_upper = [col for col in required_columns_upper if col in data.columns]
        # Check for lowercase columns
        available_columns_lower = [col for col in required_columns_lower if col in data.columns]
        
        # Use whichever set has more matches
        if len(available_columns_upper) >= len(available_columns_lower):
            available_columns = available_columns_upper
            required_columns = required_columns_upper
        else:
            available_columns = available_columns_lower
            required_columns = required_columns_lower
        
        if len(available_columns) < 4:  # Need at least OHLC
            issues.append(f"Missing required columns. Available: {available_columns}")
        
        # Check for remaining missing values
        missing_values = data[available_columns].isnull().sum().sum()
        if missing_values > 0:
            issues.append(f"Missing values still present: {missing_values}")
        
        # Check for non-positive prices
        price_columns = ['Open', 'High', 'Low', 'Close']
        for col in price_columns:
            if col in data.columns and (data[col] <= 0).any():
                issues.append(f"Non-positive values in {col}")
        
        # Check for extreme values that might indicate cleaning issues
        if 'Close' in data.columns and len(data) > 1:
            daily_returns = data['Close'].pct_change().dropna()
            extreme_returns = (abs(daily_returns) > 0.5).sum()  # More than 50% daily change
            if extreme_returns > len(data) * 0.05:  # More than 5% of days
                issues.append(f"Excessive extreme returns: {extreme_returns} days")
        
        # Use existing validation function
        is_structurally_valid, structural_issues = validate_stock_data(data, symbol)
        issues.extend(structural_issues)
        
        is_valid = len(issues) == 0
        
        if not is_valid:
            logger.warning(f"Cleaned data validation failed for {symbol}: {issues}")
        
        return is_valid, issues


class DataValidator:
    """
    Comprehensive data validation utilities.
    """
    
    @staticmethod
    def check_data_completeness(data: pd.DataFrame, symbol: str, 
                              required_date_range: Tuple[str, str] = None) -> Dict[str, Any]:
        """
        Check data completeness for a given symbol.
        
        Args:
            data: Stock data DataFrame
            symbol: Stock symbol
            required_date_range: Optional tuple of (start_date, end_date) strings
            
        Returns:
            Dictionary with completeness statistics
        """
        stats = {
            "symbol": symbol,
            "total_rows": len(data),
            "date_range": None,
            "missing_dates": 0,
            "completeness_score": 0.0,
            "issues": []
        }
        
        if data.empty:
            stats["issues"].append("Empty dataset")
            return stats
        
        # Get actual date range
        if isinstance(data.index, pd.DatetimeIndex):
            actual_start = data.index.min()
            actual_end = data.index.max()
            stats["date_range"] = (actual_start.strftime("%Y-%m-%d"), 
                                 actual_end.strftime("%Y-%m-%d"))
            
            # Check for missing dates if required range is provided
            if required_date_range:
                try:
                    required_start = pd.to_datetime(required_date_range[0])
                    required_end = pd.to_datetime(required_date_range[1])
                    
                    # Create expected date range (business days only)
                    expected_dates = pd.bdate_range(start=required_start, end=required_end)
                    actual_dates = set(data.index.date)
                    expected_dates_set = set(expected_dates.date)
                    
                    missing_dates = expected_dates_set - actual_dates
                    stats["missing_dates"] = len(missing_dates)
                    
                    # Calculate completeness score
                    stats["completeness_score"] = 1.0 - (len(missing_dates) / len(expected_dates))
                    
                    if len(missing_dates) > 0:
                        stats["issues"].append(f"Missing {len(missing_dates)} trading days")
                        
                except Exception as e:
                    stats["issues"].append(f"Date range validation error: {str(e)}")
        else:
            stats["issues"].append("Data index is not DatetimeIndex")
        
        return stats
    
    @staticmethod
    def check_data_consistency(data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """
        Check internal data consistency.
        
        Args:
            data: Stock data DataFrame
            symbol: Stock symbol
            
        Returns:
            Dictionary with consistency check results
        """
        results = {
            "symbol": symbol,
            "ohlc_consistency": True,
            "volume_consistency": True,
            "price_continuity": True,
            "issues": []
        }
        
        if data.empty:
            results["issues"].append("Empty dataset")
            return results
        
        price_columns = ['Open', 'High', 'Low', 'Close']
        available_price_cols = [col for col in price_columns if col in data.columns]
        
        if len(available_price_cols) >= 4:
            # Check OHLC relationships
            ohlc_issues = 0
            for idx in data.index:
                row = data.loc[idx]
                
                # High should be the highest
                if not (row['High'] >= max(row['Open'], row['Close'], row['Low'])):
                    ohlc_issues += 1
                
                # Low should be the lowest
                if not (row['Low'] <= min(row['Open'], row['Close'], row['High'])):
                    ohlc_issues += 1
            
            if ohlc_issues > 0:
                results["ohlc_consistency"] = False
                results["issues"].append(f"OHLC inconsistencies: {ohlc_issues} rows")
        
        # Check volume consistency
        if 'Volume' in data.columns:
            negative_volume = (data['Volume'] < 0).sum()
            if negative_volume > 0:
                results["volume_consistency"] = False
                results["issues"].append(f"Negative volume values: {negative_volume} rows")
        
        # Check price continuity (no extreme gaps)
        if 'Close' in data.columns and len(data) > 1:
            price_changes = data['Close'].pct_change().dropna()
            extreme_gaps = (abs(price_changes) > 0.5).sum()  # More than 50% change
            
            if extreme_gaps > 0:
                results["price_continuity"] = False
                results["issues"].append(f"Extreme price gaps: {extreme_gaps} occurrences")
        
        return results


class DataNormalizer:
    """
    Data normalization and feature preparation utilities.
    Handles scaling, splitting, and feature combination for model training.
    """
    
    def __init__(self):
        """Initialize DataNormalizer."""
        self.scalers = {}  # Store scalers for each feature
        self.normalization_stats = {}
    
    def normalize_features(self, data: pd.DataFrame, feature_columns: List[str], 
                          method: str = 'minmax') -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Normalize numerical features using specified method.
        
        Args:
            data: DataFrame with features to normalize
            feature_columns: List of column names to normalize
            method: Normalization method ('minmax', 'standard', 'robust')
            
        Returns:
            Tuple of (normalized_data, normalization_stats)
        """
        logger.info(f"Normalizing {len(feature_columns)} features using {method} method")
        
        if data.empty:
            return data, {"error": "Empty data"}
        
        normalized_data = data.copy()
        stats = {
            "method": method,
            "features_normalized": [],
            "normalization_params": {},
            "original_ranges": {},
            "normalized_ranges": {}
        }
        
        available_columns = [col for col in feature_columns if col in data.columns]
        if not available_columns:
            logger.warning("No specified feature columns found in data")
            return normalized_data, {"error": "No feature columns found"}
        
        for column in available_columns:
            if not pd.api.types.is_numeric_dtype(data[column]):
                logger.warning(f"Skipping non-numeric column: {column}")
                continue
            
            # Skip boolean columns as they can cause issues with arithmetic operations
            if data[column].dtype == bool:
                logger.warning(f"Skipping boolean column: {column}")
                continue
            
            # Store original range
            original_min = data[column].min()
            original_max = data[column].max()
            stats["original_ranges"][column] = (original_min, original_max)
            
            # Apply normalization
            if method == 'minmax':
                normalized_data[column], params = self._apply_minmax_scaling(data[column])
            elif method == 'standard':
                normalized_data[column], params = self._apply_standard_scaling(data[column])
            elif method == 'robust':
                normalized_data[column], params = self._apply_robust_scaling(data[column])
            else:
                logger.error(f"Unknown normalization method: {method}")
                continue
            
            # Store normalization parameters
            stats["normalization_params"][column] = params
            stats["features_normalized"].append(column)
            
            # Store normalized range
            norm_min = normalized_data[column].min()
            norm_max = normalized_data[column].max()
            stats["normalized_ranges"][column] = (norm_min, norm_max)
            
            logger.debug(f"Normalized {column}: {original_min:.4f}-{original_max:.4f} -> "
                        f"{norm_min:.4f}-{norm_max:.4f}")
        
        # Store stats for this normalization
        self.normalization_stats[f"{method}_{len(self.normalization_stats)}"] = stats
        
        logger.info(f"Normalization completed: {len(stats['features_normalized'])} features processed")
        return normalized_data, stats
    
    def _apply_minmax_scaling(self, series: pd.Series) -> Tuple[pd.Series, Dict[str, float]]:
        """Apply min-max scaling to a series."""
        min_val = series.min()
        max_val = series.max()
        
        if min_val == max_val:  # No variation
            return series, {"min": min_val, "max": max_val, "range": 0}
        
        # Scale to [0, 1] range
        scaled_series = (series - min_val) / (max_val - min_val)
        
        params = {
            "min": min_val,
            "max": max_val,
            "range": max_val - min_val
        }
        
        return scaled_series, params
    
    def _apply_standard_scaling(self, series: pd.Series) -> Tuple[pd.Series, Dict[str, float]]:
        """Apply standard scaling (z-score normalization) to a series."""
        mean_val = series.mean()
        std_val = series.std()
        
        if std_val == 0:  # No variation
            return series - mean_val, {"mean": mean_val, "std": std_val}
        
        # Scale to mean=0, std=1
        scaled_series = (series - mean_val) / std_val
        
        params = {
            "mean": mean_val,
            "std": std_val
        }
        
        return scaled_series, params
    
    def _apply_robust_scaling(self, series: pd.Series) -> Tuple[pd.Series, Dict[str, float]]:
        """Apply robust scaling using median and IQR."""
        median_val = series.median()
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        
        if iqr == 0:  # No variation
            return series - median_val, {"median": median_val, "iqr": iqr, "q1": q1, "q3": q3}
        
        # Scale using median and IQR
        scaled_series = (series - median_val) / iqr
        
        params = {
            "median": median_val,
            "iqr": iqr,
            "q1": q1,
            "q3": q3
        }
        
        return scaled_series, params
    
    def split_train_test(self, data: pd.DataFrame, test_size: float = 0.2, 
                        time_based: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """
        Split data into training and testing sets.
        
        Args:
            data: DataFrame to split
            test_size: Proportion of data for testing (0.0 to 1.0)
            time_based: If True, use chronological split; if False, use random split
            
        Returns:
            Tuple of (train_data, test_data, split_stats)
        """
        logger.info(f"Splitting data: {len(data)} rows, test_size={test_size}, time_based={time_based}")
        
        if data.empty:
            return data, data, {"error": "Empty data"}
        
        if not 0 < test_size < 1:
            raise ValueError("test_size must be between 0 and 1")
        
        stats = {
            "total_rows": len(data),
            "test_size": test_size,
            "time_based": time_based,
            "train_rows": 0,
            "test_rows": 0,
            "split_date": None
        }
        
        if time_based:
            # Chronological split - use last test_size portion for testing
            split_index = int(len(data) * (1 - test_size))
            train_data = data.iloc[:split_index].copy()
            test_data = data.iloc[split_index:].copy()
            
            if isinstance(data.index, pd.DatetimeIndex):
                stats["split_date"] = data.index[split_index].strftime("%Y-%m-%d")
        else:
            # Random split
            np.random.seed(42)  # For reproducible results
            shuffled_indices = np.random.permutation(len(data))
            split_index = int(len(data) * (1 - test_size))
            
            train_indices = shuffled_indices[:split_index]
            test_indices = shuffled_indices[split_index:]
            
            train_data = data.iloc[train_indices].copy()
            test_data = data.iloc[test_indices].copy()
        
        stats["train_rows"] = len(train_data)
        stats["test_rows"] = len(test_data)
        
        logger.info(f"Data split completed: {stats['train_rows']} train, {stats['test_rows']} test")
        
        return train_data, test_data, stats
    
    def combine_features(self, price_data: pd.DataFrame, 
                        technical_indicators: pd.DataFrame = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Combine price data with technical indicators.
        
        Args:
            price_data: DataFrame with OHLCV data
            technical_indicators: Optional DataFrame with technical indicators
            
        Returns:
            Tuple of (combined_data, combination_stats)
        """
        logger.info("Combining features from multiple data sources")
        
        if price_data.empty:
            return price_data, {"error": "Empty price data"}
        
        combined_data = price_data.copy()
        stats = {
            "original_features": list(price_data.columns),
            "added_features": [],
            "total_features": len(price_data.columns),
            "rows_before": len(price_data),
            "rows_after": 0,
            "missing_data_handled": 0
        }
        
        # Add technical indicators
        if technical_indicators is not None and not technical_indicators.empty:
            # Align indices and merge
            aligned_indicators = technical_indicators.reindex(combined_data.index)
            
            for col in technical_indicators.columns:
                if col not in combined_data.columns:
                    combined_data[col] = aligned_indicators[col]
                    stats["added_features"].append(col)
                    logger.debug(f"Added technical indicator: {col}")
        
        # Sentiment data removed - no longer using sentiment features
        
        # Handle missing values in combined data
        missing_before = combined_data.isnull().sum().sum()
        if missing_before > 0:
            # Forward fill then backward fill for time series data
            combined_data = combined_data.ffill().bfill()
            missing_after = combined_data.isnull().sum().sum()
            stats["missing_data_handled"] = missing_before - missing_after
            
            if missing_after > 0:
                # Fill remaining missing values with 0 (neutral for sentiment, etc.)
                combined_data = combined_data.fillna(0)
                logger.warning(f"Filled {missing_after} remaining missing values with 0")
        
        stats["total_features"] = len(combined_data.columns)
        stats["rows_after"] = len(combined_data)
        
        logger.info(f"Feature combination completed: {len(stats['added_features'])} features added, "
                   f"{stats['total_features']} total features")
        
        return combined_data, stats
    
    def prepare_model_inputs(self, data: pd.DataFrame, target_column: str,
                           feature_columns: List[str] = None,
                           sequence_length: int = None) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Prepare data for model training by creating feature matrices and target vectors.
        
        Args:
            data: Combined and normalized DataFrame
            target_column: Name of target column
            feature_columns: List of feature column names (if None, use all except target)
            sequence_length: For LSTM models, create sequences of this length
            
        Returns:
            Tuple of (X, y, preparation_stats)
        """
        logger.info(f"Preparing model inputs: target='{target_column}', "
                   f"sequence_length={sequence_length}")
        
        if data.empty:
            return np.array([]), np.array([]), {"error": "Empty data"}
        
        if target_column not in data.columns:
            return np.array([]), np.array([]), {"error": f"Target column '{target_column}' not found"}
        
        # Select feature columns
        if feature_columns is None:
            feature_columns = [col for col in data.columns if col != target_column]
        else:
            feature_columns = [col for col in feature_columns if col in data.columns and col != target_column]
        
        if not feature_columns:
            return np.array([]), np.array([]), {"error": "No valid feature columns"}
        
        stats = {
            "target_column": target_column,
            "feature_columns": feature_columns,
            "sequence_length": sequence_length,
            "original_rows": len(data),
            "feature_count": len(feature_columns),
            "samples_created": 0,
            "input_shape": None,
            "target_shape": None
        }
        
        # Extract features and target
        X_data = data[feature_columns].values
        y_data = data[target_column].values
        
        if sequence_length is not None and sequence_length > 1:
            # Create sequences for LSTM
            X_sequences = []
            y_sequences = []
            
            for i in range(sequence_length, len(data)):
                X_sequences.append(X_data[i-sequence_length:i])
                y_sequences.append(y_data[i])
            
            X = np.array(X_sequences)
            y = np.array(y_sequences)
            
            stats["samples_created"] = len(X_sequences)
            stats["input_shape"] = X.shape
            stats["target_shape"] = y.shape
            
            logger.info(f"Created {len(X_sequences)} sequences of length {sequence_length}")
        else:
            # Regular feature matrix
            X = X_data
            y = y_data
            
            stats["samples_created"] = len(X)
            stats["input_shape"] = X.shape
            stats["target_shape"] = y.shape
        
        logger.info(f"Model input preparation completed: X shape {X.shape}, y shape {y.shape}")
        
        return X, y, stats
    
    def get_normalization_summary(self) -> Dict[str, Any]:
        """
        Get summary of all normalization operations performed.
        
        Returns:
            Dictionary with normalization statistics
        """
        if not self.normalization_stats:
            return {"message": "No normalization operations performed yet"}
        
        total_operations = len(self.normalization_stats)
        methods_used = list(set(stats["method"] for stats in self.normalization_stats.values()))
        total_features_normalized = sum(len(stats["features_normalized"]) 
                                      for stats in self.normalization_stats.values())
        
        return {
            "total_operations": total_operations,
            "methods_used": methods_used,
            "total_features_normalized": total_features_normalized,
            "individual_stats": self.normalization_stats
        }

class DataProcessor:
    """
    Main data processing class that orchestrates data cleaning, validation,
    normalization, and feature engineering for the trading system.
    """
    
    def __init__(self):
        """Initialize DataProcessor with its component processors."""
        self.data_cleaner = DataCleaner()
        self.validator = DataValidator()
        self.normalizer = DataNormalizer()
        logger.info("DataProcessor initialized")
    
    def process_stock_data(self, data: pd.DataFrame, symbol: str, file_path: str = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Process raw stock data through cleaning, validation, and normalization.
        
        Args:
            data: Raw stock data DataFrame
            symbol: Stock symbol for logging
            file_path: Optional path to the original data file for transformation
            
        Returns:
            Tuple of (processed_data, processing_stats)
        """
        logger.info(f"Processing stock data for {symbol}")
        
        # Check if we need to transform the data format
        if file_path and os.path.exists(file_path) and "test_data" in file_path:
            logger.info(f"Attempting to transform test data from {file_path}")
            transformed_data = transform_yfinance_test_data(file_path)
            if transformed_data is not None:
                data = transformed_data
                logger.info(f"Successfully transformed test data for {symbol}")
        
        stats = {
            "symbol": symbol,
            "original_rows": len(data),
            "final_rows": 0,
            "cleaning_stats": {},
            "validation_stats": {},
            "normalization_stats": {}
        }
        
        # Step 1: Clean the data
        cleaned_data, cleaning_stats = self.data_cleaner.clean_stock_data(data, symbol)
        stats["cleaning_stats"] = cleaning_stats
        
        if cleaned_data.empty:
            logger.warning(f"Cleaning resulted in empty data for {symbol}")
            return cleaned_data, stats
        
        # Step 2: Validate the cleaned data
        is_valid, validation_issues = self.data_cleaner.validate_cleaned_data(cleaned_data, symbol)
        stats["validation_stats"] = {
            "is_valid": is_valid,
            "issues": validation_issues
        }
        
        if not is_valid:
            logger.warning(f"Data validation failed for {symbol}: {validation_issues}")
            # Continue processing but log the issues
        
        # Step 3: Normalize price data for modeling
        price_columns = ['Open', 'High', 'Low', 'Close']
        available_price_cols = [col for col in price_columns if col in cleaned_data.columns]
        
        if available_price_cols:
            normalized_data, norm_stats = self.normalizer.normalize_features(
                cleaned_data, available_price_cols, method='minmax'
            )
            stats["normalization_stats"] = norm_stats
        else:
            normalized_data = cleaned_data
            stats["normalization_stats"] = {"error": "No price columns available for normalization"}
        
        stats["final_rows"] = len(normalized_data)
        
        logger.info(f"Data processing completed for {symbol}: "
                   f"{stats['final_rows']}/{stats['original_rows']} rows retained")
        
        return normalized_data, stats
    
    def prepare_features(self, price_data: pd.DataFrame, 
                        technical_data: pd.DataFrame = None,
                        sentiment_data: pd.DataFrame = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Prepare features by combining price data with technical indicators and sentiment.
        
        Args:
            price_data: Processed price data
            technical_data: Technical indicators data
            sentiment_data: Sentiment scores data
            
        Returns:
            Tuple of (feature_data, feature_stats)
        """
        logger.info("Preparing features for modeling")
        
        # Combine all data sources
        combined_data, combination_stats = self.normalizer.combine_features(
            price_data, technical_data, sentiment_data
        )
        
        # Normalize all features to the same scale
        all_features = list(combined_data.columns)
        normalized_features, normalization_stats = self.normalizer.normalize_features(
            combined_data, all_features, method='standard'
        )
        
        stats = {
            "combination_stats": combination_stats,
            "normalization_stats": normalization_stats,
            "total_features": len(normalized_features.columns),
            "feature_names": list(normalized_features.columns)
        }
        
        logger.info(f"Feature preparation completed: {stats['total_features']} features available")
        
        return normalized_features, stats
    
    def create_training_data(self, feature_data: pd.DataFrame, 
                           target_column: str = 'Close',
                           sequence_length: int = 10) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Create training data for machine learning models.
        
        Args:
            feature_data: Normalized feature data
            target_column: Target column for prediction
            sequence_length: Sequence length for LSTM models
            
        Returns:
            Tuple of (X, y, preparation_stats)
        """
        logger.info(f"Creating training data with sequence length {sequence_length}")
        
        # Prepare model inputs
        X, y, stats = self.normalizer.prepare_model_inputs(
            feature_data, 
            target_column=target_column,
            sequence_length=sequence_length
        )
        
        logger.info(f"Training data created: X shape {X.shape}, y shape {y.shape}")
        
        return X, y, stats
