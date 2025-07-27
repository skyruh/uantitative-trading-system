"""
Feature builder module for combining price data with technical indicators.
Implements the FeatureBuilder class for comprehensive feature engineering.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any
import logging
from datetime import datetime

from src.data.technical_indicators import TechnicalIndicators
from src.data.data_storage import DataStorage
from src.interfaces.data_interfaces import IFeatureEngineer


class FeatureBuilder(IFeatureEngineer):
    """
    Feature builder class that combines price data with technical indicators.
    
    Features:
    - Calculates and integrates technical indicators with price data
    - Handles missing data and calculation failures gracefully
    - Stores calculated indicators alongside price data
    - Provides comprehensive error handling and logging
    """
    
    def __init__(self, data_storage: Optional[DataStorage] = None):
        """
        Initialize FeatureBuilder.
        
        Args:
            data_storage: DataStorage instance for saving/loading data
        """
        self.data_storage = data_storage or DataStorage()
        self.logger = logging.getLogger(__name__)
        self.technical_indicators = TechnicalIndicators()
    
    def calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators from price data.
        
        Args:
            data: DataFrame with OHLCV price data
            
        Returns:
            DataFrame with added technical indicator columns
            
        Raises:
            ValueError: If input data is invalid
        """
        if data.empty:
            raise ValueError("Input data is empty")
            
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        try:
            self.logger.info(f"Calculating technical indicators for {len(data)} data points")
            
            # Calculate all technical indicators
            result = self.technical_indicators.calculate_all_indicators(data)
            
            # Log indicator calculation results
            indicator_columns = ['rsi_14', 'sma_50', 'bb_upper', 'bb_middle', 'bb_lower']
            for col in indicator_columns:
                if col in result.columns:
                    valid_count = result[col].notna().sum()
                    self.logger.debug(f"{col}: {valid_count}/{len(result)} valid values")
            
            self.logger.info("Successfully calculated technical indicators")
            return result
            
        except Exception as e:
            self.logger.error(f"Error calculating technical indicators: {str(e)}")
            # Return original data with empty indicator columns on error
            result = data.copy()
            indicator_columns = ['rsi_14', 'sma_50', 'bb_upper', 'bb_middle', 'bb_lower']
            for col in indicator_columns:
                result[col] = np.nan
            return result
    
    def calculate_sentiment_scores(self, news_data: List[Dict]) -> List[float]:
        """
        Calculate sentiment scores from news headlines.
        
        Note: This is a placeholder implementation. In task 5, this will be
        replaced with actual DistilBERT sentiment analysis.
        
        Args:
            news_data: List of news dictionaries with 'title' field
            
        Returns:
            List of sentiment scores between -1 and 1
        """
        if not news_data:
            return []
        
        try:
            # Placeholder implementation - returns neutral sentiment
            # This will be replaced with actual DistilBERT implementation in task 5
            sentiment_scores = [0.0] * len(news_data)
            
            self.logger.info(f"Calculated sentiment scores for {len(news_data)} news items (placeholder)")
            return sentiment_scores
            
        except Exception as e:
            self.logger.error(f"Error calculating sentiment scores: {str(e)}")
            return [0.0] * len(news_data)
    
    def build_features(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """
        Combine all features for model training.
        
        Args:
            price_data: DataFrame with price and technical indicator data
            
        Returns:
            DataFrame with combined features
        """
        if price_data.empty:
            self.logger.warning("Price data is empty, returning empty DataFrame")
            return pd.DataFrame()
        
        try:
            result = price_data.copy()
            
            # Sentiment data removed - no longer using sentiment features
            
            # Ensure all required feature columns exist
            required_features = ['rsi_14', 'sma_50', 'bb_upper', 'bb_middle', 'bb_lower']
            for feature in required_features:
                if feature not in result.columns:
                    result[feature] = np.nan
                    self.logger.warning(f"Missing feature column '{feature}', filled with NaN")
            
            # Add derived features
            result = self._add_derived_features(result)
            
            self.logger.info(f"Built feature set with {len(result.columns)} columns for {len(result)} data points")
            return result
            
        except Exception as e:
            self.logger.error(f"Error building features: {str(e)}")
            return price_data.copy()
    
    def _add_derived_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add derived features from existing data.
        
        Args:
            data: DataFrame with basic features
            
        Returns:
            DataFrame with additional derived features
        """
        try:
            result = data.copy()
            
            # Price-based features
            if 'close' in result.columns:
                # Price change and returns
                result['price_change'] = result['close'].diff()
                result['price_return'] = result['close'].pct_change(fill_method=None)
                
                # Volatility (rolling standard deviation of returns)
                result['volatility_10d'] = result['price_return'].rolling(window=10).std()
                
            # Technical indicator-based features
            if 'close' in result.columns and 'bb_middle' in result.columns:
                # Bollinger Band position
                bb_width = result['bb_upper'] - result['bb_lower']
                result['bb_position'] = (result['close'] - result['bb_lower']) / bb_width
                result['bb_width'] = bb_width
                
            if 'close' in result.columns and 'sma_50' in result.columns:
                # Price relative to SMA
                result['price_sma_ratio'] = result['close'] / result['sma_50']
                
            # Volume-based features
            if 'volume' in result.columns:
                result['volume_sma_20'] = result['volume'].rolling(window=20).mean()
                result['volume_ratio'] = result['volume'] / result['volume_sma_20']
                
            self.logger.debug(f"Added {len(result.columns) - len(data.columns)} derived features")
            return result
            
        except Exception as e:
            self.logger.error(f"Error adding derived features: {str(e)}")
            return data
    
    def process_symbol_features(self, symbol: str, price_data: Optional[pd.DataFrame] = None, 
                              news_data: Optional[List[Dict]] = None, 
                              save_indicators: bool = True) -> pd.DataFrame:
        """
        Process complete feature set for a symbol.
        
        Args:
            symbol: Stock symbol
            price_data: Price data (if None, loads from storage)
            news_data: News data (if None, loads from storage)
            save_indicators: Whether to save calculated indicators to storage
            
        Returns:
            DataFrame with complete feature set
        """
        try:
            self.logger.info(f"Processing features for symbol: {symbol}")
            
            # Load price data if not provided
            if price_data is None:
                price_data = self.data_storage.load_stock_data(symbol)
                if price_data.empty:
                    self.logger.warning(f"No price data found for {symbol}")
                    return pd.DataFrame()
            
            # Load news data if not provided
            if news_data is None:
                news_data = self.data_storage.load_news_data(symbol)
            
            # Calculate technical indicators
            data_with_indicators = self.calculate_technical_indicators(price_data)
            
            # Calculate sentiment scores
            sentiment_scores = self.calculate_sentiment_scores(news_data)
            
            # Build complete feature set
            features = self.build_features(data_with_indicators, sentiment_scores)
            
            # Save indicators to storage if requested
            if save_indicators and not features.empty:
                success = self.data_storage.save_indicators_data(symbol, features)
                if success:
                    self.logger.info(f"Saved indicators data for {symbol}")
                else:
                    self.logger.warning(f"Failed to save indicators data for {symbol}")
            
            self.logger.info(f"Successfully processed features for {symbol}: {len(features)} rows, {len(features.columns)} columns")
            return features
            
        except Exception as e:
            self.logger.error(f"Error processing features for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def build_features_for_all_stocks(self, symbols: List[str]) -> bool:
        """
        Build features for all stocks in the provided list.
        
        Args:
            symbols: List of stock symbols
            
        Returns:
            True if successful for at least one symbol, False otherwise
        """
        try:
            self.logger.info(f"Building features for {len(symbols)} stocks")
            
            results = self.process_multiple_symbols(symbols, save_indicators=True)
            
            success_count = len(results)
            if success_count > 0:
                self.logger.info(f"Successfully built features for {success_count}/{len(symbols)} stocks")
                return True
            else:
                self.logger.error(f"Failed to build features for any of the {len(symbols)} stocks")
                return False
                
        except Exception as e:
            self.logger.error(f"Error building features for stocks: {str(e)}")
            return False
    
    def process_multiple_symbols(self, symbols: List[str], 
                               save_indicators: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Process features for multiple symbols.
        
        Args:
            symbols: List of stock symbols
            save_indicators: Whether to save calculated indicators to storage
            
        Returns:
            Dictionary mapping symbols to their feature DataFrames
        """
        results = {}
        
        for symbol in symbols:
            try:
                features = self.process_symbol_features(symbol, save_indicators=save_indicators)
                if not features.empty:
                    results[symbol] = features
                    self.logger.info(f"Successfully processed {symbol}")
                else:
                    self.logger.warning(f"No features generated for {symbol}")
                    
            except Exception as e:
                self.logger.error(f"Failed to process {symbol}: {str(e)}")
                continue
        
        self.logger.info(f"Processed features for {len(results)}/{len(symbols)} symbols")
        return results
    
    def validate_features(self, features: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate feature data quality.
        
        Args:
            features: DataFrame with feature data
            
        Returns:
            Dictionary with validation results
        """
        if features.empty:
            return {"valid": False, "error": "Empty feature data"}
        
        validation_results = {
            "valid": True,
            "total_rows": len(features),
            "total_columns": len(features.columns),
            "missing_data": {},
            "data_quality": {},
            "warnings": []
        }
        
        try:
            # Check for missing data
            for column in features.columns:
                missing_count = features[column].isna().sum()
                missing_percentage = (missing_count / len(features)) * 100
                validation_results["missing_data"][column] = {
                    "count": int(missing_count),
                    "percentage": round(missing_percentage, 2)
                }
                
                if missing_percentage > 50:
                    validation_results["warnings"].append(f"Column '{column}' has {missing_percentage:.1f}% missing data")
            
            # Check data quality for key indicators
            key_indicators = ['rsi_14', 'sma_50', 'bb_upper', 'bb_middle', 'bb_lower']
            for indicator in key_indicators:
                if indicator in features.columns:
                    valid_data = features[indicator].dropna()
                    if len(valid_data) > 0:
                        validation_results["data_quality"][indicator] = {
                            "min": float(valid_data.min()),
                            "max": float(valid_data.max()),
                            "mean": float(valid_data.mean()),
                            "valid_count": len(valid_data)
                        }
                        
                        # RSI should be between 0 and 100
                        if indicator == 'rsi_14':
                            invalid_rsi = ((valid_data < 0) | (valid_data > 100)).sum()
                            if invalid_rsi > 0:
                                validation_results["warnings"].append(f"Found {invalid_rsi} invalid RSI values")
            
            # Check for infinite values
            inf_columns = []
            for column in features.select_dtypes(include=[np.number]).columns:
                if np.isinf(features[column]).any():
                    inf_columns.append(column)
            
            if inf_columns:
                validation_results["warnings"].append(f"Found infinite values in columns: {inf_columns}")
            
            self.logger.info(f"Feature validation completed: {len(validation_results['warnings'])} warnings")
            
        except Exception as e:
            self.logger.error(f"Error during feature validation: {str(e)}")
            validation_results["valid"] = False
            validation_results["error"] = str(e)
        
        return validation_results
    
    def get_feature_summary(self, features: pd.DataFrame) -> Dict[str, Any]:
        """
        Get summary statistics for features.
        
        Args:
            features: DataFrame with feature data
            
        Returns:
            Dictionary with feature summary
        """
        if features.empty:
            return {"error": "Empty feature data"}
        
        try:
            summary = {
                "shape": features.shape,
                "date_range": {},
                "feature_stats": {},
                "data_completeness": {}
            }
            
            # Date range information
            if 'date' in features.columns:
                date_col = pd.to_datetime(features['date'])
                summary["date_range"] = {
                    "start": date_col.min().strftime('%Y-%m-%d'),
                    "end": date_col.max().strftime('%Y-%m-%d'),
                    "days": (date_col.max() - date_col.min()).days
                }
            
            # Feature statistics
            numeric_columns = features.select_dtypes(include=[np.number]).columns
            for column in numeric_columns:
                if column in features.columns:
                    col_data = features[column].dropna()
                    if len(col_data) > 0:
                        summary["feature_stats"][column] = {
                            "count": len(col_data),
                            "mean": float(col_data.mean()),
                            "std": float(col_data.std()),
                            "min": float(col_data.min()),
                            "max": float(col_data.max())
                        }
            
            # Data completeness
            total_rows = len(features)
            for column in features.columns:
                valid_count = features[column].notna().sum()
                summary["data_completeness"][column] = {
                    "valid_count": int(valid_count),
                    "completeness_percentage": round((valid_count / total_rows) * 100, 2)
                }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating feature summary: {str(e)}")
            return {"error": str(e)}