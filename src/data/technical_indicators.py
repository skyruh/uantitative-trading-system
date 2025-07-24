"""
Technical indicators module for quantitative trading system.
Implements RSI, Simple Moving Average, and Bollinger Bands calculations.
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """
    Class for calculating technical indicators from price data.
    Implements RSI, SMA, and Bollinger Bands with proper error handling.
    """
    
    @staticmethod
    def calculate_rsi(data: pd.DataFrame, period: int = 14, price_col: str = 'close') -> pd.Series:
        """
        Calculate Relative Strength Index (RSI).
        
        Args:
            data: DataFrame with price data
            period: RSI calculation period (default: 14)
            price_col: Column name for price data (default: 'close')
            
        Returns:
            Series with RSI values
            
        Raises:
            ValueError: If insufficient data or invalid parameters
        """
        if data.empty:
            raise ValueError("Input data is empty")
            
        if price_col not in data.columns:
            raise ValueError(f"Price column '{price_col}' not found in data")
            
        if len(data) < period + 1:
            logger.warning(f"Insufficient data for RSI calculation. Need at least {period + 1} rows, got {len(data)}")
            return pd.Series(index=data.index, dtype=float)
            
        try:
            # Calculate price changes
            delta = data[price_col].diff()
            
            # Separate gains and losses
            gains = delta.where(delta > 0, 0)
            losses = -delta.where(delta < 0, 0)
            
            # Calculate average gains and losses using exponential moving average
            avg_gains = gains.ewm(span=period, adjust=False).mean()
            avg_losses = losses.ewm(span=period, adjust=False).mean()
            
            # Calculate RSI
            rs = avg_gains / avg_losses
            rsi = 100 - (100 / (1 + rs))
            
            # Set first 'period' values to NaN as they're not reliable
            rsi.iloc[:period] = np.nan
            
            return rsi
            
        except Exception as e:
            logger.error(f"Error calculating RSI: {str(e)}")
            return pd.Series(index=data.index, dtype=float)
    
    @staticmethod
    def calculate_sma(data: pd.DataFrame, period: int = 50, price_col: str = 'close') -> pd.Series:
        """
        Calculate Simple Moving Average (SMA).
        
        Args:
            data: DataFrame with price data
            period: SMA calculation period (default: 50)
            price_col: Column name for price data (default: 'close')
            
        Returns:
            Series with SMA values
            
        Raises:
            ValueError: If insufficient data or invalid parameters
        """
        if data.empty:
            raise ValueError("Input data is empty")
            
        if price_col not in data.columns:
            raise ValueError(f"Price column '{price_col}' not found in data")
            
        if len(data) < period:
            logger.warning(f"Insufficient data for SMA calculation. Need at least {period} rows, got {len(data)}")
            return pd.Series(index=data.index, dtype=float)
            
        try:
            # Calculate simple moving average
            sma = data[price_col].rolling(window=period, min_periods=period).mean()
            
            return sma
            
        except Exception as e:
            logger.error(f"Error calculating SMA: {str(e)}")
            return pd.Series(index=data.index, dtype=float)
    
    @staticmethod
    def calculate_bollinger_bands(data: pd.DataFrame, period: int = 20, std_dev: float = 2.0, 
                                price_col: str = 'close') -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Bollinger Bands (upper, middle, lower).
        
        Args:
            data: DataFrame with price data
            period: Moving average period (default: 20)
            std_dev: Standard deviation multiplier (default: 2.0)
            price_col: Column name for price data (default: 'close')
            
        Returns:
            Tuple of (upper_band, middle_band, lower_band) Series
            
        Raises:
            ValueError: If insufficient data or invalid parameters
        """
        if data.empty:
            raise ValueError("Input data is empty")
            
        if price_col not in data.columns:
            raise ValueError(f"Price column '{price_col}' not found in data")
            
        if len(data) < period:
            logger.warning(f"Insufficient data for Bollinger Bands calculation. Need at least {period} rows, got {len(data)}")
            empty_series = pd.Series(index=data.index, dtype=float)
            return empty_series, empty_series, empty_series
            
        try:
            # Calculate middle band (SMA)
            middle_band = data[price_col].rolling(window=period, min_periods=period).mean()
            
            # Calculate standard deviation
            rolling_std = data[price_col].rolling(window=period, min_periods=period).std()
            
            # Calculate upper and lower bands
            upper_band = middle_band + (rolling_std * std_dev)
            lower_band = middle_band - (rolling_std * std_dev)
            
            return upper_band, middle_band, lower_band
            
        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands: {str(e)}")
            empty_series = pd.Series(index=data.index, dtype=float)
            return empty_series, empty_series, empty_series
    
    @classmethod
    def calculate_all_indicators(cls, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators and add them to the DataFrame.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added technical indicator columns
        """
        if data.empty:
            logger.warning("Input data is empty, returning empty DataFrame")
            return data.copy()
            
        result = data.copy()
        
        try:
            # Calculate RSI (14-day)
            result['rsi_14'] = cls.calculate_rsi(data, period=14)
            
            # Calculate SMA (50-day)
            result['sma_50'] = cls.calculate_sma(data, period=50)
            
            # Calculate Bollinger Bands (20-day)
            bb_upper, bb_middle, bb_lower = cls.calculate_bollinger_bands(data, period=20)
            result['bb_upper'] = bb_upper
            result['bb_middle'] = bb_middle
            result['bb_lower'] = bb_lower
            
            logger.info(f"Successfully calculated technical indicators for {len(data)} data points")
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {str(e)}")
            
        return result