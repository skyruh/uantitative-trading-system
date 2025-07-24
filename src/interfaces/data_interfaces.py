"""
Core data interfaces for the quantitative trading system.
Defines contracts for data sources, storage, and processing components.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from datetime import datetime
import pandas as pd


class IDataSource(ABC):
    """Interface for external data sources."""
    
    @abstractmethod
    def fetch_stock_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch OHLCV data for a stock symbol."""
        pass
    
    @abstractmethod
    def fetch_news_data(self, symbol: str, start_date: str, end_date: str) -> List[Dict]:
        """Fetch news headlines for sentiment analysis."""
        pass


class IDataStorage(ABC):
    """Interface for data storage operations."""
    
    @abstractmethod
    def save_stock_data(self, symbol: str, data: pd.DataFrame) -> bool:
        """Save stock data to storage."""
        pass
    
    @abstractmethod
    def load_stock_data(self, symbol: str) -> pd.DataFrame:
        """Load stock data from storage."""
        pass
    
    @abstractmethod
    def save_news_data(self, symbol: str, news: List[Dict]) -> bool:
        """Save news data to storage."""
        pass
    
    @abstractmethod
    def load_news_data(self, symbol: str) -> List[Dict]:
        """Load news data from storage."""
        pass


class IDataProcessor(ABC):
    """Interface for data processing operations."""
    
    @abstractmethod
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate data."""
        pass
    
    @abstractmethod
    def normalize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Normalize data features."""
        pass
    
    @abstractmethod
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate data integrity and completeness."""
        pass


class IFeatureEngineer(ABC):
    """Interface for feature engineering operations."""
    
    @abstractmethod
    def calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators from price data."""
        pass
    
    @abstractmethod
    def calculate_sentiment_scores(self, news_data: List[Dict]) -> List[float]:
        """Calculate sentiment scores from news headlines."""
        pass
    
    @abstractmethod
    def build_features(self, price_data: pd.DataFrame, sentiment_data: List[float]) -> pd.DataFrame:
        """Combine all features for model training."""
        pass