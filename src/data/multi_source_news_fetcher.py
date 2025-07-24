"""
Enhanced news data fetcher that combines multiple sources for more comprehensive sentiment data.
"""

import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import requests
import os
import json
from pathlib import Path

from src.data.news_data_fetcher import NewsDataFetcher
from src.data.data_storage import DataStorage


class MultiSourceNewsFetcher:
    """
    Enhanced news data fetcher that combines multiple sources.
    
    Features:
    - Multiple news sources (YFinance, Alpha Vantage, NewsAPI, etc.)
    - Fallback mechanism when primary source fails
    - Deduplication of news from multiple sources
    - Caching to reduce API calls
    """
    
    def __init__(self, 
                 data_storage: Optional[DataStorage] = None,
                 config_path: str = "config/news_api_keys.json"):
        """
        Initialize multi-source news fetcher.
        
        Args:
            data_storage: Data storage for saving fetched news
            config_path: Path to API keys configuration file
        """
        self.data_storage = data_storage or DataStorage()
        self.logger = logging.getLogger(__name__)
        
        # Initialize default news fetcher (YFinance)
        self.default_fetcher = NewsDataFetcher(data_storage=self.data_storage)
        
        # Load API keys
        self.api_keys = self._load_api_keys(config_path)
        
        # Initialize additional news sources
        self.sources = {
            "yfinance": self.default_fetcher,
            "alpha_vantage": self._has_key("alpha_vantage"),
            "newsapi": self._has_key("newsapi"),
            "finnhub": self._has_key("finnhub")
        }
        
        self.logger.info(f"MultiSourceNewsFetcher initialized with sources: {list(self.sources.keys())}")
    
    def _load_api_keys(self, config_path: str) -> Dict:
        """Load API keys from configuration file."""
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    return json.load(f)
            else:
                self.logger.warning(f"API keys configuration file not found: {config_path}")
                return {}
        except Exception as e:
            self.logger.error(f"Error loading API keys: {str(e)}")
            return {}
    
    def _has_key(self, source: str) -> bool:
        """Check if API key exists for a source."""
        return source in self.api_keys and self.api_keys[source] != ""
    
    def fetch_news_for_symbol(self, 
                             symbol: str, 
                             start_date: str, 
                             end_date: str) -> List[Dict]:
        """
        Fetch news headlines from multiple sources.
        
        Args:
            symbol: Stock symbol (e.g., 'RELIANCE', 'TCS')
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            List of news dictionaries with filtered and cleaned data
        """
        self.logger.info(f"Fetching news for {symbol} from multiple sources")
        
        # Try to load from cache first
        cached_news = self._load_from_cache(symbol, start_date, end_date)
        if cached_news:
            self.logger.info(f"Loaded {len(cached_news)} news items from cache for {symbol}")
            return cached_news
        
        all_news = []
        
        # Try YFinance first (default source)
        yfinance_news = self.default_fetcher.fetch_news_for_symbol(symbol, start_date, end_date)
        if yfinance_news:
            self.logger.info(f"Found {len(yfinance_news)} news items from YFinance for {symbol}")
            all_news.extend(yfinance_news)
        
        # Try Alpha Vantage if available
        if self.sources["alpha_vantage"]:
            alpha_vantage_news = self._fetch_from_alpha_vantage(symbol, start_date, end_date)
            if alpha_vantage_news:
                self.logger.info(f"Found {len(alpha_vantage_news)} news items from Alpha Vantage for {symbol}")
                all_news.extend(alpha_vantage_news)
        
        # Try NewsAPI if available
        if self.sources["newsapi"]:
            newsapi_news = self._fetch_from_newsapi(symbol, start_date, end_date)
            if newsapi_news:
                self.logger.info(f"Found {len(newsapi_news)} news items from NewsAPI for {symbol}")
                all_news.extend(newsapi_news)
        
        # Try Finnhub if available
        if self.sources["finnhub"]:
            finnhub_news = self._fetch_from_finnhub(symbol, start_date, end_date)
            if finnhub_news:
                self.logger.info(f"Found {len(finnhub_news)} news items from Finnhub for {symbol}")
                all_news.extend(finnhub_news)
        
        # Deduplicate news items
        deduplicated_news = self._deduplicate_news(all_news)
        
        # Save to cache
        if deduplicated_news:
            self._save_to_cache(symbol, start_date, end_date, deduplicated_news)
        
        self.logger.info(f"Total news items after deduplication: {len(deduplicated_news)}")
        return deduplicated_news
    
    def _fetch_from_alpha_vantage(self, symbol: str, start_date: str, end_date: str) -> List[Dict]:
        """Fetch news from Alpha Vantage API."""
        try:
            api_key = self.api_keys.get("alpha_vantage", "")
            if not api_key:
                return []
            
            # Remove .NS or .BO suffix for Alpha Vantage
            clean_symbol = symbol.replace('.NS', '').replace('.BO', '')
            
            # Alpha Vantage news endpoint
            url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={clean_symbol}&apikey={api_key}"
            
            response = requests.get(url)
            if response.status_code != 200:
                self.logger.warning(f"Alpha Vantage API error: {response.status_code}")
                return []
            
            data = response.json()
            if "feed" not in data:
                self.logger.warning("No news feed in Alpha Vantage response")
                return []
            
            # Convert to standard format
            news_items = []
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            
            for item in data["feed"]:
                try:
                    # Parse time
                    time_published = item.get("time_published", "")
                    if time_published:
                        # Format: YYYYMMDDTHHMMSS
                        dt = datetime.strptime(time_published, "%Y%m%dT%H%M%S")
                        
                        # Check if within date range
                        if start_dt <= dt <= end_dt:
                            news_items.append({
                                'symbol': clean_symbol,
                                'title': item.get("title", ""),
                                'summary': item.get("summary", ""),
                                'publisher': item.get("source", "Alpha Vantage"),
                                'publish_time': int(dt.timestamp()),
                                'url': item.get("url", ""),
                                'source': "alpha_vantage"
                            })
                except Exception as e:
                    self.logger.warning(f"Error processing Alpha Vantage news item: {str(e)}")
            
            return news_items
            
        except Exception as e:
            self.logger.error(f"Error fetching news from Alpha Vantage: {str(e)}")
            return []
    
    def _fetch_from_newsapi(self, symbol: str, start_date: str, end_date: str) -> List[Dict]:
        """Fetch news from NewsAPI."""
        try:
            api_key = self.api_keys.get("newsapi", "")
            if not api_key:
                return []
            
            # Remove .NS or .BO suffix
            clean_symbol = symbol.replace('.NS', '').replace('.BO', '')
            
            # For Indian stocks, add company name or NSE/BSE for better results
            # This requires a mapping of symbols to company names
            company_name = self._get_company_name(clean_symbol)
            query = f"{clean_symbol} {company_name} stock"
            
            # NewsAPI endpoint
            url = (
                f"https://newsapi.org/v2/everything?"
                f"q={query}&"
                f"from={start_date}&"
                f"to={end_date}&"
                f"language=en&"
                f"sortBy=publishedAt&"
                f"apiKey={api_key}"
            )
            
            response = requests.get(url)
            if response.status_code != 200:
                self.logger.warning(f"NewsAPI error: {response.status_code}")
                return []
            
            data = response.json()
            if "articles" not in data:
                self.logger.warning("No articles in NewsAPI response")
                return []
            
            # Convert to standard format
            news_items = []
            
            for item in data["articles"]:
                try:
                    # Parse time
                    time_published = item.get("publishedAt", "")
                    if time_published:
                        dt = datetime.fromisoformat(time_published.replace("Z", "+00:00"))
                        
                        news_items.append({
                            'symbol': clean_symbol,
                            'title': item.get("title", ""),
                            'summary': item.get("description", ""),
                            'publisher': item.get("source", {}).get("name", "NewsAPI"),
                            'publish_time': int(dt.timestamp()),
                            'url': item.get("url", ""),
                            'source': "newsapi"
                        })
                except Exception as e:
                    self.logger.warning(f"Error processing NewsAPI item: {str(e)}")
            
            return news_items
            
        except Exception as e:
            self.logger.error(f"Error fetching news from NewsAPI: {str(e)}")
            return []
    
    def _fetch_from_finnhub(self, symbol: str, start_date: str, end_date: str) -> List[Dict]:
        """Fetch news from Finnhub API."""
        try:
            api_key = self.api_keys.get("finnhub", "")
            if not api_key:
                return []
            
            # Remove .NS or .BO suffix
            clean_symbol = symbol.replace('.NS', '').replace('.BO', '')
            
            # Convert dates to timestamps
            start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp())
            end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp())
            
            # Finnhub endpoint
            url = (
                f"https://finnhub.io/api/v1/company-news?"
                f"symbol={clean_symbol}&"
                f"from={start_date}&"
                f"to={end_date}&"
                f"token={api_key}"
            )
            
            response = requests.get(url)
            if response.status_code != 200:
                self.logger.warning(f"Finnhub API error: {response.status_code}")
                return []
            
            data = response.json()
            if not data or not isinstance(data, list):
                self.logger.warning("Invalid response from Finnhub")
                return []
            
            # Convert to standard format
            news_items = []
            
            for item in data:
                try:
                    # Get timestamp
                    timestamp = item.get("datetime", 0)
                    
                    if timestamp >= start_ts and timestamp <= end_ts:
                        news_items.append({
                            'symbol': clean_symbol,
                            'title': item.get("headline", ""),
                            'summary': item.get("summary", ""),
                            'publisher': item.get("source", "Finnhub"),
                            'publish_time': timestamp,
                            'url': item.get("url", ""),
                            'source': "finnhub"
                        })
                except Exception as e:
                    self.logger.warning(f"Error processing Finnhub news item: {str(e)}")
            
            return news_items
            
        except Exception as e:
            self.logger.error(f"Error fetching news from Finnhub: {str(e)}")
            return []
    
    def _deduplicate_news(self, news_items: List[Dict]) -> List[Dict]:
        """Deduplicate news items based on title similarity."""
        if not news_items:
            return []
        
        # Use a simple approach: keep track of seen titles
        seen_titles = set()
        deduplicated_news = []
        
        for item in news_items:
            title = item.get('title', '').strip().lower()
            
            # Skip empty titles
            if not title:
                continue
            
            # Simple deduplication: check if we've seen this title before
            if title not in seen_titles:
                seen_titles.add(title)
                deduplicated_news.append(item)
        
        return deduplicated_news
    
    def _get_company_name(self, symbol: str) -> str:
        """Get company name for a symbol."""
        # This is a placeholder - in a real implementation, you would have a mapping
        # of symbols to company names, or use an API to get this information
        
        # For now, return empty string
        return ""
    
    def _load_from_cache(self, symbol: str, start_date: str, end_date: str) -> List[Dict]:
        """Load news from cache if available and not expired."""
        try:
            cache_dir = Path("data/cache/news")
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            cache_file = cache_dir / f"{symbol.replace('.', '_')}_{start_date}_{end_date}.json"
            
            if cache_file.exists():
                # Check if cache is expired (older than 24 hours)
                if (datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)).total_seconds() < 86400:
                    with open(cache_file, 'r') as f:
                        return json.load(f)
            
            return []
            
        except Exception as e:
            self.logger.warning(f"Error loading from cache: {str(e)}")
            return []
    
    def _save_to_cache(self, symbol: str, start_date: str, end_date: str, news_items: List[Dict]):
        """Save news to cache."""
        try:
            cache_dir = Path("data/cache/news")
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            cache_file = cache_dir / f"{symbol.replace('.', '_')}_{start_date}_{end_date}.json"
            
            with open(cache_file, 'w') as f:
                json.dump(news_items, f, indent=2)
                
        except Exception as e:
            self.logger.warning(f"Error saving to cache: {str(e)}")