"""
Portfolio monitoring module for diversification controls.
Implements portfolio diversification and concentration limits.
"""

import logging
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict

from ..interfaces.trading_interfaces import Position, TradingSignal


@dataclass
class PortfolioConfig:
    """Configuration for portfolio diversification parameters."""
    max_positions: int = 30  # Maximum number of positions
    min_positions: int = 20  # Minimum number of positions for full diversification
    max_position_pct: float = 0.05  # Maximum 5% in any single stock
    max_sector_pct: float = 0.20  # Maximum 20% in any single sector
    max_market_cap_pct: float = 0.60  # Maximum 60% in any market cap category
    sentiment_position_adjustment: float = 0.20  # ±20% position size adjustment
    rebalance_threshold: float = 0.02  # 2% threshold for rebalancing


@dataclass
class PortfolioMetrics:
    """Portfolio diversification metrics."""
    total_positions: int
    total_exposure: float
    largest_position_pct: float
    concentration_risk: float
    diversification_score: float
    sector_distribution: Dict[str, float]
    market_cap_distribution: Dict[str, float]


class PortfolioMonitor:
    """
    Monitors portfolio diversification and implements concentration limits.
    
    Implements:
    - Position tracking across 20-30 stocks
    - Position rejection for overweight stocks
    - Sentiment-based position size adjustments (±20%)
    - Sector and market cap diversification controls
    """
    
    def __init__(self, config: Optional[PortfolioConfig] = None):
        """
        Initialize portfolio monitor with configuration.
        
        Args:
            config: Portfolio configuration parameters
        """
        self.config = config or PortfolioConfig()
        self.logger = logging.getLogger(__name__)
        
        # Stock metadata for diversification (would be loaded from external source)
        self.stock_metadata = self._initialize_stock_metadata()
        
    def _initialize_stock_metadata(self) -> Dict[str, Dict[str, str]]:
        """
        Initialize stock metadata for diversification analysis.
        In production, this would be loaded from external data source.
        
        Returns:
            Dictionary mapping stock symbols to metadata
        """
        # Sample metadata for major Indian stocks
        # In production, this would be loaded from a comprehensive database
        return {
            "RELIANCE": {"sector": "Energy", "market_cap": "Large"},
            "TCS": {"sector": "IT", "market_cap": "Large"},
            "INFY": {"sector": "IT", "market_cap": "Large"},
            "HDFCBANK": {"sector": "Banking", "market_cap": "Large"},
            "ICICIBANK": {"sector": "Banking", "market_cap": "Large"},
            "HINDUNILVR": {"sector": "FMCG", "market_cap": "Large"},
            "ITC": {"sector": "FMCG", "market_cap": "Large"},
            "SBIN": {"sector": "Banking", "market_cap": "Large"},
            "BHARTIARTL": {"sector": "Telecom", "market_cap": "Large"},
            "ASIANPAINT": {"sector": "Paints", "market_cap": "Large"},
            "MARUTI": {"sector": "Auto", "market_cap": "Large"},
            "BAJFINANCE": {"sector": "NBFC", "market_cap": "Large"},
            "WIPRO": {"sector": "IT", "market_cap": "Large"},
            "TECHM": {"sector": "IT", "market_cap": "Large"},
            "ULTRACEMCO": {"sector": "Cement", "market_cap": "Large"},
            "TITAN": {"sector": "Jewelry", "market_cap": "Large"},
            # Add more stocks as needed
        }
    
    def check_position_limits(self, new_signal: TradingSignal, current_positions: List[Position],
                            portfolio_value: float) -> Tuple[bool, str]:
        """
        Check if new position violates diversification limits.
        
        Args:
            new_signal: New trading signal to validate
            current_positions: List of current open positions
            portfolio_value: Current portfolio value
            
        Returns:
            Tuple of (is_allowed, reason)
            
        Requirements: 7.3, 7.4 - Portfolio diversification limits
        """
        try:
            # Check maximum number of positions
            if len(current_positions) >= self.config.max_positions:
                return False, f"Maximum positions limit reached ({self.config.max_positions})"
            
            # Check if we already have a position in this stock
            existing_position = self._find_existing_position(new_signal.symbol, current_positions)
            if existing_position:
                return False, f"Already have position in {new_signal.symbol}"
            
            # Calculate proposed position size with sentiment adjustment
            adjusted_size = self._calculate_sentiment_adjusted_size(
                new_signal.risk_adjusted_size, new_signal.sentiment_score
            )
            
            # Check individual position size limit
            if adjusted_size > self.config.max_position_pct:
                return False, f"Position size {adjusted_size:.3f} exceeds limit {self.config.max_position_pct:.3f}"
            
            # Check sector concentration
            sector_ok, sector_msg = self._check_sector_limits(
                new_signal.symbol, adjusted_size, current_positions, portfolio_value
            )
            if not sector_ok:
                return False, sector_msg
            
            # Check market cap concentration
            market_cap_ok, market_cap_msg = self._check_market_cap_limits(
                new_signal.symbol, adjusted_size, current_positions, portfolio_value
            )
            if not market_cap_ok:
                return False, market_cap_msg
            
            return True, "Position within diversification limits"
            
        except Exception as e:
            self.logger.error(f"Error checking position limits: {e}")
            return False, f"Error validating position: {e}"
    
    def _find_existing_position(self, symbol: str, positions: List[Position]) -> Optional[Position]:
        """Find existing position for a symbol."""
        for position in positions:
            if position.symbol == symbol and position.status == 'open':
                return position
        return None
    
    def _calculate_sentiment_adjusted_size(self, base_size: float, sentiment_score: float) -> float:
        """
        Calculate position size with sentiment adjustment.
        
        Args:
            base_size: Base position size
            sentiment_score: Sentiment score between -1 and +1
            
        Returns:
            Adjusted position size
            
        Requirements: 7.6 - Sentiment-based position size adjustments (±20%)
        """
        # Clamp sentiment score to valid range
        sentiment_score = max(-1.0, min(1.0, sentiment_score))
        
        # Calculate adjustment factor
        adjustment = sentiment_score * self.config.sentiment_position_adjustment
        
        # Apply adjustment
        adjusted_size = base_size * (1 + adjustment)
        
        # Ensure within reasonable bounds
        adjusted_size = max(0.005, min(0.1, adjusted_size))  # 0.5% to 10%
        
        return adjusted_size
    
    def _check_sector_limits(self, symbol: str, position_size: float, 
                           current_positions: List[Position], portfolio_value: float) -> Tuple[bool, str]:
        """
        Check sector concentration limits.
        
        Args:
            symbol: Stock symbol
            position_size: Proposed position size
            current_positions: Current positions
            portfolio_value: Portfolio value
            
        Returns:
            Tuple of (is_within_limits, message)
        """
        try:
            # Get sector for new stock
            if symbol not in self.stock_metadata:
                # Allow unknown stocks but log warning
                self.logger.warning(f"No metadata found for {symbol}, allowing position")
                return True, "Unknown stock metadata"
            
            new_sector = self.stock_metadata[symbol]["sector"]
            
            # Calculate current sector exposure
            sector_exposure = self._calculate_sector_exposure(current_positions, portfolio_value)
            
            # Add proposed position to sector exposure
            current_sector_exposure = sector_exposure.get(new_sector, 0.0)
            new_sector_exposure = current_sector_exposure + position_size
            
            # Check sector limit
            if new_sector_exposure > self.config.max_sector_pct:
                return False, (f"Sector {new_sector} exposure {new_sector_exposure:.3f} "
                             f"would exceed limit {self.config.max_sector_pct:.3f}")
            
            return True, "Sector limits OK"
            
        except Exception as e:
            self.logger.error(f"Error checking sector limits: {e}")
            return True, "Error checking sector limits, allowing position"
    
    def _check_market_cap_limits(self, symbol: str, position_size: float,
                               current_positions: List[Position], portfolio_value: float) -> Tuple[bool, str]:
        """
        Check market cap concentration limits.
        
        Args:
            symbol: Stock symbol
            position_size: Proposed position size
            current_positions: Current positions
            portfolio_value: Portfolio value
            
        Returns:
            Tuple of (is_within_limits, message)
        """
        try:
            # Get market cap for new stock
            if symbol not in self.stock_metadata:
                return True, "Unknown stock metadata"
            
            new_market_cap = self.stock_metadata[symbol]["market_cap"]
            
            # Calculate current market cap exposure
            market_cap_exposure = self._calculate_market_cap_exposure(current_positions, portfolio_value)
            
            # Add proposed position to market cap exposure
            current_market_cap_exposure = market_cap_exposure.get(new_market_cap, 0.0)
            new_market_cap_exposure = current_market_cap_exposure + position_size
            
            # Check market cap limit
            if new_market_cap_exposure > self.config.max_market_cap_pct:
                return False, (f"Market cap {new_market_cap} exposure {new_market_cap_exposure:.3f} "
                             f"would exceed limit {self.config.max_market_cap_pct:.3f}")
            
            return True, "Market cap limits OK"
            
        except Exception as e:
            self.logger.error(f"Error checking market cap limits: {e}")
            return True, "Error checking market cap limits, allowing position"
    
    def _calculate_sector_exposure(self, positions: List[Position], portfolio_value: float) -> Dict[str, float]:
        """Calculate current sector exposure."""
        sector_exposure = defaultdict(float)
        
        for position in positions:
            if position.status == 'open' and position.symbol in self.stock_metadata:
                sector = self.stock_metadata[position.symbol]["sector"]
                exposure = position.current_value / portfolio_value
                sector_exposure[sector] += exposure
        
        return dict(sector_exposure)
    
    def _calculate_market_cap_exposure(self, positions: List[Position], portfolio_value: float) -> Dict[str, float]:
        """Calculate current market cap exposure."""
        market_cap_exposure = defaultdict(float)
        
        for position in positions:
            if position.status == 'open' and position.symbol in self.stock_metadata:
                market_cap = self.stock_metadata[position.symbol]["market_cap"]
                exposure = position.current_value / portfolio_value
                market_cap_exposure[market_cap] += exposure
        
        return dict(market_cap_exposure)
    
    def get_portfolio_metrics(self, positions: List[Position], portfolio_value: float) -> PortfolioMetrics:
        """
        Calculate comprehensive portfolio diversification metrics.
        
        Args:
            positions: List of current positions
            portfolio_value: Current portfolio value
            
        Returns:
            Portfolio diversification metrics
        """
        try:
            open_positions = [p for p in positions if p.status == 'open']
            
            if not open_positions:
                return PortfolioMetrics(
                    total_positions=0,
                    total_exposure=0.0,
                    largest_position_pct=0.0,
                    concentration_risk=0.0,
                    diversification_score=0.0,
                    sector_distribution={},
                    market_cap_distribution={}
                )
            
            # Calculate basic metrics
            total_positions = len(open_positions)
            total_exposure = sum(p.current_value for p in open_positions) / portfolio_value
            
            # Calculate position sizes
            position_sizes = [p.current_value / portfolio_value for p in open_positions]
            largest_position_pct = max(position_sizes)
            
            # Calculate concentration risk (Herfindahl-Hirschman Index)
            concentration_risk = sum(size ** 2 for size in position_sizes)
            
            # Calculate diversification score (inverse of concentration)
            diversification_score = 1.0 / concentration_risk if concentration_risk > 0 else 0.0
            
            # Calculate sector and market cap distributions
            sector_distribution = self._calculate_sector_exposure(open_positions, portfolio_value)
            market_cap_distribution = self._calculate_market_cap_exposure(open_positions, portfolio_value)
            
            return PortfolioMetrics(
                total_positions=total_positions,
                total_exposure=total_exposure,
                largest_position_pct=largest_position_pct,
                concentration_risk=concentration_risk,
                diversification_score=diversification_score,
                sector_distribution=sector_distribution,
                market_cap_distribution=market_cap_distribution
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio metrics: {e}")
            return PortfolioMetrics(
                total_positions=0,
                total_exposure=0.0,
                largest_position_pct=0.0,
                concentration_risk=0.0,
                diversification_score=0.0,
                sector_distribution={},
                market_cap_distribution={}
            )
    
    def suggest_rebalancing(self, positions: List[Position], portfolio_value: float) -> List[str]:
        """
        Suggest portfolio rebalancing actions.
        
        Args:
            positions: Current positions
            portfolio_value: Portfolio value
            
        Returns:
            List of rebalancing suggestions
        """
        suggestions = []
        
        try:
            metrics = self.get_portfolio_metrics(positions, portfolio_value)
            
            # Check if portfolio is too concentrated
            if metrics.largest_position_pct > self.config.max_position_pct + self.config.rebalance_threshold:
                suggestions.append(f"Reduce largest position (currently {metrics.largest_position_pct:.2%})")
            
            # Check sector concentration
            for sector, exposure in metrics.sector_distribution.items():
                if exposure > self.config.max_sector_pct + self.config.rebalance_threshold:
                    suggestions.append(f"Reduce {sector} sector exposure (currently {exposure:.2%})")
            
            # Check market cap concentration
            for market_cap, exposure in metrics.market_cap_distribution.items():
                if exposure > self.config.max_market_cap_pct + self.config.rebalance_threshold:
                    suggestions.append(f"Reduce {market_cap} cap exposure (currently {exposure:.2%})")
            
            # Check if portfolio is under-diversified
            if metrics.total_positions < self.config.min_positions:
                suggestions.append(f"Add more positions for diversification (currently {metrics.total_positions})")
            
            return suggestions
            
        except Exception as e:
            self.logger.error(f"Error generating rebalancing suggestions: {e}")
            return ["Error generating suggestions"]
    
    def get_diversification_score(self, positions: List[Position], portfolio_value: float) -> float:
        """
        Calculate overall diversification score (0-1, higher is better).
        
        Args:
            positions: Current positions
            portfolio_value: Portfolio value
            
        Returns:
            Diversification score between 0 and 1
        """
        try:
            metrics = self.get_portfolio_metrics(positions, portfolio_value)
            
            # Normalize diversification score to 0-1 range
            # Perfect diversification would be 1/max_positions
            ideal_diversification = 1.0 / self.config.max_positions
            normalized_score = min(1.0, metrics.diversification_score / ideal_diversification)
            
            return normalized_score
            
        except Exception as e:
            self.logger.error(f"Error calculating diversification score: {e}")
            return 0.0
    
    def update_stock_metadata(self, metadata: Dict[str, Dict[str, str]]) -> None:
        """
        Update stock metadata for diversification analysis.
        
        Args:
            metadata: Dictionary mapping symbols to metadata
        """
        self.stock_metadata.update(metadata)
        self.logger.info(f"Updated metadata for {len(metadata)} stocks")