"""
Position sizing module for risk management.
Implements position sizing logic to limit trades to 1-2% of capital.
"""

import logging
from typing import Dict, Optional
from dataclasses import dataclass

from ..interfaces.trading_interfaces import TradingSignal


@dataclass
class PositionSizingConfig:
    """Configuration for position sizing parameters."""
    min_position_pct: float = 0.01  # 1% minimum position size
    max_position_pct: float = 0.02  # 2% maximum position size
    base_position_pct: float = 0.015  # 1.5% base position size
    sentiment_adjustment_pct: float = 0.20  # ±20% sentiment adjustment
    min_trade_value: float = 1000.0  # Minimum trade value in INR
    max_trade_value: float = 50000.0  # Maximum trade value in INR


class PositionSizer:
    """
    Handles position sizing calculations based on risk management rules.
    
    Implements:
    - Base position sizing of 1-2% of capital
    - Sentiment-based adjustments (±20%)
    - Minimum and maximum trade value limits
    """
    
    def __init__(self, config: Optional[PositionSizingConfig] = None):
        """
        Initialize position sizer with configuration.
        
        Args:
            config: Position sizing configuration parameters
        """
        self.config = config or PositionSizingConfig()
        self.logger = logging.getLogger(__name__)
        
    def calculate_position_size(self, signal: TradingSignal, portfolio_value: float, 
                              current_price: float) -> float:
        """
        Calculate appropriate position size based on risk rules.
        
        Args:
            signal: Trading signal containing confidence and sentiment
            portfolio_value: Current total portfolio value
            current_price: Current stock price
            
        Returns:
            Position size as percentage of portfolio value (0.0 to 1.0)
            
        Requirements: 7.1, 7.2 - Position sizing 1-2% of capital with sentiment adjustments
        """
        try:
            # Start with base position size
            base_size = self.config.base_position_pct
            
            # Apply sentiment-based adjustment
            sentiment_adjustment = self._calculate_sentiment_adjustment(signal.sentiment_score)
            adjusted_size = base_size * (1 + sentiment_adjustment)
            
            # Apply confidence-based adjustment
            confidence_adjustment = self._calculate_confidence_adjustment(signal.confidence)
            final_size = adjusted_size * confidence_adjustment
            
            # Ensure within min/max bounds
            final_size = max(self.config.min_position_pct, 
                           min(self.config.max_position_pct, final_size))
            
            # Check trade value limits
            trade_value = final_size * portfolio_value
            if trade_value < self.config.min_trade_value:
                final_size = self.config.min_trade_value / portfolio_value
            elif trade_value > self.config.max_trade_value:
                final_size = self.config.max_trade_value / portfolio_value
            
            # Calculate number of shares
            shares = int((final_size * portfolio_value) / current_price)
            if shares == 0:  # Ensure at least 1 share if trade value allows
                shares = 1
            actual_size = (shares * current_price) / portfolio_value
            
            self.logger.info(f"Position sizing for {signal.symbol}: "
                           f"base={base_size:.3f}, sentiment_adj={sentiment_adjustment:.3f}, "
                           f"confidence_adj={confidence_adjustment:.3f}, final={actual_size:.3f}")
            
            return actual_size
            
        except Exception as e:
            self.logger.error(f"Error calculating position size for {signal.symbol}: {e}")
            return self.config.min_position_pct
    
    def _calculate_sentiment_adjustment(self, sentiment_score: float) -> float:
        """
        Calculate position size adjustment based on sentiment score.
        
        Args:
            sentiment_score: Sentiment score between -1 and +1
            
        Returns:
            Adjustment factor between -0.2 and +0.2 (±20%)
        """
        # Clamp sentiment score to valid range
        sentiment_score = max(-1.0, min(1.0, sentiment_score))
        
        # Convert to adjustment factor
        adjustment = sentiment_score * self.config.sentiment_adjustment_pct
        
        return adjustment
    
    def _calculate_confidence_adjustment(self, confidence: float) -> float:
        """
        Calculate position size adjustment based on signal confidence.
        
        Args:
            confidence: Signal confidence between 0 and 1
            
        Returns:
            Adjustment factor between 0.5 and 1.0
        """
        # Clamp confidence to valid range
        confidence = max(0.0, min(1.0, confidence))
        
        # Scale confidence to adjustment factor (0.5 to 1.0)
        # Low confidence reduces position size, high confidence maintains it
        adjustment = 0.5 + (confidence * 0.5)
        
        return adjustment
    
    def get_max_shares(self, portfolio_value: float, stock_price: float) -> int:
        """
        Get maximum number of shares that can be purchased.
        
        Args:
            portfolio_value: Current portfolio value
            stock_price: Current stock price
            
        Returns:
            Maximum number of shares
        """
        max_trade_value = portfolio_value * self.config.max_position_pct
        max_shares = int(max_trade_value / stock_price)
        
        return max_shares
    
    def validate_position_size(self, position_size: float, portfolio_value: float) -> bool:
        """
        Validate if position size is within acceptable limits.
        
        Args:
            position_size: Position size as percentage of portfolio
            portfolio_value: Current portfolio value
            
        Returns:
            True if position size is valid, False otherwise
        """
        if position_size < self.config.min_position_pct:
            self.logger.warning(f"Position size {position_size:.3f} below minimum {self.config.min_position_pct}")
            return False
            
        if position_size > self.config.max_position_pct:
            self.logger.warning(f"Position size {position_size:.3f} above maximum {self.config.max_position_pct}")
            return False
            
        trade_value = position_size * portfolio_value
        if trade_value < self.config.min_trade_value:
            self.logger.warning(f"Trade value {trade_value:.2f} below minimum {self.config.min_trade_value}")
            return False
            
        if trade_value > self.config.max_trade_value:
            self.logger.warning(f"Trade value {trade_value:.2f} above maximum {self.config.max_trade_value}")
            return False
            
        return True