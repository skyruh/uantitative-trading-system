"""
Signal Generator for the quantitative trading system.
Combines LSTM and DQN outputs to generate validated trading signals.
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Optional, Any
import logging

from src.interfaces.trading_interfaces import ISignalGenerator, TradingSignal
from src.models.lstm_model import LSTMModel
from src.models.dqn_agent import DQNAgent


class SignalGenerator(ISignalGenerator):
    """
    Generates trading signals by combining LSTM price predictions and DQN action selection.
    
    Features:
    - Combines LSTM probability predictions with DQN Q-values
    - Validates signals based on confidence thresholds
    - Incorporates sentiment analysis for signal strength
    - Calculates risk-adjusted position sizes
    """
    
    def __init__(self, lstm_model: Optional[LSTMModel] = None, 
                 dqn_agent: Optional[DQNAgent] = None,
                 min_confidence: float = 0.6,
                 sentiment_weight: float = 0.2,
                 lstm_weight: float = 0.4,
                 dqn_weight: float = 0.4):
        """
        Initialize signal generator.
        
        Args:
            lstm_model: Trained LSTM model for price prediction
            dqn_agent: Trained DQN agent for action selection
            min_confidence: Minimum confidence threshold for signal validation
            sentiment_weight: Weight for sentiment score in signal calculation
            lstm_weight: Weight for LSTM prediction in signal calculation
            dqn_weight: Weight for DQN Q-values in signal calculation
        """
        self.lstm_model = lstm_model
        self.dqn_agent = dqn_agent
        self.min_confidence = min_confidence
        self.sentiment_weight = sentiment_weight
        self.lstm_weight = lstm_weight
        self.dqn_weight = dqn_weight
        
        self.logger = logging.getLogger(__name__)
        
        # Validate weights sum to 1
        total_weight = sentiment_weight + lstm_weight + dqn_weight
        if abs(total_weight - 1.0) > 0.01:
            self.logger.warning(f"Signal weights sum to {total_weight}, not 1.0. Normalizing...")
            self.sentiment_weight /= total_weight
            self.lstm_weight /= total_weight
            self.dqn_weight /= total_weight
        
        self.logger.info(f"SignalGenerator initialized with weights: "
                        f"sentiment={self.sentiment_weight:.2f}, "
                        f"lstm={self.lstm_weight:.2f}, "
                        f"dqn={self.dqn_weight:.2f}")
    
    def set_models(self, lstm_model: LSTMModel, dqn_agent: DQNAgent) -> None:
        """
        Set the LSTM model and DQN agent.
        
        Args:
            lstm_model: Trained LSTM model
            dqn_agent: Trained DQN agent
        """
        self.lstm_model = lstm_model
        self.dqn_agent = dqn_agent
        self.logger.info("Models set successfully")
    
    def _prepare_state_vector(self, market_data: Dict, lstm_prediction: float, 
                             sentiment_score: float) -> np.ndarray:
        """
        Prepare state vector for DQN input.
        
        Args:
            market_data: Current market data
            lstm_prediction: LSTM price movement prediction
            sentiment_score: Sentiment analysis score
            
        Returns:
            State vector for DQN
        """
        try:
            # Extract key features from market data
            features = []
            
            # Price-based features
            features.append(market_data.get('close', 0.0))
            features.append(market_data.get('volume', 0.0))
            features.append(market_data.get('high', 0.0) - market_data.get('low', 0.0))  # Daily range
            
            # Technical indicators
            features.append(market_data.get('rsi_14', 50.0))
            features.append(market_data.get('sma_50', market_data.get('close', 0.0)))
            features.append(market_data.get('bb_upper', 0.0))
            features.append(market_data.get('bb_lower', 0.0))
            
            # Model predictions and sentiment
            features.append(lstm_prediction)
            features.append(sentiment_score)
            
            # Relative position in Bollinger Bands
            close_price = market_data.get('close', 0.0)
            bb_upper = market_data.get('bb_upper', close_price)
            bb_lower = market_data.get('bb_lower', close_price)
            if bb_upper != bb_lower:
                bb_position = (close_price - bb_lower) / (bb_upper - bb_lower)
            else:
                bb_position = 0.5
            features.append(bb_position)
            
            # Normalize features to reasonable ranges
            state_vector = np.array(features, dtype=np.float32)
            
            # Handle any NaN or infinite values
            state_vector = np.nan_to_num(state_vector, nan=0.0, posinf=1.0, neginf=-1.0)
            
            return state_vector
            
        except Exception as e:
            self.logger.error(f"Error preparing state vector: {str(e)}")
            # Return default state vector
            return np.zeros(10, dtype=np.float32)
    
    def _calculate_confidence_score(self, lstm_prediction: float, dqn_q_values: Dict[str, float],
                                   sentiment_score: float, market_data: Dict) -> float:
        """
        Calculate overall confidence score for the trading signal.
        
        Args:
            lstm_prediction: LSTM price movement prediction (0-1)
            dqn_q_values: DQN Q-values for all actions
            sentiment_score: Sentiment analysis score (-1 to 1)
            market_data: Current market data
            
        Returns:
            Confidence score (0-1)
        """
        try:
            # LSTM confidence: how far from 0.5 (neutral)
            lstm_confidence = abs(lstm_prediction - 0.5) * 2  # Scale to 0-1
            
            # DQN confidence: difference between best and second-best Q-values
            q_values_list = list(dqn_q_values.values())
            q_values_sorted = sorted(q_values_list, reverse=True)
            if len(q_values_sorted) >= 2:
                dqn_confidence = min(1.0, (q_values_sorted[0] - q_values_sorted[1]) / 2.0)
            else:
                dqn_confidence = 0.0
            
            # Sentiment confidence: absolute value of sentiment
            sentiment_confidence = abs(sentiment_score)
            
            # Technical indicator confidence (RSI extremes)
            rsi = market_data.get('rsi_14', 50.0)
            if rsi <= 30 or rsi >= 70:
                technical_confidence = 0.8  # High confidence at RSI extremes
            else:
                technical_confidence = 0.4  # Lower confidence in neutral zone
            
            # Combine confidences with weights
            total_confidence = (
                self.lstm_weight * lstm_confidence +
                self.dqn_weight * dqn_confidence +
                self.sentiment_weight * sentiment_confidence +
                0.2 * technical_confidence  # Small weight for technical indicators
            )
            
            # Normalize to 0-1 range
            total_confidence = max(0.0, min(1.0, total_confidence))
            
            self.logger.debug(f"Confidence calculation: LSTM={lstm_confidence:.3f}, "
                            f"DQN={dqn_confidence:.3f}, sentiment={sentiment_confidence:.3f}, "
                            f"technical={technical_confidence:.3f}, total={total_confidence:.3f}")
            
            return total_confidence
            
        except Exception as e:
            self.logger.error(f"Error calculating confidence score: {str(e)}")
            return 0.0
    
    def _determine_action(self, lstm_prediction: float, dqn_q_values: Dict[str, float],
                         sentiment_score: float) -> str:
        """
        Determine the final trading action based on model outputs.
        
        Args:
            lstm_prediction: LSTM price movement prediction (0-1)
            dqn_q_values: DQN Q-values for all actions
            sentiment_score: Sentiment analysis score (-1 to 1)
            
        Returns:
            Trading action ('buy', 'sell', 'hold')
        """
        try:
            # Get DQN's preferred action
            dqn_action = max(dqn_q_values, key=dqn_q_values.get)
            
            # LSTM signal: >0.6 suggests price increase, <0.4 suggests decrease
            if lstm_prediction > 0.6:
                lstm_signal = 'buy'
            elif lstm_prediction < 0.4:
                lstm_signal = 'sell'
            else:
                lstm_signal = 'hold'
            
            # Sentiment signal
            if sentiment_score > 0.3:
                sentiment_signal = 'buy'
            elif sentiment_score < -0.3:
                sentiment_signal = 'sell'
            else:
                sentiment_signal = 'hold'
            
            # Combine signals with majority voting and DQN preference
            signals = [dqn_action, lstm_signal, sentiment_signal]
            
            # Count votes for each action
            buy_votes = signals.count('buy')
            sell_votes = signals.count('sell')
            hold_votes = signals.count('hold')
            
            # Determine final action
            if buy_votes > sell_votes and buy_votes > hold_votes:
                final_action = 'buy'
            elif sell_votes > buy_votes and sell_votes > hold_votes:
                final_action = 'sell'
            else:
                # Default to DQN action in case of tie or hold majority
                final_action = dqn_action
            
            self.logger.debug(f"Action determination: DQN={dqn_action}, LSTM={lstm_signal}, "
                            f"sentiment={sentiment_signal}, final={final_action}")
            
            return final_action
            
        except Exception as e:
            self.logger.error(f"Error determining action: {str(e)}")
            return 'hold'
    
    def _calculate_risk_adjusted_size(self, confidence: float, sentiment_score: float,
                                    base_position_size: float = 0.02) -> float:
        """
        Calculate risk-adjusted position size based on confidence and sentiment.
        
        Args:
            confidence: Signal confidence score (0-1)
            sentiment_score: Sentiment analysis score (-1 to 1)
            base_position_size: Base position size as fraction of capital
            
        Returns:
            Risk-adjusted position size
        """
        try:
            # Adjust size based on confidence
            confidence_multiplier = 0.5 + (confidence * 0.5)  # Range: 0.5 to 1.0
            
            # Adjust size based on sentiment strength (±20% as per requirements)
            sentiment_multiplier = 1.0 + (abs(sentiment_score) * 0.2)
            
            # Calculate final position size
            adjusted_size = base_position_size * confidence_multiplier * sentiment_multiplier
            
            # Ensure position size stays within reasonable bounds (0.5% to 3% of capital)
            adjusted_size = max(0.005, min(0.03, adjusted_size))
            
            self.logger.debug(f"Position size calculation: base={base_position_size:.3f}, "
                            f"confidence_mult={confidence_multiplier:.3f}, "
                            f"sentiment_mult={sentiment_multiplier:.3f}, "
                            f"final={adjusted_size:.3f}")
            
            return adjusted_size
            
        except Exception as e:
            self.logger.error(f"Error calculating risk-adjusted size: {str(e)}")
            return base_position_size
    
    def generate_signal(self, symbol: str, market_data: Dict, 
                       lstm_prediction: float, dqn_q_values: Dict[str, float],
                       sentiment_score: float) -> TradingSignal:
        """
        Generate trading signal based on model outputs.
        
        Args:
            symbol: Stock symbol
            market_data: Current market data
            lstm_prediction: LSTM price movement prediction (0-1)
            dqn_q_values: DQN Q-values for all actions
            sentiment_score: Sentiment analysis score (-1 to 1)
            
        Returns:
            Generated trading signal
        """
        try:
            self.logger.debug(f"Generating signal for {symbol}")
            
            # Calculate confidence score
            confidence = self._calculate_confidence_score(
                lstm_prediction, dqn_q_values, sentiment_score, market_data
            )
            
            # Determine trading action
            action = self._determine_action(lstm_prediction, dqn_q_values, sentiment_score)
            
            # Calculate risk-adjusted position size
            risk_adjusted_size = self._calculate_risk_adjusted_size(confidence, sentiment_score)
            
            # Create trading signal
            signal = TradingSignal(
                symbol=symbol,
                timestamp=datetime.now(),
                action=action,
                confidence=confidence,
                lstm_prediction=lstm_prediction,
                dqn_q_values=dqn_q_values.copy(),
                sentiment_score=sentiment_score,
                risk_adjusted_size=risk_adjusted_size
            )
            
            self.logger.info(f"Signal generated for {symbol}: {action} "
                           f"(confidence={confidence:.3f}, size={risk_adjusted_size:.3f})")
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error generating signal for {symbol}: {str(e)}")
            # Return default hold signal
            return TradingSignal(
                symbol=symbol,
                timestamp=datetime.now(),
                action='hold',
                confidence=0.0,
                lstm_prediction=0.5,
                dqn_q_values={'buy': 0.0, 'sell': 0.0, 'hold': 0.0},
                sentiment_score=0.0,
                risk_adjusted_size=0.01
            )
    
    def validate_signal(self, signal: TradingSignal) -> bool:
        """
        Validate trading signal before execution.
        
        Args:
            signal: Trading signal to validate
            
        Returns:
            True if signal is valid, False otherwise
        """
        try:
            # Check confidence threshold
            if signal.confidence < self.min_confidence:
                self.logger.debug(f"Signal rejected: confidence {signal.confidence:.3f} "
                                f"below threshold {self.min_confidence}")
                return False
            
            # Check action validity
            if signal.action not in ['buy', 'sell', 'hold']:
                self.logger.warning(f"Signal rejected: invalid action '{signal.action}'")
                return False
            
            # Check position size bounds
            if signal.risk_adjusted_size <= 0 or signal.risk_adjusted_size > 0.05:
                self.logger.warning(f"Signal rejected: invalid position size {signal.risk_adjusted_size}")
                return False
            
            # Check LSTM prediction bounds
            if not (0 <= signal.lstm_prediction <= 1):
                self.logger.warning(f"Signal rejected: invalid LSTM prediction {signal.lstm_prediction}")
                return False
            
            # Check sentiment score bounds
            if not (-1 <= signal.sentiment_score <= 1):
                self.logger.warning(f"Signal rejected: invalid sentiment score {signal.sentiment_score}")
                return False
            
            # Check symbol validity
            if not signal.symbol or len(signal.symbol) == 0:
                self.logger.warning("Signal rejected: empty symbol")
                return False
            
            # Additional validation for hold signals
            if signal.action == 'hold':
                # Hold signals should have lower confidence requirements
                return True
            
            # For buy/sell signals, require higher confidence
            if signal.action in ['buy', 'sell'] and signal.confidence < 0.7:
                self.logger.debug(f"Buy/sell signal rejected: confidence {signal.confidence:.3f} "
                                f"below 0.7 threshold")
                return False
            
            self.logger.debug(f"Signal validated successfully for {signal.symbol}: {signal.action}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating signal: {str(e)}")
            return False
    
    def generate_signals_batch(self, market_data_batch: Dict[str, Dict]) -> Dict[str, TradingSignal]:
        """
        Generate trading signals for multiple symbols in batch.
        
        Args:
            market_data_batch: Dictionary mapping symbols to their market data
            
        Returns:
            Dictionary mapping symbols to their trading signals
        """
        signals = {}
        
        for symbol, market_data in market_data_batch.items():
            try:
                # Get LSTM prediction if model is available
                if self.lstm_model and self.lstm_model.is_trained:
                    # This would require proper sequence preparation in practice
                    lstm_prediction = 0.5  # Placeholder - would use actual model prediction
                else:
                    lstm_prediction = 0.5  # Neutral prediction
                
                # Get DQN Q-values if agent is available
                if self.dqn_agent and self.dqn_agent.is_trained:
                    state_vector = self._prepare_state_vector(
                        market_data, lstm_prediction, market_data.get('sentiment_score', 0.0)
                    )
                    dqn_q_values = self.dqn_agent.get_q_values(state_vector)
                else:
                    dqn_q_values = {'buy': 0.0, 'sell': 0.0, 'hold': 0.0}
                
                # Generate signal
                signal = self.generate_signal(
                    symbol=symbol,
                    market_data=market_data,
                    lstm_prediction=lstm_prediction,
                    dqn_q_values=dqn_q_values,
                    sentiment_score=market_data.get('sentiment_score', 0.0)
                )
                
                # Validate signal
                if self.validate_signal(signal):
                    signals[symbol] = signal
                else:
                    self.logger.debug(f"Signal for {symbol} failed validation")
                
            except Exception as e:
                self.logger.error(f"Error generating signal for {symbol}: {str(e)}")
                continue
        
        self.logger.info(f"Generated {len(signals)} valid signals from {len(market_data_batch)} symbols")
        return signals
    
    def update_confidence_threshold(self, new_threshold: float) -> None:
        """
        Update minimum confidence threshold for signal validation.
        
        Args:
            new_threshold: New confidence threshold (0-1)
        """
        if 0 <= new_threshold <= 1:
            self.min_confidence = new_threshold
            self.logger.info(f"Confidence threshold updated to {new_threshold}")
        else:
            self.logger.warning(f"Invalid confidence threshold {new_threshold}. Must be between 0 and 1.")
    
    def get_signal_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about signal generation.
        
        Returns:
            Dictionary with signal generation statistics
        """
        return {
            'min_confidence': self.min_confidence,
            'weights': {
                'sentiment': self.sentiment_weight,
                'lstm': self.lstm_weight,
                'dqn': self.dqn_weight
            },
            'models_available': {
                'lstm': self.lstm_model is not None and (self.lstm_model.is_trained if hasattr(self.lstm_model, 'is_trained') else False),
                'dqn': self.dqn_agent is not None and (self.dqn_agent.is_trained if hasattr(self.dqn_agent, 'is_trained') else False)
            }
        }