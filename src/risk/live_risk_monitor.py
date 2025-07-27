#!/usr/bin/env python3
"""
Live Risk Monitoring System for real-time risk management.
Monitors positions, portfolio exposure, and implements circuit breakers.
"""

import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import logging
import json


class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertType(Enum):
    POSITION_LIMIT = "position_limit"
    PORTFOLIO_DRAWDOWN = "portfolio_drawdown"
    DAILY_LOSS_LIMIT = "daily_loss_limit"
    CONCENTRATION_RISK = "concentration_risk"
    VOLATILITY_SPIKE = "volatility_spike"
    MARGIN_CALL = "margin_call"
    CIRCUIT_BREAKER = "circuit_breaker"


@dataclass
class RiskAlert:
    """Represents a risk management alert."""
    alert_id: str
    alert_type: AlertType
    risk_level: RiskLevel
    symbol: Optional[str]
    message: str
    value: float
    threshold: float
    timestamp: datetime
    acknowledged: bool = False
    
    def to_dict(self) -> Dict:
        return {
            'alert_id': self.alert_id,
            'alert_type': self.alert_type.value,
            'risk_level': self.risk_level.value,
            'symbol': self.symbol,
            'message': self.message,
            'value': self.value,
            'threshold': self.threshold,
            'timestamp': self.timestamp.isoformat(),
            'acknowledged': self.acknowledged
        }


@dataclass
class RiskLimits:
    """Risk management limits configuration."""
    max_position_size_pct: float = 5.0  # Max 5% of portfolio per position
    max_daily_loss_pct: float = 2.0     # Max 2% daily loss
    max_portfolio_drawdown_pct: float = 10.0  # Max 10% drawdown
    max_concentration_pct: float = 20.0  # Max 20% in single stock
    max_sector_exposure_pct: float = 30.0  # Max 30% in single sector
    volatility_threshold: float = 5.0    # 5% price movement threshold
    min_cash_reserve_pct: float = 10.0   # Keep 10% cash reserve


class LiveRiskMonitor:
    """
    Live risk monitoring system that continuously monitors portfolio risk
    and implements automated risk controls and circuit breakers.
    """
    
    def __init__(self, risk_limits: RiskLimits = None):
        """Initialize live risk monitor."""
        self.risk_limits = risk_limits or RiskLimits()
        self.logger = logging.getLogger(__name__)
        
        # Risk tracking
        self.alerts: List[RiskAlert] = []
        self.daily_pnl = 0.0
        self.daily_start_value = 0.0
        self.max_portfolio_value = 0.0
        self.current_drawdown = 0.0
        
        # Monitoring control
        self.is_monitoring = False
        self.monitor_thread = None
        self.check_interval = 30  # Check every 30 seconds
        
        # Callbacks for risk events
        self.alert_callbacks: List[Callable] = []
        self.circuit_breaker_callbacks: List[Callable] = []
        
        # Circuit breaker state
        self.circuit_breaker_active = False
        self.circuit_breaker_reason = None
        
        self.logger.info("Live risk monitor initialized")
    
    def add_alert_callback(self, callback: Callable):
        """Add callback for risk alerts."""
        self.alert_callbacks.append(callback)
    
    def add_circuit_breaker_callback(self, callback: Callable):
        """Add callback for circuit breaker events."""
        self.circuit_breaker_callbacks.append(callback)
    
    def start_monitoring(self):
        """Start live risk monitoring."""
        if self.is_monitoring:
            self.logger.warning("Risk monitoring is already active")
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_worker, daemon=True)
        self.monitor_thread.start()
        
        # Initialize daily tracking
        self.daily_start_value = 0.0  # Will be set on first portfolio update
        
        self.logger.info("Live risk monitoring started")
    
    def stop_monitoring(self):
        """Stop live risk monitoring."""
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=10)
        
        self.logger.info("Live risk monitoring stopped")
    
    def _monitor_worker(self):
        """Background worker for risk monitoring."""
        while self.is_monitoring:
            try:
                # Risk monitoring logic will be triggered by portfolio updates
                time.sleep(self.check_interval)
            except Exception as e:
                self.logger.error(f"Error in risk monitor worker: {e}")
                time.sleep(self.check_interval)
    
    def check_portfolio_risk(self, portfolio_value: float, positions: Dict, 
                           current_prices: Dict[str, float]) -> List[RiskAlert]:
        """
        Comprehensive portfolio risk check.
        
        Args:
            portfolio_value: Current total portfolio value
            positions: Current positions dict
            current_prices: Current market prices
            
        Returns:
            List of risk alerts
        """
        new_alerts = []
        
        # Initialize daily tracking
        if self.daily_start_value == 0.0:
            self.daily_start_value = portfolio_value
            self.max_portfolio_value = portfolio_value
        
        # Update max portfolio value and drawdown
        if portfolio_value > self.max_portfolio_value:
            self.max_portfolio_value = portfolio_value
        
        self.current_drawdown = (self.max_portfolio_value - portfolio_value) / self.max_portfolio_value * 100
        
        # Calculate daily P&L
        self.daily_pnl = (portfolio_value - self.daily_start_value) / self.daily_start_value * 100
        
        # 1. Check daily loss limit
        if self.daily_pnl < -self.risk_limits.max_daily_loss_pct:
            alert = self._create_alert(
                AlertType.DAILY_LOSS_LIMIT,
                RiskLevel.CRITICAL,
                None,
                f"Daily loss limit exceeded: {self.daily_pnl:.2f}%",
                abs(self.daily_pnl),
                self.risk_limits.max_daily_loss_pct
            )
            new_alerts.append(alert)
            self._trigger_circuit_breaker("Daily loss limit exceeded")
        
        # 2. Check portfolio drawdown
        if self.current_drawdown > self.risk_limits.max_portfolio_drawdown_pct:
            risk_level = RiskLevel.CRITICAL if self.current_drawdown > 15 else RiskLevel.HIGH
            alert = self._create_alert(
                AlertType.PORTFOLIO_DRAWDOWN,
                risk_level,
                None,
                f"Portfolio drawdown: {self.current_drawdown:.2f}%",
                self.current_drawdown,
                self.risk_limits.max_portfolio_drawdown_pct
            )
            new_alerts.append(alert)
            
            if risk_level == RiskLevel.CRITICAL:
                self._trigger_circuit_breaker("Excessive portfolio drawdown")
        
        # 3. Check position size limits
        for symbol, position in positions.items():
            if symbol in current_prices:
                position_value = abs(position.quantity) * current_prices[symbol]
                position_pct = (position_value / portfolio_value) * 100
                
                if position_pct > self.risk_limits.max_position_size_pct:
                    risk_level = RiskLevel.HIGH if position_pct > 7 else RiskLevel.MEDIUM
                    alert = self._create_alert(
                        AlertType.POSITION_LIMIT,
                        risk_level,
                        symbol,
                        f"Position size limit exceeded: {position_pct:.2f}%",
                        position_pct,
                        self.risk_limits.max_position_size_pct
                    )
                    new_alerts.append(alert)
        
        # 4. Check concentration risk
        total_long_exposure = sum(
            abs(pos.quantity) * current_prices.get(symbol, 0)
            for symbol, pos in positions.items()
            if pos.quantity > 0 and symbol in current_prices
        )
        
        if total_long_exposure > 0:
            for symbol, position in positions.items():
                if position.quantity > 0 and symbol in current_prices:
                    position_value = position.quantity * current_prices[symbol]
                    concentration_pct = (position_value / total_long_exposure) * 100
                    
                    if concentration_pct > self.risk_limits.max_concentration_pct:
                        alert = self._create_alert(
                            AlertType.CONCENTRATION_RISK,
                            RiskLevel.MEDIUM,
                            symbol,
                            f"Concentration risk: {concentration_pct:.2f}% of long exposure",
                            concentration_pct,
                            self.risk_limits.max_concentration_pct
                        )
                        new_alerts.append(alert)
        
        # 5. Check volatility spikes
        for symbol, position in positions.items():
            if symbol in current_prices:
                current_price = current_prices[symbol]
                # Simple volatility check based on position's average price
                if hasattr(position, 'avg_price') and position.avg_price > 0:
                    price_change_pct = abs((current_price - position.avg_price) / position.avg_price) * 100
                    
                    if price_change_pct > self.risk_limits.volatility_threshold:
                        risk_level = RiskLevel.HIGH if price_change_pct > 10 else RiskLevel.MEDIUM
                        alert = self._create_alert(
                            AlertType.VOLATILITY_SPIKE,
                            risk_level,
                            symbol,
                            f"High volatility detected: {price_change_pct:.2f}% price movement",
                            price_change_pct,
                            self.risk_limits.volatility_threshold
                        )
                        new_alerts.append(alert)
        
        # Add new alerts and notify callbacks
        for alert in new_alerts:
            self.alerts.append(alert)
            self._notify_alert_callbacks(alert)
        
        return new_alerts
    
    def _create_alert(self, alert_type: AlertType, risk_level: RiskLevel, 
                     symbol: Optional[str], message: str, value: float, 
                     threshold: float) -> RiskAlert:
        """Create a new risk alert."""
        alert_id = f"{alert_type.value}_{int(datetime.now().timestamp())}"
        
        return RiskAlert(
            alert_id=alert_id,
            alert_type=alert_type,
            risk_level=risk_level,
            symbol=symbol,
            message=message,
            value=value,
            threshold=threshold,
            timestamp=datetime.now()
        )
    
    def _trigger_circuit_breaker(self, reason: str):
        """Trigger circuit breaker to halt trading."""
        if not self.circuit_breaker_active:
            self.circuit_breaker_active = True
            self.circuit_breaker_reason = reason
            
            self.logger.critical(f"CIRCUIT BREAKER ACTIVATED: {reason}")
            
            # Create critical alert
            alert = self._create_alert(
                AlertType.CIRCUIT_BREAKER,
                RiskLevel.CRITICAL,
                None,
                f"Circuit breaker activated: {reason}",
                0.0,
                0.0
            )
            self.alerts.append(alert)
            
            # Notify callbacks
            for callback in self.circuit_breaker_callbacks:
                try:
                    callback(reason)
                except Exception as e:
                    self.logger.error(f"Error in circuit breaker callback: {e}")
    
    def _notify_alert_callbacks(self, alert: RiskAlert):
        """Notify alert callbacks."""
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error(f"Error in alert callback: {e}")
    
    def reset_circuit_breaker(self):
        """Reset circuit breaker (manual intervention required)."""
        self.circuit_breaker_active = False
        self.circuit_breaker_reason = None
        self.logger.info("Circuit breaker reset")
    
    def reset_daily_tracking(self, current_portfolio_value: float):
        """Reset daily tracking (call at market open)."""
        self.daily_start_value = current_portfolio_value
        self.daily_pnl = 0.0
        self.logger.info("Daily risk tracking reset")
    
    def get_active_alerts(self, risk_level: Optional[RiskLevel] = None) -> List[RiskAlert]:
        """Get active (unacknowledged) alerts."""
        alerts = [alert for alert in self.alerts if not alert.acknowledged]
        
        if risk_level:
            alerts = [alert for alert in alerts if alert.risk_level == risk_level]
        
        return alerts
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge a risk alert."""
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                self.logger.info(f"Alert acknowledged: {alert_id}")
                return True
        return False
    
    def get_risk_summary(self) -> Dict:
        """Get current risk summary."""
        active_alerts = self.get_active_alerts()
        critical_alerts = [a for a in active_alerts if a.risk_level == RiskLevel.CRITICAL]
        high_alerts = [a for a in active_alerts if a.risk_level == RiskLevel.HIGH]
        
        return {
            'circuit_breaker_active': self.circuit_breaker_active,
            'circuit_breaker_reason': self.circuit_breaker_reason,
            'daily_pnl_pct': self.daily_pnl,
            'current_drawdown_pct': self.current_drawdown,
            'total_alerts': len(self.alerts),
            'active_alerts': len(active_alerts),
            'critical_alerts': len(critical_alerts),
            'high_alerts': len(high_alerts),
            'risk_limits': {
                'max_daily_loss_pct': self.risk_limits.max_daily_loss_pct,
                'max_drawdown_pct': self.risk_limits.max_portfolio_drawdown_pct,
                'max_position_size_pct': self.risk_limits.max_position_size_pct,
                'max_concentration_pct': self.risk_limits.max_concentration_pct
            }
        }
    
    def save_alerts(self, filepath: str):
        """Save alerts to file."""
        alerts_data = [alert.to_dict() for alert in self.alerts]
        
        with open(filepath, 'w') as f:
            json.dump(alerts_data, f, indent=2)
        
        self.logger.info(f"Risk alerts saved to {filepath}")


# Example usage
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Create risk monitor
    risk_monitor = LiveRiskMonitor()
    
    # Add alert callback
    def alert_callback(alert: RiskAlert):
        print(f"RISK ALERT: {alert.risk_level.value.upper()} - {alert.message}")
    
    def circuit_breaker_callback(reason: str):
        print(f"🚨 CIRCUIT BREAKER ACTIVATED: {reason}")
    
    risk_monitor.add_alert_callback(alert_callback)
    risk_monitor.add_circuit_breaker_callback(circuit_breaker_callback)
    
    # Start monitoring
    risk_monitor.start_monitoring()
    
    # Simulate portfolio check
    mock_positions = {
        'RELIANCE.NS': type('Position', (), {'quantity': 100, 'avg_price': 1400})()
    }
    mock_prices = {'RELIANCE.NS': 1500.0}
    
    alerts = risk_monitor.check_portfolio_risk(100000, mock_positions, mock_prices)
    print(f"Generated {len(alerts)} alerts")
    
    # Get risk summary
    summary = risk_monitor.get_risk_summary()
    print("Risk Summary:", summary)