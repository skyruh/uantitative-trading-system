#!/usr/bin/env python3
"""
Live Trading Orchestrator for paper trading with real-time data.
Coordinates real-time data, signal generation, order execution, and risk monitoring.
"""

import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
import json

from ..data.realtime_data_streamer import RealtimeDataStreamer
from ..trading.paper_trading_engine import PaperTradingEngine, OrderSide, OrderType
from ..trading.signal_generator import SignalGenerator
from ..trading.simple_strategy import SimpleTradingStrategy
from ..risk.live_risk_monitor import LiveRiskMonitor, RiskAlert
from ..config.settings import config


class LiveTradingOrchestrator:
    """
    Orchestrates live paper trading by coordinating:
    - Real-time data streaming
    - Signal generation
    - Order execution
    - Risk monitoring
    - Performance tracking
    """
    
    def __init__(self, symbols: List[str], initial_capital: float = 1000000.0):
        """
        Initialize live trading orchestrator.
        
        Args:
            symbols: List of symbols to trade
            initial_capital: Starting capital for paper trading
        """
        self.symbols = symbols
        self.initial_capital = initial_capital
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.data_streamer = RealtimeDataStreamer(symbols, update_interval=10)
        self.paper_engine = PaperTradingEngine(initial_capital)
        self.risk_monitor = LiveRiskMonitor()
        
        # Trading components
        self.signal_generator = SignalGenerator()
        self.strategy = SimpleTradingStrategy(
            self.signal_generator,
            None,  # We'll handle position management differently
            allow_short_selling=config.backtest.allow_short_selling
        )
        
        # State tracking
        self.is_trading = False
        self.trading_thread = None
        self.last_signal_time = {}
        self.signal_cooldown = 60  # 60 seconds between signals for same symbol
        
        # Performance tracking
        self.daily_trades = 0
        self.session_start_time = None
        self.session_start_value = initial_capital
        
        # Setup callbacks
        self._setup_callbacks()
        
        self.logger.info(f"Live trading orchestrator initialized for {len(symbols)} symbols")
    
    def _setup_callbacks(self):
        """Setup callbacks for data updates and risk alerts."""
        # Data update callback
        self.data_streamer.add_subscriber(self._on_market_data_update)
        
        # Risk alert callbacks
        def risk_alert_callback(alert: RiskAlert):
            self.logger.warning(f"RISK ALERT: {alert.message}")
            if alert.risk_level.value in ['high', 'critical']:
                self._handle_risk_alert(alert)
        
        def circuit_breaker_callback(reason: str):
            self.logger.critical(f"CIRCUIT BREAKER ACTIVATED: {reason}")
            self._handle_circuit_breaker(reason)
        
        self.risk_monitor.add_alert_callback(risk_alert_callback)
        self.risk_monitor.add_circuit_breaker_callback(circuit_breaker_callback)
    
    def start_live_trading(self):
        """Start live paper trading."""
        if self.is_trading:
            self.logger.warning("Live trading is already active")
            return
        
        self.logger.info("Starting live paper trading...")
        
        # Initialize session tracking
        self.session_start_time = datetime.now()
        self.session_start_value = self.initial_capital
        self.daily_trades = 0
        
        # Start components
        self.data_streamer.start_streaming()
        self.risk_monitor.start_monitoring()
        
        # Start trading loop
        self.is_trading = True
        self.trading_thread = threading.Thread(target=self._trading_worker, daemon=True)
        self.trading_thread.start()
        
        self.logger.info("Live paper trading started successfully")
    
    def stop_live_trading(self):
        """Stop live paper trading."""
        if not self.is_trading:
            self.logger.warning("Live trading is not active")
            return
        
        self.logger.info("Stopping live paper trading...")
        
        # Stop trading loop
        self.is_trading = False
        
        # Stop components
        self.data_streamer.stop_streaming()
        self.risk_monitor.stop_monitoring()
        
        # Wait for trading thread to finish
        if self.trading_thread and self.trading_thread.is_alive():
            self.trading_thread.join(timeout=10)
        
        # Generate session summary
        self._generate_session_summary()
        
        self.logger.info("Live paper trading stopped")
    
    def _trading_worker(self):
        """Main trading loop worker."""
        self.logger.info("Trading worker started")
        
        while self.is_trading:
            try:
                # Check if market is open
                if not self.data_streamer.is_market_open():
                    self.logger.info("Market is closed, pausing trading...")
                    time.sleep(300)  # Check every 5 minutes
                    continue
                
                # Get current market data
                current_data = self.data_streamer.get_all_current_data()
                
                if not current_data:
                    self.logger.debug("No market data available, waiting...")
                    time.sleep(30)
                    continue
                
                # Check risk before generating signals
                if self.risk_monitor.circuit_breaker_active:
                    self.logger.warning("Circuit breaker active, skipping signal generation")
                    time.sleep(60)
                    continue
                
                # Generate and process signals
                self._process_trading_signals(current_data)
                
                # Update risk monitoring
                self._update_risk_monitoring(current_data)
                
                # Wait before next iteration
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error in trading worker: {e}")
                time.sleep(30)
        
        self.logger.info("Trading worker stopped")
    
    def _on_market_data_update(self, market_data: Dict):
        """Handle real-time market data updates."""
        self.logger.debug(f"Received market data update for {len(market_data)} symbols")
        
        # Process pending orders
        self._process_pending_orders(market_data)
    
    def _process_trading_signals(self, market_data: Dict):
        """Generate and process trading signals."""
        try:
            # Convert market data to format expected by signal generator
            formatted_data = {}
            for symbol, data in market_data.items():
                formatted_data[symbol] = {
                    'Close': data['close'],
                    'Open': data['open'],
                    'High': data['high'],
                    'Low': data['low'],
                    'Volume': data['volume']
                }
            
            # Generate signals
            signals = self.signal_generator.generate_signals_batch(formatted_data)
            
            if not signals:
                return
            
            self.logger.info(f"Generated {len(signals)} trading signals")
            
            # Process each signal
            for symbol, signal in signals.items():
                # Check signal cooldown
                if self._is_signal_on_cooldown(symbol):
                    continue
                
                # Check if we should execute this signal
                if self._should_execute_signal(signal, market_data[symbol]):
                    self._execute_signal(signal, market_data[symbol])
                    self.last_signal_time[symbol] = datetime.now()
        
        except Exception as e:
            self.logger.error(f"Error processing trading signals: {e}")
    
    def _should_execute_signal(self, signal, market_data: Dict) -> bool:
        """Determine if a signal should be executed."""
        # Don't execute hold signals
        if signal.action == 'hold':
            return False
        
        # Check if we have a position
        current_position = self.paper_engine.get_position(signal.symbol)
        
        # Apply short selling rules
        if signal.action == 'sell' and not current_position:
            if not config.backtest.allow_short_selling:
                self.logger.debug(f"Skipping sell signal for {signal.symbol} - no position and short selling disabled")
                return False
        
        # Check if we're trying to buy when we already have a long position
        if signal.action == 'buy' and current_position and current_position.quantity > 0:
            self.logger.debug(f"Skipping buy signal for {signal.symbol} - already have long position")
            return False
        
        # Check confidence threshold
        if signal.confidence < 0.3:  # Minimum confidence threshold
            return False
        
        return True
    
    def _execute_signal(self, signal, market_data: Dict):
        """Execute a trading signal."""
        try:
            current_price = market_data['price']
            
            # Calculate position size
            portfolio_value = self.paper_engine.get_portfolio_value({signal.symbol: current_price})
            position_size_pct = signal.risk_adjusted_size
            position_value = portfolio_value * position_size_pct
            quantity = int(position_value / current_price)
            
            if quantity <= 0:
                self.logger.warning(f"Invalid quantity calculated for {signal.symbol}: {quantity}")
                return
            
            # Place order
            order_side = OrderSide.BUY if signal.action == 'buy' else OrderSide.SELL
            order_id = self.paper_engine.place_order(
                symbol=signal.symbol,
                side=order_side,
                quantity=quantity,
                order_type=OrderType.MARKET
            )
            
            if order_id:
                # Execute immediately (market order)
                success = self.paper_engine.execute_market_order(order_id, current_price)
                
                if success:
                    self.daily_trades += 1
                    self.logger.info(f"Signal executed: {signal.action.upper()} {quantity} {signal.symbol} @ ₹{current_price:.2f}")
                else:
                    self.logger.error(f"Failed to execute order {order_id}")
            else:
                self.logger.error(f"Failed to place order for {signal.symbol}")
        
        except Exception as e:
            self.logger.error(f"Error executing signal for {signal.symbol}: {e}")
    
    def _process_pending_orders(self, market_data: Dict):
        """Process any pending orders."""
        pending_orders = self.paper_engine.get_open_orders()
        
        for order in pending_orders:
            if order.symbol in market_data:
                current_price = market_data[order.symbol]['price']
                
                # For market orders, execute immediately
                if order.order_type == OrderType.MARKET:
                    self.paper_engine.execute_market_order(order.order_id, current_price)
    
    def _update_risk_monitoring(self, market_data: Dict):
        """Update risk monitoring with current portfolio state."""
        try:
            # Get current prices
            current_prices = {symbol: data['price'] for symbol, data in market_data.items()}
            
            # Get portfolio value and positions
            portfolio_value = self.paper_engine.get_portfolio_value(current_prices)
            positions = self.paper_engine.get_positions()
            
            # Run risk checks
            self.risk_monitor.check_portfolio_risk(portfolio_value, positions, current_prices)
        
        except Exception as e:
            self.logger.error(f"Error updating risk monitoring: {e}")
    
    def _is_signal_on_cooldown(self, symbol: str) -> bool:
        """Check if signal is on cooldown for a symbol."""
        if symbol not in self.last_signal_time:
            return False
        
        time_since_last = datetime.now() - self.last_signal_time[symbol]
        return time_since_last.total_seconds() < self.signal_cooldown
    
    def _handle_risk_alert(self, alert: RiskAlert):
        """Handle high-priority risk alerts."""
        # For high/critical alerts, we might want to close positions
        if alert.alert_type.value in ['daily_loss_limit', 'portfolio_drawdown']:
            self.logger.warning(f"Considering position closure due to: {alert.message}")
            # Could implement automatic position closure here
    
    def _handle_circuit_breaker(self, reason: str):
        """Handle circuit breaker activation."""
        # Cancel all pending orders
        pending_orders = self.paper_engine.get_open_orders()
        for order in pending_orders:
            self.paper_engine.cancel_order(order.order_id)
        
        self.logger.critical(f"All pending orders cancelled due to circuit breaker: {reason}")
    
    def _generate_session_summary(self):
        """Generate trading session summary."""
        current_prices = self.data_streamer.get_all_current_data()
        price_dict = {symbol: data['price'] for symbol, data in current_prices.items()}
        
        performance = self.paper_engine.get_performance_summary(price_dict)
        risk_summary = self.risk_monitor.get_risk_summary()
        
        session_duration = datetime.now() - self.session_start_time
        
        # Convert datetime objects to strings for JSON serialization
        market_status = self.data_streamer.get_market_status()
        if 'last_update' in market_status and hasattr(market_status['last_update'], 'isoformat'):
            market_status['last_update'] = market_status['last_update'].isoformat()
        
        summary = {
            'session_start': self.session_start_time.isoformat(),
            'session_duration_minutes': session_duration.total_seconds() / 60,
            'daily_trades': self.daily_trades,
            'performance': performance,
            'risk_summary': risk_summary,
            'final_positions': len(self.paper_engine.get_positions()),
            'market_status': market_status
        }
        
        # Save session summary
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"data/performance/live_session_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Session summary saved to {filename}")
        
        # Print summary
        print("\n" + "="*60)
        print("LIVE TRADING SESSION SUMMARY")
        print("="*60)
        print(f"Duration: {session_duration}")
        print(f"Total Trades: {self.daily_trades}")
        print(f"Portfolio Value: ₹{performance['portfolio_value']:,.2f}")
        print(f"Total Return: {performance['total_return_pct']:.2f}%")
        print(f"Max Drawdown: {performance['max_drawdown_pct']:.2f}%")
        print(f"Open Positions: {performance['open_positions']}")
        print(f"Total Commission: ₹{performance['total_commission']:,.2f}")
        print("="*60)
    
    def get_live_status(self) -> Dict:
        """Get current live trading status."""
        current_data = self.data_streamer.get_all_current_data()
        price_dict = {symbol: data['price'] for symbol, data in current_data.items()}
        
        performance = self.paper_engine.get_performance_summary(price_dict)
        risk_summary = self.risk_monitor.get_risk_summary()
        market_status = self.data_streamer.get_market_status()
        
        return {
            'is_trading': self.is_trading,
            'session_duration_minutes': (datetime.now() - self.session_start_time).total_seconds() / 60 if self.session_start_time else 0,
            'daily_trades': self.daily_trades,
            'performance': performance,
            'risk_summary': risk_summary,
            'market_status': market_status,
            'symbols_tracked': len(self.symbols),
            'active_positions': len(self.paper_engine.get_positions()),
            'pending_orders': len(self.paper_engine.get_open_orders())
        }


# Example usage
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Test symbols
    symbols = ["RELIANCE.NS", "TCS.NS", "INFY.NS"]
    
    # Create live trading orchestrator
    orchestrator = LiveTradingOrchestrator(symbols, initial_capital=100000)
    
    try:
        # Start live trading
        orchestrator.start_live_trading()
        
        # Run for a few minutes
        time.sleep(300)  # 5 minutes
        
    except KeyboardInterrupt:
        print("Stopping live trading...")
    finally:
        orchestrator.stop_live_trading()