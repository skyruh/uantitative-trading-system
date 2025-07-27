#!/usr/bin/env python3
"""
Live Paper Trading Runner
Starts the live trading system with real-time data and paper trading.
"""

import sys
import os
import signal
import time
from datetime import datetime

# Add src to Python path
sys.path.append('src')

from src.trading.live_trading_orchestrator import LiveTradingOrchestrator
from src.config.settings import config
import logging

# Global orchestrator for signal handling
orchestrator = None


def signal_handler(signum, frame):
    """Handle system signals for graceful shutdown."""
    print(f"\nReceived signal {signum}, shutting down gracefully...")
    
    if orchestrator:
        orchestrator.stop_live_trading()
    
    print("Live trading stopped. Goodbye!")
    sys.exit(0)


def setup_signal_handlers():
    """Setup signal handlers for graceful shutdown."""
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def print_banner():
    """Print startup banner."""
    print("=" * 80)
    print("🚀 QUANTITATIVE TRADING SYSTEM - LIVE PAPER TRADING")
    print("=" * 80)
    print(f"📅 Session Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"💰 Initial Capital: ₹{config.backtest.initial_capital:,.2f}")
    print(f"📊 Short Selling: {'ENABLED' if config.backtest.allow_short_selling else 'DISABLED'}")
    print("=" * 80)


def load_trading_symbols():
    """Load symbols for trading."""
    symbols_file = config.data.stock_symbols_file
    
    try:
        with open(symbols_file, 'r') as f:
            symbols = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        
        # Limit to first 10 symbols for live trading to avoid rate limits
        symbols = symbols[:10]
        
        print(f"📈 Loaded {len(symbols)} symbols for live trading:")
        for i, symbol in enumerate(symbols, 1):
            print(f"   {i:2d}. {symbol}")
        
        return symbols
        
    except FileNotFoundError:
        print(f"❌ Symbols file not found: {symbols_file}")
        print("Using default symbols...")
        return ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS"]


def print_live_status(orchestrator):
    """Print live trading status."""
    status = orchestrator.get_live_status()
    
    print("\n" + "=" * 60)
    print("📊 LIVE TRADING STATUS")
    print("=" * 60)
    
    # Market status
    market = status['market_status']
    market_emoji = "🟢" if market['is_open'] else "🔴"
    print(f"{market_emoji} Market Status: {'OPEN' if market['is_open'] else 'CLOSED'}")
    print(f"⏰ Current Time: {market['current_time']}")
    print(f"📈 Symbols Tracked: {status['symbols_tracked']}")
    
    # Performance
    perf = status['performance']
    return_emoji = "📈" if perf['total_return_pct'] >= 0 else "📉"
    print(f"\n💰 Portfolio Value: ₹{perf['portfolio_value']:,.2f}")
    print(f"{return_emoji} Total Return: {perf['total_return_pct']:+.2f}%")
    print(f"💸 Total Commission: ₹{perf['total_commission']:,.2f}")
    
    # Trading activity
    print(f"\n🔄 Daily Trades: {status['daily_trades']}")
    print(f"📊 Active Positions: {status['active_positions']}")
    print(f"⏳ Pending Orders: {status['pending_orders']}")
    
    # Risk status
    risk = status['risk_summary']
    risk_emoji = "🚨" if risk['circuit_breaker_active'] else "✅"
    print(f"\n{risk_emoji} Risk Status: {'CIRCUIT BREAKER ACTIVE' if risk['circuit_breaker_active'] else 'NORMAL'}")
    print(f"📉 Daily P&L: {risk['daily_pnl_pct']:+.2f}%")
    print(f"⚠️  Active Alerts: {risk['active_alerts']} ({risk['critical_alerts']} critical)")
    
    print("=" * 60)


def main():
    """Main live trading function."""
    global orchestrator
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'logs/live_trading_{datetime.now().strftime("%Y%m%d")}.log'),
            logging.StreamHandler()
        ]
    )
    
    # Setup signal handlers
    setup_signal_handlers()
    
    # Print banner
    print_banner()
    
    # Load trading symbols
    symbols = load_trading_symbols()
    
    if not symbols:
        print("❌ No symbols loaded. Exiting...")
        return
    
    try:
        # Create orchestrator
        print("\n🔧 Initializing live trading system...")
        orchestrator = LiveTradingOrchestrator(
            symbols=symbols,
            initial_capital=config.backtest.initial_capital
        )
        
        # Start live trading
        print("🚀 Starting live paper trading...")
        orchestrator.start_live_trading()
        
        print("\n✅ Live trading started successfully!")
        print("\n📋 Commands:")
        print("   - Press Ctrl+C to stop trading")
        print("   - Status updates every 60 seconds")
        print("   - Check logs for detailed activity")
        
        # Main loop - print status updates
        last_status_time = time.time()
        
        while True:
            time.sleep(10)  # Check every 10 seconds
            
            # Print status every 60 seconds
            if time.time() - last_status_time >= 60:
                print_live_status(orchestrator)
                last_status_time = time.time()
    
    except KeyboardInterrupt:
        print("\n⏹️  Keyboard interrupt received...")
    
    except Exception as e:
        print(f"\n❌ Error in live trading: {e}")
        logging.error(f"Live trading error: {e}")
    
    finally:
        if orchestrator:
            print("\n🛑 Stopping live trading...")
            orchestrator.stop_live_trading()
        
        print("\n👋 Live trading session ended.")


if __name__ == "__main__":
    main()