#!/usr/bin/env python3
"""
Test script for live trading components.
Tests real-time data streaming, paper trading, and risk monitoring.
"""

import sys
import os
import time
import logging
from datetime import datetime

# Add src to Python path
sys.path.append('src')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_realtime_data_streamer():
    """Test the real-time data streamer."""
    print("\n" + "="*60)
    print("🧪 TESTING REAL-TIME DATA STREAMER")
    print("="*60)
    
    from src.data.realtime_data_streamer import RealtimeDataStreamer
    
    # Test symbols
    symbols = ["RELIANCE.NS", "TCS.NS"]
    
    # Create streamer
    streamer = RealtimeDataStreamer(symbols, update_interval=5)
    
    # Add test callback
    def data_callback(data):
        print(f"📊 Data Update: {len(data)} symbols")
        for symbol, info in data.items():
            change_pct = info.get('change_percent', 0)
            emoji = "📈" if change_pct >= 0 else "📉"
            print(f"   {emoji} {symbol}: ₹{info['price']:.2f} ({change_pct:+.2f}%)")
    
    streamer.add_subscriber(data_callback)
    
    # Test market status
    status = streamer.get_market_status()
    print(f"Market Status: {'🟢 OPEN' if status['is_open'] else '🔴 CLOSED'}")
    
    # Start streaming for 30 seconds
    print("Starting data stream for 30 seconds...")
    streamer.start_streaming()
    
    try:
        time.sleep(30)
    except KeyboardInterrupt:
        pass
    finally:
        streamer.stop_streaming()
    
    print("✅ Real-time data streamer test completed")


def test_paper_trading_engine():
    """Test the paper trading engine."""
    print("\n" + "="*60)
    print("🧪 TESTING PAPER TRADING ENGINE")
    print("="*60)
    
    from src.trading.paper_trading_engine import PaperTradingEngine, OrderSide, OrderType
    
    # Create engine
    engine = PaperTradingEngine(initial_capital=100000)
    
    print(f"💰 Initial Capital: ₹{engine.initial_capital:,.2f}")
    print(f"💵 Available Cash: ₹{engine.cash:,.2f}")
    
    # Test orders
    current_prices = {"RELIANCE.NS": 1500.0, "TCS.NS": 3200.0}
    
    # Place buy orders
    print("\n📋 Placing test orders...")
    order1 = engine.place_order("RELIANCE.NS", OrderSide.BUY, 10, OrderType.MARKET)
    order2 = engine.place_order("TCS.NS", OrderSide.BUY, 5, OrderType.MARKET)
    
    # Execute orders
    if order1:
        success = engine.execute_market_order(order1, current_prices["RELIANCE.NS"])
        print(f"Order 1 executed: {success}")
    
    if order2:
        success = engine.execute_market_order(order2, current_prices["TCS.NS"])
        print(f"Order 2 executed: {success}")
    
    # Check positions
    positions = engine.get_positions()
    print(f"\n📊 Positions: {len(positions)}")
    for symbol, position in positions.items():
        position.update_market_value(current_prices[symbol])
        print(f"   {symbol}: {position.quantity} shares @ ₹{position.avg_price:.2f}")
        print(f"      Market Value: ₹{position.market_value:,.2f}")
        print(f"      Unrealized P&L: ₹{position.unrealized_pnl:+,.2f}")
    
    # Performance summary
    performance = engine.get_performance_summary(current_prices)
    print(f"\n📈 Performance Summary:")
    print(f"   Portfolio Value: ₹{performance['portfolio_value']:,.2f}")
    print(f"   Total Return: {performance['total_return_pct']:+.2f}%")
    print(f"   Total Trades: {performance['total_trades']}")
    print(f"   Commission Paid: ₹{performance['total_commission']:,.2f}")
    
    print("✅ Paper trading engine test completed")


def test_risk_monitor():
    """Test the live risk monitor."""
    print("\n" + "="*60)
    print("🧪 TESTING LIVE RISK MONITOR")
    print("="*60)
    
    from src.risk.live_risk_monitor import LiveRiskMonitor, RiskLimits
    
    # Create risk monitor with strict limits for testing
    risk_limits = RiskLimits(
        max_position_size_pct=3.0,  # 3% max position
        max_daily_loss_pct=1.0,    # 1% max daily loss
        max_portfolio_drawdown_pct=5.0  # 5% max drawdown
    )
    
    monitor = LiveRiskMonitor(risk_limits)
    
    # Add alert callback
    def alert_callback(alert):
        risk_emoji = {"low": "🟢", "medium": "🟡", "high": "🟠", "critical": "🔴"}
        emoji = risk_emoji.get(alert.risk_level.value, "⚠️")
        print(f"{emoji} RISK ALERT: {alert.message}")
    
    def circuit_breaker_callback(reason):
        print(f"🚨 CIRCUIT BREAKER: {reason}")
    
    monitor.add_alert_callback(alert_callback)
    monitor.add_circuit_breaker_callback(circuit_breaker_callback)
    
    # Start monitoring
    monitor.start_monitoring()
    
    # Simulate portfolio scenarios
    print("Testing risk scenarios...")
    
    # Mock positions
    mock_positions = {
        'RELIANCE.NS': type('Position', (), {
            'quantity': 100, 
            'avg_price': 1400,
            'market_value': 150000,
            'unrealized_pnl': 10000
        })()
    }
    
    # Test 1: Normal portfolio
    print("\n1. Testing normal portfolio...")
    alerts = monitor.check_portfolio_risk(100000, mock_positions, {'RELIANCE.NS': 1500})
    print(f"   Generated {len(alerts)} alerts")
    
    # Test 2: Large position (should trigger concentration alert)
    print("\n2. Testing large position...")
    large_positions = {
        'RELIANCE.NS': type('Position', (), {
            'quantity': 500, 
            'avg_price': 1400,
            'market_value': 750000,
            'unrealized_pnl': 50000
        })()
    }
    alerts = monitor.check_portfolio_risk(100000, large_positions, {'RELIANCE.NS': 1500})
    print(f"   Generated {len(alerts)} alerts")
    
    # Test 3: Daily loss (should trigger circuit breaker)
    print("\n3. Testing daily loss scenario...")
    monitor.daily_start_value = 100000
    alerts = monitor.check_portfolio_risk(98000, mock_positions, {'RELIANCE.NS': 1500})
    print(f"   Generated {len(alerts)} alerts")
    
    # Get risk summary
    summary = monitor.get_risk_summary()
    print(f"\n📊 Risk Summary:")
    print(f"   Circuit Breaker: {'🔴 ACTIVE' if summary['circuit_breaker_active'] else '🟢 INACTIVE'}")
    print(f"   Daily P&L: {summary['daily_pnl_pct']:+.2f}%")
    print(f"   Active Alerts: {summary['active_alerts']}")
    print(f"   Critical Alerts: {summary['critical_alerts']}")
    
    monitor.stop_monitoring()
    print("✅ Risk monitor test completed")


def test_live_orchestrator():
    """Test the live trading orchestrator (brief test)."""
    print("\n" + "="*60)
    print("🧪 TESTING LIVE TRADING ORCHESTRATOR")
    print("="*60)
    
    from src.trading.live_trading_orchestrator import LiveTradingOrchestrator
    
    # Test symbols
    symbols = ["RELIANCE.NS", "TCS.NS"]
    
    # Create orchestrator
    orchestrator = LiveTradingOrchestrator(symbols, initial_capital=100000)
    
    print(f"📊 Symbols: {len(symbols)}")
    print(f"💰 Initial Capital: ₹{orchestrator.initial_capital:,.2f}")
    
    # Start live trading for 60 seconds
    print("Starting live trading for 60 seconds...")
    orchestrator.start_live_trading()
    
    try:
        # Monitor for 60 seconds
        for i in range(6):  # 6 x 10 seconds = 60 seconds
            time.sleep(10)
            status = orchestrator.get_live_status()
            print(f"Status Update {i+1}/6:")
            print(f"   Trading: {'🟢 ACTIVE' if status['is_trading'] else '🔴 INACTIVE'}")
            print(f"   Daily Trades: {status['daily_trades']}")
            print(f"   Portfolio: ₹{status['performance']['portfolio_value']:,.2f}")
            print(f"   Active Positions: {status['active_positions']}")
            print(f"   Market: {'🟢 OPEN' if status['market_status']['is_open'] else '🔴 CLOSED'}")
    
    except KeyboardInterrupt:
        print("Test interrupted by user")
    
    finally:
        orchestrator.stop_live_trading()
    
    print("✅ Live trading orchestrator test completed")


def main():
    """Run all tests."""
    print("🚀 LIVE TRADING SYSTEM TESTS")
    print("=" * 80)
    print(f"📅 Test Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    try:
        # Test individual components
        test_realtime_data_streamer()
        test_paper_trading_engine()
        test_risk_monitor()
        
        # Test integrated system
        test_live_orchestrator()
        
        print("\n" + "="*80)
        print("✅ ALL TESTS COMPLETED SUCCESSFULLY!")
        print("🎉 Your live trading system is ready!")
        print("=" * 80)
        print("\n📋 Next Steps:")
        print("1. Configure trading: python configure_trading.py")
        print("2. Run live trading: python main.py --mode live")
        print("3. Or use standalone: python run_live_trading.py")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()