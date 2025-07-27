#!/usr/bin/env python3
"""
Configuration script for trading system settings.
Easily toggle between different trading modes.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.config.settings import config

def configure_for_intraday_trading():
    """Configure system for intraday trading with short selling enabled."""
    print("🔧 Configuring for Intraday Trading...")
    config.backtest.allow_short_selling = True
    print("✅ Short selling: ENABLED")
    print("   - Can sell stocks without owning them")
    print("   - Suitable for intraday trading strategies")
    print("   - Both buy and sell signals will be executed")

def configure_for_positional_trading():
    """Configure system for positional trading with short selling disabled."""
    print("🔧 Configuring for Positional Trading...")
    config.backtest.allow_short_selling = False
    print("✅ Short selling: DISABLED")
    print("   - Can only sell stocks you own")
    print("   - Suitable for long-term positional trading")
    print("   - Sell signals without positions will be ignored")

def show_current_config():
    """Display current trading configuration."""
    print("\n📊 Current Trading Configuration:")
    print(f"   Short Selling: {'ENABLED' if config.backtest.allow_short_selling else 'DISABLED'}")
    print(f"   Initial Capital: ₹{config.backtest.initial_capital:,.2f}")
    print(f"   Transaction Cost: {config.backtest.transaction_cost*100:.2f}%")
    print(f"   Slippage: {config.backtest.slippage*100:.3f}%")
    print(f"   Use Leverage: {'YES' if config.backtest.use_leverage else 'NO'}")

def main():
    """Main configuration interface."""
    print("=" * 60)
    print("🚀 QUANTITATIVE TRADING SYSTEM CONFIGURATOR")
    print("=" * 60)
    
    show_current_config()
    
    print("\n📋 Available Configurations:")
    print("1. Intraday Trading (Short Selling Enabled)")
    print("2. Positional Trading (Short Selling Disabled)")
    print("3. Show Current Config")
    print("4. Exit")
    
    while True:
        try:
            choice = input("\n🔹 Select configuration (1-4): ").strip()
            
            if choice == '1':
                configure_for_intraday_trading()
                show_current_config()
                break
            elif choice == '2':
                configure_for_positional_trading()
                show_current_config()
                break
            elif choice == '3':
                show_current_config()
            elif choice == '4':
                print("👋 Exiting configurator...")
                break
            else:
                print("❌ Invalid choice. Please select 1-4.")
                
        except KeyboardInterrupt:
            print("\n👋 Exiting configurator...")
            break
    
    print("\n✨ Configuration complete! You can now run your backtest.")
    print("   Run: python main.py --mode backtest")
    print("   Or:  python debug_backtest.py")

if __name__ == "__main__":
    main()