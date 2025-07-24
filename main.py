#!/usr/bin/env python3
"""
Main entry point for the Quantitative Trading System.
Integrated system orchestrator for end-to-end trading operations.
"""

import sys
import os
import argparse
import signal

# Add src to Python path
sys.path.append('src')

from src.config.settings import config
from src.utils.logging_utils import get_logger, log_system_startup, log_system_shutdown
from src.utils.validation_utils import validate_system_dependencies
from src.system.trading_system_orchestrator import TradingSystemOrchestrator

# Global orchestrator instance for signal handling
orchestrator = None


def signal_handler(signum, frame):
    """Handle system signals for graceful shutdown."""
    logger = get_logger("Main")
    logger.info(f"Received signal {signum}, initiating graceful shutdown...")
    
    if orchestrator:
        orchestrator.shutdown_system()
    
    sys.exit(0)


def setup_signal_handlers():
    """Setup signal handlers for graceful shutdown."""
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def validate_system_dependencies():
    """Validate system dependencies and setup."""
    logger = get_logger("Main")
    
    # Log system startup
    log_system_startup()
    
    # Validate system dependencies
    from src.utils.validation_utils import validate_system_dependencies as validate_deps
    is_valid, issues = validate_deps()
    if not is_valid:
        logger.error("System validation failed:")
        for issue in issues:
            logger.error(f"  - {issue}")
        logger.error("Please install required packages: pip install -r requirements.txt")
        return False
    
    logger.info("System validation passed")
    return True


def run_setup_mode(orchestrator: TradingSystemOrchestrator):
    """Run system setup and validation."""
    logger = get_logger("Main")
    
    logger.info("Running system setup and validation...")
    
    if orchestrator.startup_system():
        logger.info("System setup completed successfully")
        
        # Display system status
        status = orchestrator.get_system_status()
        logger.info("System Status:")
        for key, value in status["system_state"].items():
            status_symbol = "[OK]" if value else "[X]"
            logger.info(f"  {status_symbol} {key}")
        
        logger.info("\nNext steps:")
        logger.info("1. Run data collection: python main.py --mode collect")
        logger.info("2. Train models: python main.py --mode train")
        logger.info("3. Run backtesting: python main.py --mode backtest")
        logger.info("4. Run complete workflow: python main.py --mode complete")
        
        return True
    else:
        logger.error("System setup failed")
        return False


def run_data_collection_mode(orchestrator: TradingSystemOrchestrator):
    """Run data collection workflow."""
    logger = get_logger("Main")
    
    logger.info("Starting data collection workflow...")
    
    if not orchestrator.startup_system():
        logger.error("System startup failed")
        return False
    
    success = orchestrator.run_data_collection_workflow()
    
    if success:
        logger.info("Data collection completed successfully")
        logger.info("Next step: python main.py --mode train")
    else:
        logger.error("Data collection failed")
    
    return success


def run_training_mode(orchestrator: TradingSystemOrchestrator):
    """Run model training workflow."""
    logger = get_logger("Main")
    
    logger.info("Starting model training workflow...")
    
    if not orchestrator.startup_system():
        logger.error("System startup failed")
        return False
    
    success = orchestrator.run_model_training_workflow()
    
    if success:
        logger.info("Model training completed successfully")
        logger.info("Next step: python main.py --mode backtest")
    else:
        logger.error("Model training failed")
    
    return success


def run_backtesting_mode(orchestrator: TradingSystemOrchestrator):
    """Run backtesting workflow."""
    logger = get_logger("Main")
    
    logger.info("Starting backtesting workflow...")
    
    if not orchestrator.startup_system():
        logger.error("System startup failed")
        return False
    
    success = orchestrator.run_backtesting_workflow()
    
    if success:
        logger.info("Backtesting completed successfully")
        
        # Display system readiness
        status = orchestrator.get_system_status()
        if status["system_state"]["ready_for_trading"]:
            logger.info("System is ready for live trading operations")
        else:
            logger.warning("System performance targets not met - review before live trading")
    else:
        logger.error("Backtesting failed")
    
    return success


def run_complete_workflow(orchestrator: TradingSystemOrchestrator):
    """Run complete end-to-end workflow."""
    logger = get_logger("Main")
    
    logger.info("Starting complete system workflow...")
    logger.info("This will run: data collection → model training → backtesting")
    
    success = orchestrator.run_complete_system_workflow()
    
    if success:
        logger.info("Complete workflow finished successfully")
        
        # Display final system status
        status = orchestrator.get_system_status()
        logger.info("\nFinal System Status:")
        for key, value in status["system_state"].items():
            status_symbol = "[OK]" if value else "[X]"
            logger.info(f"  {status_symbol} {key}")
        
        if status["system_state"]["ready_for_trading"]:
            logger.info("\n🎉 System is ready for live trading operations!")
        else:
            logger.warning("\n⚠️  System validation issues - review performance before live trading")
    else:
        logger.error("Complete workflow failed")
    
    return success


def run_status_mode(orchestrator: TradingSystemOrchestrator):
    """Display current system status."""
    logger = get_logger("Main")
    
    logger.info("Checking system status...")
    
    try:
        status = orchestrator.get_system_status()
        
        logger.info(f"Environment: {status['environment']}")
        logger.info(f"Timestamp: {status['timestamp']}")
        
        logger.info("\nSystem State:")
        for key, value in status["system_state"].items():
            status_symbol = "[OK]" if value else "[X]"
            logger.info(f"  {status_symbol} {key}")
        
        logger.info("\nHealth Status:")
        for key, value in status["health_status"].items():
            logger.info(f"  {key}: {value}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to get system status: {str(e)}")
        return False


def main():
    """Main function with integrated system orchestration."""
    global orchestrator
    
    parser = argparse.ArgumentParser(description="Quantitative Trading System")
    parser.add_argument("--mode", 
                       choices=["setup", "collect", "train", "backtest", "complete", "status"], 
                       default="setup", 
                       help="Operation mode")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--env", 
                       choices=["development", "testing", "production"], 
                       default="development", 
                       help="Environment")
    
    args = parser.parse_args()
    
    # Setup signal handlers for graceful shutdown
    setup_signal_handlers()
    
    logger = get_logger("Main")
    
    try:
        # Validate system dependencies first
        if not validate_system_dependencies():
            logger.error("System dependency validation failed")
            return 1
        
        # Initialize orchestrator
        orchestrator = TradingSystemOrchestrator(environment=args.env)
        
        # Route to appropriate workflow based on mode
        success = False
        
        if args.mode == "setup":
            success = run_setup_mode(orchestrator)
            
        elif args.mode == "collect":
            success = run_data_collection_mode(orchestrator)
            
        elif args.mode == "train":
            success = run_training_mode(orchestrator)
            
        elif args.mode == "backtest":
            success = run_backtesting_mode(orchestrator)
            
        elif args.mode == "complete":
            success = run_complete_workflow(orchestrator)
            
        elif args.mode == "status":
            success = run_status_mode(orchestrator)
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        logger.info("System interrupted by user")
        return 0
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.exception("Full traceback:")
        return 1
    finally:
        if orchestrator:
            orchestrator.shutdown_system()
        log_system_shutdown()


if __name__ == "__main__":
    sys.exit(main())