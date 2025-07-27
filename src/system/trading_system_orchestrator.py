"""
Trading System Orchestrator - Main system integration and workflow management.
Coordinates all components for end-to-end trading system operation.
"""

import os
import sys
import time
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd

from src.config.settings import config
from src.utils.logging_utils import get_logger
from src.data.stock_data_fetcher import StockDataFetcher
from src.data.data_processor import DataProcessor
from src.data.data_storage import DataStorage
from src.data.feature_builder import FeatureBuilder
from src.models.lstm_trainer import LSTMTrainer
from src.models.lstm_model import LSTMModel
from src.models.dqn_trainer import DQNTrainer
from src.models.dqn_agent import DQNAgent
from src.trading.signal_generator import SignalGenerator
from src.trading.position_manager import PositionManager
from src.risk.risk_manager import RiskManager
from src.backtesting.backtest_engine import BacktestEngine
from src.monitoring.performance_tracker import PerformanceTracker
from src.monitoring.system_health import SystemHealthMonitor
from src.monitoring.visualization import TradingVisualizer


class TradingSystemOrchestrator:
    """
    Main orchestrator that coordinates all system components for end-to-end operation.
    Manages the complete workflow from data collection to performance reporting.
    """
    
    def __init__(self, environment: str = "development"):
        """Initialize the trading system orchestrator."""
        self.environment = environment
        config.environment = environment
        self.logger = get_logger(f"TradingSystemOrchestrator-{environment}")
        
        # Initialize system health monitor
        self.health_monitor = SystemHealthMonitor()
        
        # Initialize core components
        self._initialize_components()
        
        # System state tracking
        self.system_state = {
            "initialized": False,
            "data_collected": False,
            "models_trained": False,
            "backtest_completed": False,
            "ready_for_trading": False
        }
        
        self.logger.info(f"Trading System Orchestrator initialized for {environment} environment")
    
    def _initialize_components(self):
        """Initialize all system components."""
        try:
            # Data components
            self.data_storage = DataStorage()
            self.stock_fetcher = StockDataFetcher(data_storage=self.data_storage)
            self.data_processor = DataProcessor()
            self.feature_builder = FeatureBuilder()
            
            # Model components
            lstm_model = LSTMModel()
            self.lstm_trainer = LSTMTrainer(model=lstm_model)
            dqn_agent = DQNAgent()
            self.dqn_trainer = DQNTrainer(dqn_agent=dqn_agent, lstm_model=lstm_model)
            
            # Trading components
            self.signal_generator = SignalGenerator()
            self.position_manager = PositionManager()
            self.risk_manager = RiskManager()
            
            # Load trained models into signal generator
            self._load_models_into_signal_generator()
            
            # Import and create trading strategy
            from src.trading.simple_strategy import SimpleTradingStrategy
            self.trading_strategy = SimpleTradingStrategy(
                self.signal_generator, 
                self.position_manager,
                allow_short_selling=config.backtest.allow_short_selling
            )
            
            # Analysis components
            self.backtest_engine = BacktestEngine()
            self.performance_tracker = PerformanceTracker()
            self.visualizer = TradingVisualizer()
            
            self.logger.info("All system components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {str(e)}")
            raise
    
    def startup_system(self) -> bool:
        """
        Complete system startup sequence with validation.
        Returns True if startup successful, False otherwise.
        """
        self.logger.info("Starting trading system startup sequence...")
        
        try:
            # 1. Validate system dependencies and configuration
            if not self._validate_system_setup():
                return False
            
            # 2. Initialize directories and logging
            self._setup_directories()
            
            # 3. Validate component health
            if not self.health_monitor.check_system_health():
                self.logger.error("System health check failed")
                return False
            
            # 4. Load existing data and models if available
            self._load_existing_state()
            
            self.system_state["initialized"] = True
            self.logger.info("Trading system startup completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"System startup failed: {str(e)}")
            self.logger.exception("Startup error details:")
            return False
    
    def run_data_collection_workflow(self) -> bool:
        """
        Execute complete data collection workflow.
        Returns True if successful, False otherwise.
        """
        self.logger.info("Starting data collection workflow...")
        
        try:
            # Get stock symbols to process
            symbols = config.get_stock_symbols()
            self.logger.info(f"Processing {len(symbols)} stock symbols")
            
            # 1. Collect stock price data
            self.logger.info("Collecting stock price data...")
            start_date = config.data.start_date
            end_date = config.data.end_date
            stock_data_success = self.stock_fetcher.fetch_all_stocks_data(
                start_date=start_date,
                end_date=end_date,
                symbols=symbols
            )
            
            if not stock_data_success:
                self.logger.error("Stock data collection failed")
                return False
            
            # News data collection has been removed
            self.logger.info("Skipping news data collection (sentiment analysis removed)")
            
            # 3. Process and clean collected data
            self.logger.info("Processing and cleaning data...")
            processing_success = True
            
            # Process each stock's data individually
            for symbol in symbols:
                try:
                    # Load stock data
                    stock_data = self.data_storage.load_stock_data(symbol)
                    if stock_data is not None and not stock_data.empty:
                        # Process the data
                        processed_data, stats = self.data_processor.process_stock_data(stock_data, symbol)
                        
                        # Save processed data
                        if not self.data_storage.save_processed_data(symbol, processed_data):
                            self.logger.warning(f"Failed to save processed data for {symbol}")
                            processing_success = False
                    else:
                        self.logger.warning(f"No data available for {symbol}")
                except Exception as e:
                    self.logger.error(f"Error processing data for {symbol}: {str(e)}")
                    processing_success = False
            
            if not processing_success:
                self.logger.error("Data processing failed")
                return False
            
            # 4. Build features (technical indicators + sentiment)
            self.logger.info("Building features...")
            feature_success = self.feature_builder.build_features_for_all_stocks(symbols)
            
            if not feature_success:
                self.logger.error("Feature building failed")
                return False
            
            # Sentiment analysis has been removed
            self.logger.info("Skipping sentiment integration (sentiment analysis removed)")
            
            self.system_state["data_collected"] = True
            self.logger.info("Data collection workflow completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Data collection workflow failed: {str(e)}")
            self.logger.exception("Data collection error details:")
            return False
    
    def run_model_training_workflow(self) -> bool:
        """
        Execute complete model training workflow.
        Returns True if successful, False otherwise.
        """
        self.logger.info("Starting model training workflow...")
        
        if not self.system_state["data_collected"]:
            self.logger.error("Cannot train models - data collection not completed")
            return False
        
        try:
            symbols = config.get_stock_symbols()
            
            # 1. Train LSTM models for price prediction
            self.logger.info("Training LSTM models...")
            lstm_success = self.lstm_trainer.train_models_for_all_stocks(symbols)
            
            if not lstm_success:
                self.logger.error("LSTM training failed")
                return False
            
            # 2. Train DQN agent for trading decisions
            self.logger.info("Training DQN agent...")
            
            # Load processed data for DQN training
            training_data = {}
            for symbol in symbols:
                try:
                    # Try different naming conventions for the processed file
                    base_symbol = symbol.split('.')[0] if '.' in symbol else symbol
                    
                    # Check for file with exact symbol name
                    data_path = f"data/processed/{symbol.replace('.', '_')}_processed.csv"
                    
                    # If not found, try with base symbol (without .NS)
                    if not os.path.exists(data_path):
                        data_path = f"data/processed/{base_symbol}_processed.csv"
                    
                    # If still not found, skip this symbol
                    if not os.path.exists(data_path):
                        self.logger.warning(f"No processed data found for DQN training: {symbol}")
                        continue
                    
                    # Load data
                    data = pd.read_csv(data_path)
                    if not data.empty:
                        training_data[symbol] = data
                        
                except Exception as e:
                    self.logger.warning(f"Error loading processed data for {symbol}: {str(e)}")
                    continue
            
            if training_data:
                dqn_success = self.dqn_trainer.train(training_data)
            else:
                self.logger.error("No processed data available for DQN training")
                dqn_success = False
            
            if not dqn_success:
                self.logger.error("DQN training failed")
                return False
            
            # 3. Validate trained models
            self.logger.info("Validating trained models...")
            validation_success = self._validate_trained_models(symbols)
            
            if not validation_success:
                self.logger.error("Model validation failed")
                return False
            
            self.system_state["models_trained"] = True
            self.logger.info("Model training workflow completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Model training workflow failed: {str(e)}")
            self.logger.exception("Model training error details:")
            return False
    
    def run_backtesting_workflow(self) -> bool:
        """
        Execute complete backtesting workflow.
        Returns True if successful, False otherwise.
        """
        self.logger.info("Starting backtesting workflow...")
        
        if not self.system_state["models_trained"]:
            self.logger.error("Cannot run backtesting - models not trained")
            return False
        
        try:
            symbols = config.get_stock_symbols()
            
            # 1. Initialize backtesting engine
            self.logger.info("Initializing backtesting engine...")
            backtest_init_success = self.backtest_engine.initialize_backtest(
                start_date=config.backtest.test_start_date,
                end_date=config.backtest.test_end_date,
                initial_capital=config.backtest.initial_capital
            )
            
            if not backtest_init_success:
                self.logger.error("Backtesting initialization failed")
                return False
            
            # 1.5. Load market data for backtesting
            self.logger.info("Loading market data for backtesting...")
            market_data = self._load_backtest_market_data(symbols)
            if not market_data:
                self.logger.error("Failed to load market data for backtesting")
                return False
            
            data_load_success = self.backtest_engine.load_market_data(market_data)
            if not data_load_success:
                self.logger.error("Failed to load market data into backtest engine")
                return False
            
            # 2. Run backtesting simulation
            self.logger.info("Running backtesting simulation...")
            backtest_results = self.backtest_engine.run_backtest(self.trading_strategy)
            
            if backtest_results is None:
                self.logger.error("Backtesting simulation failed")
                return False
            
            # 3. Calculate performance metrics
            self.logger.info("Calculating performance metrics...")
            performance_metrics = self.performance_tracker.calculate_real_time_metrics()
            
            # 4. Generate performance report
            self.logger.info("Generating performance report...")
            report_success = self._generate_performance_report(performance_metrics, backtest_results)
            
            if not report_success:
                self.logger.warning("Performance report generation had issues")
            
            # 5. Create visualizations
            self.logger.info("Creating performance visualizations...")
            
            try:
                # Convert PerformanceMetrics dataclass to dict for visualization
                if hasattr(performance_metrics, '__dict__'):
                    metrics_dict = performance_metrics.__dict__
                elif hasattr(performance_metrics, '_asdict'):
                    metrics_dict = performance_metrics._asdict()
                else:
                    metrics_dict = performance_metrics if isinstance(performance_metrics, dict) else {}
                
                viz_success = self.visualizer.create_backtest_visualizations(
                    backtest_results, metrics_dict
                )
                
                if viz_success:
                    self.logger.info("Performance visualizations created successfully")
                else:
                    self.logger.info("No visualizations were created (likely due to insufficient data)")
                    
            except Exception as e:
                self.logger.info(f"Visualization creation skipped due to technical issue. Core backtesting completed successfully.")
                self.logger.debug(f"Visualization error details: {e}")
            
            self.system_state["backtest_completed"] = True
            self.system_state["ready_for_trading"] = self._validate_performance_targets(performance_metrics)
            
            self.logger.info("Backtesting workflow completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Backtesting workflow failed: {str(e)}")
            self.logger.exception("Backtesting error details:")
            return False
    
    def run_complete_system_workflow(self) -> bool:
        """
        Execute the complete end-to-end system workflow.
        Returns True if all workflows successful, False otherwise.
        """
        self.logger.info("Starting complete system workflow...")
        
        start_time = time.time()
        
        try:
            # 1. System startup
            if not self.startup_system():
                self.logger.error("System startup failed")
                return False
            
            # 2. Data collection
            if not self.run_data_collection_workflow():
                self.logger.error("Data collection workflow failed")
                return False
            
            # 3. Model training
            if not self.run_model_training_workflow():
                self.logger.error("Model training workflow failed")
                return False
            
            # 4. Backtesting
            if not self.run_backtesting_workflow():
                self.logger.error("Backtesting workflow failed")
                return False
            
            # 5. Final system validation
            final_validation = self._perform_final_system_validation()
            
            end_time = time.time()
            total_time = end_time - start_time
            
            self.logger.info(f"Complete system workflow finished in {total_time:.2f} seconds")
            
            if final_validation:
                self.logger.info("System is ready for trading operations")
            else:
                self.logger.warning("System validation issues detected - review before trading")
            
            return final_validation
            
        except Exception as e:
            self.logger.error(f"Complete system workflow failed: {str(e)}")
            self.logger.exception("System workflow error details:")
            return False
    
    def shutdown_system(self):
        """Graceful system shutdown."""
        self.logger.info("Initiating system shutdown...")
        
        try:
            # Save current system state
            self._save_system_state()
            
            # Close any open resources
            self._cleanup_resources()
            
            self.logger.info("System shutdown completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error during system shutdown: {str(e)}")
    
    def get_system_status(self) -> Dict:
        """Get current system status and health information."""
        health_status = self.health_monitor.get_system_status()
        
        return {
            "system_state": self.system_state,
            "environment": self.environment,
            "health_status": health_status,
            "configuration": config.to_dict(),
            "timestamp": datetime.now().isoformat()
        }
    
    def _validate_system_setup(self) -> bool:
        """Validate system setup and configuration."""
        try:
            # Validate configuration
            if not config.validate_config():
                self.logger.error("Configuration validation failed")
                return False
            
            # Check required directories
            required_dirs = [
                config.data.data_directory,
                config.model.model_save_directory,
                config.logging.log_directory
            ]
            
            for directory in required_dirs:
                if not os.path.exists(directory):
                    self.logger.warning(f"Directory {directory} does not exist, will be created")
            
            return True
            
        except Exception as e:
            self.logger.error(f"System setup validation failed: {str(e)}")
            return False
    
    def _setup_directories(self):
        """Create required directories if they don't exist."""
        directories = [
            config.data.data_directory,
            config.model.model_save_directory,
            config.logging.log_directory,
            "data/stocks",
            "data/news",
            "data/indicators",
            "data/performance",
            "data/backups",
            "models/checkpoints"
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def _load_models_into_signal_generator(self):
        """Load trained models into the signal generator."""
        try:
            # Load LSTM model for RELIANCE (the only trained model we have)
            lstm_model = None
            dqn_agent = None
            
            # Try to load LSTM model
            model_dir = "models/RELIANCE"
            if os.path.exists(model_dir):
                lstm_model_file = os.path.join(model_dir, "lstm_model.keras")
                if os.path.exists(lstm_model_file):
                    try:
                        lstm_model = LSTMModel()
                        lstm_model.load_model(lstm_model_file)
                        self.logger.info("Loaded LSTM model for signal generation")
                    except Exception as e:
                        self.logger.warning(f"Failed to load LSTM model: {e}")
            
            # Try to load DQN agent (if available)
            dqn_model_file = "models/dqn_agent.pkl"
            if os.path.exists(dqn_model_file):
                try:
                    dqn_agent = DQNAgent()
                    dqn_agent.load_model(dqn_model_file)
                    self.logger.info("Loaded DQN agent for signal generation")
                except Exception as e:
                    self.logger.warning(f"Failed to load DQN agent: {e}")
            
            # Set models in signal generator if available
            if lstm_model or dqn_agent:
                if lstm_model and dqn_agent:
                    self.signal_generator.set_models(lstm_model, dqn_agent)
                    self.logger.info("Both LSTM and DQN models loaded into signal generator")
                elif lstm_model:
                    self.signal_generator.lstm_model = lstm_model
                    self.logger.info("LSTM model loaded into signal generator")
                elif dqn_agent:
                    self.signal_generator.dqn_agent = dqn_agent
                    self.logger.info("DQN agent loaded into signal generator")
            else:
                self.logger.warning("No trained models found to load into signal generator")
                
        except Exception as e:
            self.logger.error(f"Error loading models into signal generator: {e}")

    def _load_existing_state(self):
        """Load existing system state if available."""
        try:
            # Check if models exist
            model_dir = config.model.model_save_directory
            models_found = False
            
            # Check for individual symbol model directories
            if os.path.exists(model_dir):
                for item in os.listdir(model_dir):
                    item_path = os.path.join(model_dir, item)
                    if os.path.isdir(item_path) and item != 'checkpoints':
                        # Check if this directory contains a trained model
                        model_file = os.path.join(item_path, 'lstm_model.keras')
                        if os.path.exists(model_file):
                            models_found = True
                            break
            
            if models_found:
                self.system_state["models_trained"] = True
                self.logger.info("Found existing trained models")
            
            # Check if data exists
            data_dir = config.data.data_directory
            if os.path.exists(f"{data_dir}/stocks") and os.listdir(f"{data_dir}/stocks"):
                self.system_state["data_collected"] = True
                self.logger.info("Found existing data")
            
        except Exception as e:
            self.logger.warning(f"Error loading existing state: {str(e)}")
    
    def _validate_trained_models(self, symbols: List[str]) -> bool:
        """Validate that trained models are working correctly."""
        try:
            # Test LSTM model predictions
            lstm_test_success = self.lstm_trainer.validate_models(symbols[:5])  # Test subset
            
            # Test DQN agent (skip validation for now - method not implemented)
            dqn_test_success = True  # TODO: Implement DQN validation method
            self.logger.info("DQN validation skipped - method not implemented")
            
            return lstm_test_success and dqn_test_success
            
        except Exception as e:
            self.logger.error(f"Model validation failed: {str(e)}")
            return False
    
    def _generate_performance_report(self, metrics, backtest_results: Dict) -> bool:
        """Generate comprehensive performance report."""
        try:
            report_path = f"data/performance/performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            
            with open(report_path, 'w') as f:
                f.write("QUANTITATIVE TRADING SYSTEM - PERFORMANCE REPORT\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Environment: {self.environment}\n\n")
                
                f.write("PERFORMANCE METRICS\n")
                f.write("-" * 30 + "\n")
                
                # Convert PerformanceMetrics dataclass to dict if needed
                if hasattr(metrics, '__dict__'):
                    metrics_dict = metrics.__dict__
                elif hasattr(metrics, '_asdict'):
                    metrics_dict = metrics._asdict()
                else:
                    metrics_dict = metrics if isinstance(metrics, dict) else {}
                
                for key, value in metrics_dict.items():
                    f.write(f"{key}: {value}\n")
                
                f.write("\nSYSTEM CONFIGURATION\n")
                f.write("-" * 30 + "\n")
                config_dict = config.to_dict()
                for section, values in config_dict.items():
                    f.write(f"\n{section.upper()}:\n")
                    if isinstance(values, dict):
                        for k, v in values.items():
                            f.write(f"  {k}: {v}\n")
            
            self.logger.info(f"Performance report saved to {report_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to generate performance report: {str(e)}")
            return False
    
    def _validate_performance_targets(self, metrics) -> bool:
        """Validate if performance meets target criteria."""
        try:
            targets_met = []
            
            # Convert PerformanceMetrics dataclass to dict if needed
            if hasattr(metrics, '__dict__'):
                metrics_dict = metrics.__dict__
            elif hasattr(metrics, '_asdict'):
                metrics_dict = metrics._asdict()
            else:
                metrics_dict = metrics if isinstance(metrics, dict) else {}
            
            # Check annual return target
            if metrics_dict.get('annualized_return', 0) >= config.performance.target_annual_return:
                targets_met.append("Annual Return")
            
            # Check Sharpe ratio target
            if metrics_dict.get('sharpe_ratio', 0) >= config.performance.target_sharpe_ratio:
                targets_met.append("Sharpe Ratio")
            
            # Check max drawdown limit
            if metrics_dict.get('max_drawdown', 1) <= config.performance.max_drawdown_limit:
                targets_met.append("Max Drawdown")
            
            # Check win rate target
            if metrics_dict.get('win_rate', 0) >= config.performance.target_win_rate:
                targets_met.append("Win Rate")
            
            self.logger.info(f"Performance targets met: {targets_met}")
            
            # Require at least 3 out of 4 targets to be met
            return len(targets_met) >= 3
            
        except Exception as e:
            self.logger.error(f"Performance validation failed: {str(e)}")
            return False
    
    def _perform_final_system_validation(self) -> bool:
        """Perform final comprehensive system validation."""
        try:
            validation_results = []
            
            # Check all system states
            required_states = ["initialized", "data_collected", "models_trained", "backtest_completed"]
            for state in required_states:
                if self.system_state.get(state, False):
                    validation_results.append(f"✓ {state}")
                else:
                    validation_results.append(f"✗ {state}")
            
            # Check system health
            health_status = self.health_monitor.check_system_health()
            validation_results.append(f"{'✓' if health_status else '✗'} system_health")
            
            # Log validation results
            self.logger.info("Final System Validation Results:")
            for result in validation_results:
                self.logger.info(f"  {result}")
            
            # System is valid if all required states are True and health is good
            return all(self.system_state[state] for state in required_states) and health_status
            
        except Exception as e:
            self.logger.error(f"Final system validation failed: {str(e)}")
            return False
    
    def _save_system_state(self):
        """Save current system state to file."""
        try:
            state_file = "data/system_state.json"
            import json
            
            with open(state_file, 'w') as f:
                json.dump({
                    "system_state": self.system_state,
                    "timestamp": datetime.now().isoformat(),
                    "environment": self.environment
                }, f, indent=2)
            
        except Exception as e:
            self.logger.warning(f"Failed to save system state: {str(e)}")
    
    def _load_backtest_market_data(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Load market data for backtesting.
        
        Args:
            symbols: List of stock symbols to load data for
            
        Returns:
            Dictionary mapping symbols to their market data DataFrames
        """
        try:
            market_data = {}
            
            for symbol in symbols:
                try:
                    # Try to load processed data first
                    base_symbol = symbol.split('.')[0] if '.' in symbol else symbol
                    processed_file = f"data/processed/{base_symbol}_processed.csv"
                    
                    if os.path.exists(processed_file):
                        data = pd.read_csv(processed_file)
                        if not data.empty:
                            # Ensure we have required columns for backtesting
                            required_cols = ['close', 'volume', 'high', 'low', 'open']
                            if all(col in data.columns for col in required_cols):
                                # Rename columns to match BacktestEngine expectations (capital letters)
                                column_mapping = {
                                    'open': 'Open',
                                    'high': 'High', 
                                    'low': 'Low',
                                    'close': 'Close',
                                    'volume': 'Volume'
                                }
                                data = data.rename(columns=column_mapping)
                                
                                # Set date as index if it exists
                                if 'date' in data.columns:
                                    data['Date'] = pd.to_datetime(data['date'])
                                    # Convert timezone-aware datetime to timezone-naive for backtesting compatibility
                                    if data['Date'].dt.tz is not None:
                                        data['Date'] = data['Date'].dt.tz_localize(None)
                                    data = data.set_index('Date')
                                
                                market_data[symbol] = data
                                self.logger.debug(f"Loaded processed data for {symbol}: {len(data)} rows")
                            else:
                                self.logger.warning(f"Missing required columns in processed data for {symbol}")
                        else:
                            self.logger.warning(f"Empty processed data for {symbol}")
                    else:
                        # Try to load raw stock data as fallback
                        raw_data = self.data_storage.load_stock_data(symbol)
                        if raw_data is not None and not raw_data.empty:
                            market_data[symbol] = raw_data
                            self.logger.debug(f"Loaded raw data for {symbol}: {len(raw_data)} rows")
                        else:
                            self.logger.warning(f"No data available for {symbol}")
                            
                except Exception as e:
                    self.logger.error(f"Error loading market data for {symbol}: {str(e)}")
                    continue
            
            self.logger.info(f"Loaded market data for {len(market_data)} symbols")
            

            return market_data
            
        except Exception as e:
            self.logger.error(f"Error loading backtest market data: {str(e)}")
            return {}
    
    def _cleanup_resources(self):
        """Clean up system resources."""
        try:
            # Close any open file handles, database connections, etc.
            # This is a placeholder for resource cleanup
            pass
            
        except Exception as e:
            self.logger.warning(f"Error during resource cleanup: {str(e)}")


