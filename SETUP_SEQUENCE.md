# Quantitative Trading System Setup Sequence

This document outlines the proper sequence for setting up and running the quantitative trading system.

## Setup Sequence

Run these batch files in the following order for a new setup:

1. **setup_env.bat** - Set up the Python environment with required dependencies
   ```
   batch\setup_env.bat
   ```

2. **run_setup.bat** - Initialize the system configuration and create necessary directories
   ```
   batch\run_setup.bat
   ```

3. **run_collect.bat** - Collect stock data from yfinance
   ```
   batch\run_collect.bat
   ```

4. **run_process_new_data.bat** - Process the collected data for model training
   ```
   batch\run_process_new_data.bat
   ```

5. **run_train.bat** - Train LSTM models on the processed data
   ```
   batch\run_train.bat
   ```

6. **run_backtest.bat** - Run backtests on the trained models
   ```
   batch\run_backtest.bat
   ```

7. **run_system_analysis.bat** - Analyze the system performance
   ```
   batch\run_system_analysis.bat
   ```

## All-in-One Option

Alternatively, you can use the all-in-one batch file to run the complete workflow:

```
batch\run_complete.bat
```

This will execute most of the above steps in sequence.

## Additional Operations

- **update_stock_data.py** - Update stock data to the latest date
  ```
  python update_stock_data.py
  ```

- **predict_stock_movement.py** - Make predictions using trained models
  ```
  python predict_stock_movement.py
  ```

- **run_backtest_evaluation.py** - Evaluate backtest results in detail
  ```
  python run_backtest_evaluation.py
  ```

- **validate_system.py** - Validate the system setup and dependencies
  ```
  python validate_system.py
  ```