# Quantitative Trading System Documentation

## System Overview

The Quantitative Trading System is a comprehensive machine learning-based trading platform that combines LSTM neural networks for price prediction with Deep Q-Network (DQN) reinforcement learning for action selection. The system integrates sentiment analysis from news headlines and implements robust risk management to trade Indian stocks with the goal of outperforming the NIFTY 50 benchmark.

## Installation

1. **Clone the repository** (if applicable) or ensure all files are in place

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify system setup**:
   ```bash
   python test_setup.py
   ```

4. **Initialize the system**:
   ```bash
   python main.py --mode setup
   ```

## System Architecture

The system follows a modular architecture with the following key components:

1. **Data Collection Module**: Fetches market data and news from yfinance API
2. **Data Processing Module**: Cleans and validates raw market data
3. **Feature Engineering Module**: Calculates technical indicators and sentiment scores
4. **Model Training Module**: Trains LSTM and DQN models
5. **Trading Strategy Module**: Executes trading decisions using trained models
6. **Risk Management Module**: Implements position sizing and stop-loss controls
7. **Backtesting Engine**: Simulates trading strategy on historical data
8. **Performance Monitor**: Tracks and visualizes system performance

## Usage Guide

### System Modes

The system can be operated in different modes using the main.py script:

- **Setup Mode**: Initialize and validate system configuration
  ```bash
  python main.py --mode setup
  ```

- **Data Collection**: Fetch historical market data
  ```bash
  python main.py --mode collect
  ```

- **Model Training**: Train LSTM and DQN models
  ```bash
  python main.py --mode train
  ```

- **Backtesting**: Run historical performance evaluation
  ```bash
  python main.py --mode backtest
  ```

- **Complete Workflow**: Run the entire pipeline from data collection to backtesting
  ```bash
  python main.py --mode complete
  ```

- **Status Check**: Display current system status
  ```bash
  python main.py --mode status
  ```

### Environment Configuration

The system supports different environments:

- **Development** (default): Full logging, smaller datasets for testing
- **Testing**: Optimized for automated testing
- **Production**: Minimal logging, full datasets

```bash
python main.py --env production --mode backtest
```

## Data Collection

The data collection module fetches historical stock data and news headlines from Yahoo Finance:

1. **Stock Data**: OHLCV (Open, High, Low, Close, Volume) data for 500+ Indian stocks
2. **News Headlines**: Company-specific news for sentiment analysis
3. **Date Range**: Data from January 1995 to current date
4. **Stock Universe**: NIFTY 50, mid-cap, and small-cap stocks

Data is stored in CSV format in the `data/` directory with the following structure:
- `data/stocks/`: Stock price data
- `data/news/`: News headlines
- `data/indicators/`: Calculated technical indicators
- `data/performance/`: Performance metrics and reports

## Feature Engineering

The system calculates the following technical indicators:

1. **14-day RSI**: Relative Strength Index for overbought/oversold conditions
2. **50-day SMA**: Simple Moving Average for trend identification
3. **20-day Bollinger Bands**: Upper, middle, and lower bands for volatility

Sentiment analysis is performed using a pre-trained DistilBERT model:
- Scores range from -1 (negative) to +1 (positive)
- Neutral sentiment (0) is assigned when news data is unavailable

## Model Training

### LSTM Model

The LSTM model is used for price prediction with the following architecture:
- 2 LSTM layers with 50 units each
- 0.2 dropout rate for regularization
- Adam optimizer with learning rate scheduling
- Output: Next-day price movement probability

### DQN Agent

The DQN agent optimizes trading decisions with:
- Three actions: buy, sell, hold
- Reward function based on Sharpe ratio improvement
- State representation using LSTM predictions and sentiment scores
- Experience replay and target network for stable learning

## Risk Management

The system implements comprehensive risk controls:

1. **Stop-Loss**: 5% below entry price
2. **Position Sizing**: 1-2% of capital per trade
3. **Portfolio Diversification**: 20-30 stocks maximum
4. **Sentiment Adjustment**: ±20% position size modification based on sentiment

## Backtesting

The backtesting engine simulates trading with:

1. **Initial Capital**: ₹10,00,000
2. **Transaction Costs**: 0.1% per trade
3. **No Leverage**: Trading with available capital only
4. **Historical Data**: Testing on 2020-2025 data after training on 1995-2019
5. **Benchmark Comparison**: Performance compared against NIFTY 50

## Performance Metrics

The system targets the following performance metrics:

1. **Annualized Return**: 15-20% (vs. NIFTY 50's ~10%)
2. **Sharpe Ratio**: >1.8
3. **Maximum Drawdown**: <8%
4. **Win Rate**: 60-65%

Performance reports are generated after backtesting and stored in `data/performance/`.

## System Monitoring

The system provides comprehensive monitoring:

1. **Logging**: Detailed logs in the `logs/` directory
2. **Health Monitoring**: CPU, memory, and disk usage tracking
3. **Performance Alerts**: Notifications for metric deviations
4. **System Status**: Real-time status reporting

## Optimization Guidelines

To optimize system performance:

1. **Data Quality**: Ensure clean, complete historical data
2. **Feature Selection**: Focus on most predictive indicators
3. **Model Hyperparameters**: Tune LSTM and DQN parameters
4. **Risk Parameters**: Adjust position sizing and stop-loss levels
5. **Stock Universe**: Select stocks with sufficient liquidity and data

## Troubleshooting

Common issues and solutions:

1. **Data Collection Failures**:
   - Check internet connectivity
   - Verify API rate limits
   - Ensure stock symbols are valid

2. **Model Training Issues**:
   - Check for sufficient data
   - Verify feature quality
   - Adjust learning rate or batch size

3. **Performance Below Targets**:
   - Review risk parameters
   - Check for overfitting in models
   - Analyze trade logs for patterns

4. **System Errors**:
   - Check log files for detailed error messages
   - Verify system dependencies
   - Ensure sufficient disk space and memory

## Integration Testing

Run the integration tests to verify system functionality:

```bash
python run_integration_tests.py
```

This will test:
1. Data collection workflow
2. Model training workflow
3. Backtesting workflow
4. Complete system workflow
5. Performance validation

## Best Practices

1. **Regular Updates**: Update stock data daily for best results
2. **Model Retraining**: Retrain models monthly with new data
3. **Performance Monitoring**: Review metrics weekly
4. **Risk Management**: Never disable stop-loss controls
5. **Diversification**: Maintain positions across multiple sectors

## Disclaimer

This software is for educational and research purposes only. Trading involves substantial risk of loss and is not suitable for all investors. Past performance does not guarantee future results. Always consult with a qualified financial advisor before making investment decisions.