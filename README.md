# Quantitative Trading System

A comprehensive machine learning-based trading system for Indian stock markets using LSTM and DQN models.

## 🚀 Features

- **LSTM Models**: Price prediction using technical indicators
- **DQN Agent**: Reinforcement learning for trading decisions
- **Technical Indicators**: RSI, MACD, Bollinger Bands, SMA, ATR
- **Backtesting Engine**: Comprehensive performance evaluation
- **Risk Management**: Position sizing and risk controls
- **Multi-Stock Support**: Handles 518+ Indian stocks (NSE)

## 📊 Performance

- **Batch Size Optimization**: Optimized for GPU training (96+ batch size)
- **Early Stopping**: Prevents overfitting with patience-based stopping
- **Mixed Precision**: Faster training with TensorFlow mixed precision
- **Parallel Processing**: Multi-core data processing

## 🛠️ Technology Stack

- **Python 3.8+**
- **TensorFlow 2.x**: Deep learning framework
- **Pandas**: Data manipulation
- **NumPy**: Numerical computing
- **yfinance**: Stock data fetching
- **scikit-learn**: Machine learning utilities

## 📁 Project Structure

```
trading-system/
├── src/                    # Source code
│   ├── models/            # ML models (LSTM, DQN)
│   ├── data/              # Data processing
│   ├── trading/           # Trading logic
│   ├── backtesting/       # Backtesting engine
│   ├── risk/              # Risk management
│   └── utils/             # Utilities
├── notebooks/             # Kaggle/Colab notebooks
├── config/                # Configuration files
├── batch/                 # Batch processing scripts
├── tests/                 # Unit tests
└── docs/                  # Documentation
```

## 🚀 Quick Start

### Local Development
```bash
git clone https://github.com/yourusername/trading-system.git
cd trading-system
pip install -r requirements.txt
```

### Kaggle Training
1. Fork this repository
2. Create Kaggle notebook
3. Clone repository in Kaggle:
```python
!git clone https://github.com/yourusername/trading-system.git
%cd trading-system
!pip install -r requirements.txt
```

## 📈 Usage

### Data Collection
```bash
# Collect stock data
python batch/collect_data.bat

# Process data with technical indicators
python batch/4_run_process_new_data.bat
```

### Model Training
```bash
# Train LSTM and DQN models
python batch/5_run_train.bat

# Or use main orchestrator
python main.py --mode train
```

### Backtesting
```bash
# Run backtesting
python batch/6_run_backtest.bat

# Or use main orchestrator
python main.py --mode backtest
```

## 🔧 Configuration

Edit `config/indian_stocks.txt` to modify stock symbols for training.

## 📊 Model Performance

- **Training Time**: ~4-5 minutes per epoch (local), ~1-2 minutes (Kaggle)
- **Accuracy**: 52-58% (above random baseline)
- **Batch Size**: Optimized for 96+ (local), 256+ (Kaggle)

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This trading system is for educational and research purposes only. Past performance does not guarantee future results. Always do your own research and consider consulting with a financial advisor before making investment decisions.

## 🙏 Acknowledgments

- Yahoo Finance for stock data
- TensorFlow team for the ML framework
- Kaggle for free GPU resources
- Indian stock market data providers