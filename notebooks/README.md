# Kaggle Notebooks

This directory contains Jupyter notebooks optimized for Kaggle training.

## 📓 Available Notebooks

### 1. `kaggle_lstm_training.ipynb`
- **Purpose**: Train LSTM models for price prediction
- **GPU Time**: ~8-12 hours for all stocks
- **Batch Size**: 256-512 (optimized for Kaggle GPUs)
- **Output**: Trained LSTM models saved to Google Drive

### 2. `kaggle_dqn_training.ipynb`
- **Purpose**: Train DQN agent for trading decisions
- **GPU Time**: ~6-10 hours
- **Dependencies**: Requires trained LSTM models
- **Output**: Trained DQN agent saved to Google Drive

### 3. `kaggle_backtesting.ipynb`
- **Purpose**: Comprehensive backtesting and performance analysis
- **GPU Time**: ~4-6 hours
- **Dependencies**: Requires both LSTM and DQN models
- **Output**: Performance reports and visualizations

### 4. `kaggle_data_processing.ipynb`
- **Purpose**: Process raw stock data and create technical indicators
- **GPU Time**: ~2-4 hours
- **Input**: Raw stock data from yfinance
- **Output**: Processed data ready for training

## 🚀 Usage Instructions

### Step 1: Setup
1. Create new Kaggle notebook
2. Enable GPU acceleration
3. Clone this repository:
```python
!git clone https://github.com/yourusername/trading-system.git
%cd trading-system
!pip install -r requirements.txt
```

### Step 2: Mount Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```

### Step 3: Run Notebook
- Copy the notebook content to your Kaggle notebook
- Modify paths as needed
- Run all cells

## 📊 Optimization Tips

### Batch Size Recommendations
- **T4 GPU**: batch_size = 256
- **P100 GPU**: batch_size = 512
- **V100 GPU**: batch_size = 1024

### Memory Management
```python
# Clear memory between training sessions
import gc
import tensorflow as tf

tf.keras.backend.clear_session()
gc.collect()
```

### Mixed Precision Training
```python
# Enable mixed precision for faster training
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)
```

## 📁 File Structure After Training

```
/content/drive/MyDrive/TradingSystem/
├── models/
│   ├── lstm/
│   │   ├── RELIANCE_model.keras
│   │   ├── TCS_model.keras
│   │   └── ... (all stock models)
│   └── dqn/
│       └── dqn_agent.keras
├── results/
│   ├── training_logs.json
│   ├── backtest_results.json
│   └── performance_metrics.json
└── visualizations/
    ├── training_curves.png
    ├── backtest_performance.png
    └── portfolio_analysis.png
```

## ⏱️ Training Schedule

To maximize your 30 hours/week GPU quota:

### Week 1: Data Processing + LSTM Training
- **Monday**: Data processing (4 hours)
- **Wednesday**: LSTM training batch 1 (12 hours)
- **Friday**: LSTM training batch 2 (12 hours)
- **Sunday**: Validation and testing (2 hours)

### Week 2: DQN Training + Backtesting
- **Monday**: DQN training (12 hours)
- **Wednesday**: Backtesting (8 hours)
- **Friday**: Analysis and optimization (10 hours)

## 🔧 Troubleshooting

### Common Issues
1. **Out of Memory**: Reduce batch size
2. **Session Timeout**: Save checkpoints frequently
3. **Drive Quota**: Clean up old files regularly

### Performance Monitoring
```python
# Monitor GPU usage
!nvidia-smi

# Monitor memory usage
import psutil
print(f"RAM usage: {psutil.virtual_memory().percent}%")
```