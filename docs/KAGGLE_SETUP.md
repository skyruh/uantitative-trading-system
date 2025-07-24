# Kaggle Setup Guide

Complete guide to set up and run your trading system on Kaggle with GitHub and Google Drive integration.

## 🚀 Quick Start

### 1. Fork Repository
1. Go to your GitHub repository
2. Click "Fork" to create your own copy
3. Clone to your local machine for development

### 2. Upload Data to Google Drive
Create this folder structure in Google Drive:
```
/My Drive/TradingSystem/
├── data/
│   ├── processed/          # Upload your processed CSV files here
│   │   ├── RELIANCE_processed.csv
│   │   ├── TCS_processed.csv
│   │   └── ... (all your processed stock data)
│   └── raw/               # Optional: raw data backup
├── models/
│   ├── lstm/              # LSTM models will be saved here
│   └── dqn/               # DQN models will be saved here
├── results/               # Training results and logs
└── visualizations/        # Generated charts and graphs
```

### 3. Create Kaggle Notebook
1. Go to kaggle.com/code
2. Click "New Notebook"
3. Enable GPU: Settings → Accelerator → GPU T4 x2
4. Copy content from `notebooks/kaggle_lstm_training.ipynb`

## 📊 Training Workflow

### Phase 1: Data Processing (Optional)
If you haven't processed your data yet:
```python
# Use kaggle_data_processing.ipynb
# Processes raw yfinance data into technical indicators
# Time: ~2-4 hours
```

### Phase 2: LSTM Training
```python
# Use kaggle_lstm_training.ipynb
# Trains LSTM models for all stocks
# Time: ~8-12 hours
# Output: Trained LSTM models in Google Drive
```

### Phase 3: DQN Training
```python
# Use kaggle_dqn_training.ipynb
# Trains DQN agent using LSTM models
# Time: ~6-10 hours
# Output: Trained DQN agent in Google Drive
```

### Phase 4: Backtesting
```python
# Use kaggle_backtesting.ipynb
# Comprehensive backtesting and analysis
# Time: ~4-6 hours
# Output: Performance reports and visualizations
```

## ⚙️ Optimization Settings

### GPU-Specific Batch Sizes
```python
# Auto-detection in notebooks
if 'V100' in gpu_name:
    batch_size = 1024
elif 'P100' in gpu_name:
    batch_size = 512
elif 'T4' in gpu_name:
    batch_size = 256
else:
    batch_size = 128
```

### Memory Management
```python
# Clear memory between training sessions
import gc
import tensorflow as tf

tf.keras.backend.clear_session()
gc.collect()

# Enable mixed precision
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)
```

### Session Management
- **Save frequently**: Kaggle sessions can timeout
- **Use checkpoints**: Save model checkpoints every few epochs
- **Monitor time**: Keep track of your 30-hour weekly quota

## 📈 Performance Expectations

### Training Times (Approximate)
| GPU Type | Batch Size | Time per Epoch | Total Training |
|----------|------------|----------------|----------------|
| T4       | 256        | ~1-2 minutes   | 8-12 hours     |
| P100     | 512        | ~45-90 seconds | 6-10 hours     |
| V100     | 1024       | ~30-60 seconds | 4-8 hours      |

### Expected Results
- **Accuracy**: 52-58% (above random baseline of 50%)
- **Training Loss**: Should decrease from ~0.7 to ~0.65
- **Validation Loss**: Should follow training loss closely

## 🔧 Troubleshooting

### Common Issues

#### 1. Out of Memory Error
```python
# Reduce batch size
batch_size = batch_size // 2

# Clear memory more frequently
tf.keras.backend.clear_session()
gc.collect()
```

#### 2. Session Timeout
```python
# Save checkpoints more frequently
checkpoint_callback = ModelCheckpoint(
    filepath='/content/drive/MyDrive/TradingSystem/checkpoints/{symbol}_checkpoint.keras',
    save_best_only=True,
    monitor='val_loss',
    verbose=1
)
```

#### 3. Drive Quota Exceeded
```python
# Clean up old files
import os
import glob

# Remove old checkpoints
old_checkpoints = glob.glob('/content/drive/MyDrive/TradingSystem/checkpoints/*')
for file in old_checkpoints[:-5]:  # Keep only last 5
    os.remove(file)
```

#### 4. Import Errors
```python
# Ensure repository is cloned correctly
!git clone https://github.com/yourusername/trading-system.git
%cd trading-system
!pip install -r requirements.txt

# Add to Python path
import sys
sys.path.append('/kaggle/working/trading-system/src')
```

## 📊 Monitoring Progress

### GPU Usage
```python
# Check GPU utilization
!nvidia-smi

# Monitor memory usage
import psutil
print(f"RAM: {psutil.virtual_memory().percent}%")
print(f"Disk: {psutil.disk_usage('/').percent}%")
```

### Training Progress
```python
# Log training metrics
training_log = {
    'epoch': epoch,
    'loss': history.history['loss'][-1],
    'accuracy': history.history['accuracy'][-1],
    'val_loss': history.history['val_loss'][-1],
    'val_accuracy': history.history['val_accuracy'][-1],
    'timestamp': datetime.now().isoformat()
}

# Save to Drive
with open('/content/drive/MyDrive/TradingSystem/logs/training_progress.json', 'a') as f:
    f.write(json.dumps(training_log) + '\\n')
```

## 💡 Pro Tips

### 1. Maximize GPU Usage
- Use largest possible batch size that fits in memory
- Enable mixed precision training
- Use data generators for large datasets

### 2. Efficient Data Loading
- Keep processed data in Google Drive
- Use pandas chunking for large files
- Cache frequently used data

### 3. Time Management
- Monitor your 30-hour weekly quota
- Use shorter epochs with early stopping
- Save intermediate results frequently

### 4. Experiment Tracking
- Save all hyperparameters
- Log training metrics
- Version your experiments

## 🔄 Continuous Development

### Local Development → Kaggle Training Loop
1. **Develop locally**: Write and test code on your machine
2. **Push to GitHub**: Commit and push changes
3. **Pull in Kaggle**: Update notebook with latest code
4. **Train on Kaggle**: Use powerful GPUs for training
5. **Download results**: Get trained models back to local
6. **Repeat**: Iterate and improve

### Code Updates
```python
# In Kaggle notebook, pull latest changes
!git pull origin main

# Restart kernel to reload modules
# Runtime → Restart Runtime
```

This setup gives you a professional, scalable workflow that maximizes Kaggle's free resources while maintaining code quality and version control!