@echo off
echo Running LSTM Training with and without Sentiment Features...
call venv\Scripts\activate.bat
python run_lstm_training.py %*
if %ERRORLEVEL% NEQ 0 (
    echo LSTM Training failed with error code %ERRORLEVEL%
    exit /b %ERRORLEVEL%
)
echo LSTM Training completed successfully