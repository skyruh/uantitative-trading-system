@echo off
echo Running LSTM Training Test...
call venv\Scripts\activate.bat
python run_lstm_training_test.py %*
if %ERRORLEVEL% NEQ 0 (
    echo LSTM Training Test failed with error code %ERRORLEVEL%
    exit /b %ERRORLEVEL%
)
echo LSTM Training Test completed successfully