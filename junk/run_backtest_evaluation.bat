@echo off
echo Running Backtest Evaluation...
call venv\Scripts\activate.bat
python run_backtest_evaluation.py %*
if %ERRORLEVEL% NEQ 0 (
    echo Backtest Evaluation failed with error code %ERRORLEVEL%
    exit /b %ERRORLEVEL%
)
echo Backtest Evaluation completed successfully