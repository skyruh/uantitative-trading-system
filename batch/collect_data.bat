@echo off
echo Enhanced Data Collection for Trading System...

:: Use absolute paths
set "PROJECT_ROOT=%~dp0.."
cd /d "%PROJECT_ROOT%"

:: Activate virtual environment
call venv\Scripts\activate.bat

:: Run the enhanced data collector
python src/data_collector.py config/indian_stocks.txt

:: Keep console open
pause