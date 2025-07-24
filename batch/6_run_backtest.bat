@echo off
echo Running backtesting...

:: Use absolute paths
set "PROJECT_ROOT=%~dp0.."
cd /d "%PROJECT_ROOT%"

:: Activate virtual environment
call venv\Scripts\activate.bat

:: Run backtesting mode
python main.py --mode backtest

:: Keep console open
pause