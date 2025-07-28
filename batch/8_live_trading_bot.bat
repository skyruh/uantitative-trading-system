@echo off
echo Starting Live Trading System

set "PROJECT_ROOT=%~dp0.."
cd /d "%PROJECT_ROOT%"

call venv\Scripts\activate.bat

echo configure the Trading mode
::Configure trading mode
python configure_trading.py

::start live paper trading
python main.py --mode live

:: keep console open 
pause