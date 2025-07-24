@echo off
echo Checking system status...

:: Activate virtual environment
call venv\Scripts\activate.bat

:: Run status mode
python main.py --mode status

:: Keep console open
