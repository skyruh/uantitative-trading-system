@echo off
echo Running data collection...

:: Use absolute paths
set "PROJECT_ROOT=%~dp0.."
cd /d "%PROJECT_ROOT%"

:: Activate virtual environment
call venv\Scripts\activate.bat

:: Run data collection mode
python main.py --mode collect

:: Keep console open
pause