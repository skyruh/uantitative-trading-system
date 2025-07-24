@echo off
echo Running system setup...

:: Use absolute paths
set "PROJECT_ROOT=%~dp0.."
cd /d "%PROJECT_ROOT%"

:: Activate virtual environment
call venv\Scripts\activate.bat

:: Run setup mode
python main.py --mode setup

:: Keep console open
pause