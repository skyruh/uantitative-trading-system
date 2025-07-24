@echo off
echo Running model training...

:: Use absolute paths
set "PROJECT_ROOT=%~dp0.."
cd /d "%PROJECT_ROOT%"

:: Activate virtual environment
call venv\Scripts\activate.bat

:: Run training mode
python main.py --mode train

:: Keep console open
pause
