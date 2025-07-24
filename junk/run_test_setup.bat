@echo off
echo Running system setup verification...

:: Activate virtual environment
call venv\Scripts\activate.bat

:: Run setup verification
python test_setup.py

:: Keep console open
