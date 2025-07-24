@echo off
echo Cleaning previously trained CPU models...

:: Activate virtual environment
call venv\Scripts\activate.bat

:: Run model cleaning script
python clean_models.py

:: Keep console open
pause