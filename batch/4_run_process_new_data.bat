@echo off
echo Processing new stock data from data_518 folder...

:: Change to parent directory
cd ..

:: Activate virtual environment
call venv\Scripts\activate.bat

:: Run data processing script
python process_new_data.py

:: Keep console open
pause