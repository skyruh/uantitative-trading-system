@echo off
echo Processing sentiment data for all stocks...

:: Activate virtual environment
call venv\Scripts\activate.bat

:: Run sentiment processing script
python run_sentiment_processing.py

:: Keep console open
pause