@echo off
echo Removing Sentiment Analysis from System...
call venv\Scripts\activate.bat
python clean_sentiment.py %*
if %ERRORLEVEL% NEQ 0 (
    echo Sentiment cleanup failed with error code %ERRORLEVEL%
    exit /b %ERRORLEVEL%
)
echo Sentiment analysis has been successfully removed from the system