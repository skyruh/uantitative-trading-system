@echo off
echo Fixing Date Formats in Data Files...
call venv\Scripts\activate.bat
python fix_date_formats.py --known-only
if %ERRORLEVEL% NEQ 0 (
    echo Date format fixing failed with error code %ERRORLEVEL%
    exit /b %ERRORLEVEL%
)
echo Date format fixing completed successfully