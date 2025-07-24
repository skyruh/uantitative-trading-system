@echo off
echo Running System Analysis...

:: Use absolute paths
set "PROJECT_ROOT=%~dp0.."
cd /d "%PROJECT_ROOT%"

call venv\Scripts\activate.bat
python system_analysis.py %*
if %ERRORLEVEL% NEQ 0 (
    echo System Analysis failed with error code %ERRORLEVEL%
    exit /b %ERRORLEVEL%
)
echo System Analysis completed successfully
pause