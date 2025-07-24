@echo off
echo Setting up virtual environment for Quantitative Trading System...

:: Change to parent directory
cd ..

:: Create virtual environment if it doesn't exist
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    echo Virtual environment created.
) else (
    echo Virtual environment already exists.
)

:: Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

:: Install requirements
echo Installing dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt

echo.
echo Setup complete! The virtual environment is now activated.
echo Run 'deactivate' to exit the virtual environment when done.
echo.
pause