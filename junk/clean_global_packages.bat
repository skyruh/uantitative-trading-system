@echo off
echo This script will uninstall trading system packages from the global Python environment.
echo These packages will be installed in the virtual environment instead.
echo.
echo WARNING: This will remove these packages from your global Python installation.
echo If other projects depend on these packages, they may stop working.
echo.
set /p confirm=Are you sure you want to continue? (y/n): 

if /i not "%confirm%"=="y" (
    echo Operation cancelled.
    goto :end
)

echo.
echo Uninstalling packages from global environment...

:: List of packages to uninstall from global environment
pip uninstall -y pandas numpy scipy yfinance tensorflow scikit-learn sklearn transformers torch matplotlib seaborn plotly backtrader tqdm python-dateutil pytz pytest pytest-cov black flake8 jupyter ipykernel

echo.
echo Packages have been uninstalled from the global environment.
echo Please run setup_env.bat to install them in the virtual environment.

:end
