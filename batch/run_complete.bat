@echo off
echo Running complete workflow...

:: Change to parent directory
cd ..

:: Activate virtual environment
call venv\Scripts\activate.bat

:: Run complete workflow mode
python main.py --mode complete

:: Keep console open
pause