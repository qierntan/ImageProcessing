@echo off
echo Installing requirements for Unified Object Counter...
echo.

echo Creating virtual environment...
python -m venv venv

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Installing packages...
pip install -r requirements.txt

echo.
echo Installation complete!
echo To run the application, use: python unified_object_counter.py
echo.
pause
