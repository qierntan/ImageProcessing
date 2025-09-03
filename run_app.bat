@echo off
echo ========================================
echo Running Unified Smart Object Counter
echo ========================================
echo.

echo Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Virtual environment not found
    echo Please run install.bat first
    pause
    exit /b 1
)

echo Starting application...
python unified_object_counter.py

if errorlevel 1 (
    echo.
    echo ERROR: Application failed to start
    echo Please check the error message above
    pause
)
