@echo off
echo ========================================
echo Installing Unified Smart Object Counter
echo ========================================
echo.

echo Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

echo Python found. Checking pip...
python -m pip --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: pip is not available
    echo Please ensure pip is installed with Python
    pause
    exit /b 1
)

echo Creating virtual environment...
if exist "venv" (
    echo Virtual environment already exists. Removing old one...
    rmdir /s /q "venv"
)

python -m venv venv
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)

echo Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)

echo Upgrading pip...
python -m pip install --upgrade pip

echo Installing packages...
echo This may take several minutes...
python -m pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo WARNING: Some packages failed to install
    echo Trying alternative installation method...
    echo.
    
    echo Installing packages individually...
    python -m pip install opencv-python
    python -m pip install numpy
    python -m pip install Pillow
    python -m pip install scikit-learn
    python -m pip install ultralytics
    
    if errorlevel 1 (
        echo.
        echo ERROR: Installation failed
        echo Please check your internet connection and try again
        pause
        exit /b 1
    )
)

echo.
echo ========================================
echo Installation completed successfully!
echo ========================================
echo.
echo To run the application:
echo 1. Activate the virtual environment: venv\Scripts\activate.bat
echo 2. Run: python unified_object_counter.py
echo.
echo Or simply double-click: run_app.bat
echo.
pause
