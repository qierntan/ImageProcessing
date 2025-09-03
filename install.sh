#!/bin/bash

echo "========================================"
echo "Installing Unified Smart Object Counter"
echo "========================================"
echo

echo "Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed"
    echo "Please install Python 3.8+ from https://python.org"
    exit 1
fi

echo "Python found. Checking pip..."
if ! python3 -m pip --version &> /dev/null; then
    echo "ERROR: pip is not available"
    echo "Please ensure pip is installed with Python"
    exit 1
fi

echo "Creating virtual environment..."
if [ -d "venv" ]; then
    echo "Virtual environment already exists. Removing old one..."
    rm -rf venv
fi

python3 -m venv venv
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to create virtual environment"
    exit 1
fi

echo "Activating virtual environment..."
source venv/bin/activate
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate virtual environment"
    exit 1
fi

echo "Upgrading pip..."
python -m pip install --upgrade pip

echo "Installing packages..."
echo "This may take several minutes..."
python -m pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo
    echo "WARNING: Some packages failed to install"
    echo "Trying alternative installation method..."
    echo
    
    echo "Installing packages individually..."
    python -m pip install opencv-python
    python -m pip install numpy
    python -m pip install Pillow
    python -m pip install scikit-learn
    python -m pip install ultralytics
    
    if [ $? -ne 0 ]; then
        echo
        echo "ERROR: Installation failed"
        echo "Please check your internet connection and try again"
        exit 1
    fi
fi

echo
echo "========================================"
echo "Installation completed successfully!"
echo "========================================"
echo
echo "To run the application:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Run: python unified_object_counter.py"
echo
echo "Or simply run: ./run_app.sh"
echo
