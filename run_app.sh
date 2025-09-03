#!/bin/bash

echo "========================================"
echo "Running Unified Smart Object Counter"
echo "========================================"
echo

echo "Activating virtual environment..."
source venv/bin/activate
if [ $? -ne 0 ]; then
    echo "ERROR: Virtual environment not found"
    echo "Please run install.sh first"
    exit 1
fi

echo "Starting application..."
python unified_object_counter.py

if [ $? -ne 0 ]; then
    echo
    echo "ERROR: Application failed to start"
    echo "Please check the error message above"
fi
