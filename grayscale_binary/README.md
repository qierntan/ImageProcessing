# Smart Object Counting System

A Python GUI application for counting objects in images using template matching and image processing techniques.

## Features

- **Image Upload**: Support for various image formats (JPG, PNG, BMP, TIFF, WebP)
- **Multiple View Modes**: Switch between Original, Grayscale, and Binary image views
- **Interactive ROI Selection**: Click and drag to select regions of interest
- **Template Matching**: Automatically detect and count objects similar to the selected template
- **Real-time Results**: View detection results with bounding boxes and confidence scores

## Requirements

- Python 3.7 or higher
- Required packages (see requirements.txt):
  - opencv-python==4.9.0.80
  - numpy==1.26.4
  - Pillow==10.2.0

## Installation

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the application:
   ```bash
   python grayscale_binary.py
   ```

2. **Upload Image**: Click "📁 Upload Image" to select an image file

3. **Select Template**: 
   - Click and drag on the image to select a sample object
   - The selected region will be used as a template for detection

4. **Count Objects**: Click "🔍 Count Objects" to automatically detect and count similar objects

5. **View Results**: 
   - Switch between different view modes using the dropdown
   - View detection results in the right panel
   - Red bounding boxes show detected objects

## How It Works

1. **Image Processing**: The application converts uploaded images to grayscale and binary formats
2. **Template Matching**: Uses OpenCV's template matching to find objects similar to the selected ROI
3. **Threshold Detection**: Objects with similarity scores above 0.7 are considered matches
4. **Visualization**: Displays results with bounding boxes and confidence scores

## Supported Image Formats

- JPG/JPEG
- PNG
- BMP
- TIFF
- WebP

## Tips for Best Results

- Select a clear, representative sample of the object you want to count
- Ensure good lighting and contrast in your images
- Avoid selecting regions that are too small or too large
- Use the different view modes to better understand your image structure

## Troubleshooting

- **No GUI appears**: Make sure tkinter is available (usually comes with Python)
- **Import errors**: Install required packages using `pip install -r requirements.txt`
- **Poor detection**: Try selecting a different ROI or adjusting image quality
