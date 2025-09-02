# Unified Smart Object Counter

A comprehensive object detection and counting application that combines template matching and color segmentation approaches.

## Features

### Template Matching Section
- **Appearance Options**: Choose between grayscale, color, or template matching
- **Detection Flexibility**: Enable rotation detection and/or size scaling detection
- **Combined Detection**: Apply both rotation and scaling simultaneously
- **YOLO Auto-Detection**: Automatically detects objects when loading images

### Color Segmentation Section
- **HSV Color Space**: Hue, Saturation, Value-based detection
- **BGR Color Space**: Blue, Green, Red-based detection
- **Watershed Segmentation**: Split touching objects
- **Morphological Operations**: Noise reduction and object refinement

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Option 1: Automatic Installation (Windows)
1. Double-click `install.bat`
2. Wait for installation to complete
3. Run the application: `python unified_object_counter.py`

### Option 2: Automatic Installation (Linux/Mac)
1. Make the script executable: `chmod +x install.sh`
2. Run the script: `./install.sh`
3. Run the application: `python unified_object_counter.py`

### Option 3: Manual Installation
1. Create a virtual environment:
   ```bash
   python -m venv venv
   ```

2. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Linux/Mac: `source venv/bin/activate`

3. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Select Detection Mode**: Choose between "Template Matching" or "Color Segmentation"
2. **Configure Options**: 
   - For template matching: Choose appearance method and detection flexibility options
   - For color segmentation: Adjust HSV/BGR tolerances, kernel size, and minimum area
3. **Load Image**: Click "Load Image" to select an image file (YOLO will auto-detect objects)
4. **Select Reference**: Click on a detected object or drag a box to select reference
5. **Count Objects**: Click "Count Objects" to analyze the image
6. **View Results**: Results are displayed in the left panel and drawn on the image
7. **Save Results**: Click "Save Result" to save the annotated image

## Supported Image Formats
- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- TIFF (.tiff)
- WebP (.webp)
- AVIF (.avif)

## Dependencies

- **opencv-python**: Computer vision library for image processing
- **numpy**: Numerical computing library
- **Pillow**: Python Imaging Library for image handling
- **scikit-learn**: Machine learning library (for DBSCAN clustering)
- **ultralytics**: YOLO object detection (optional, for enhanced detection)

## File Structure

```
├── unified_object_counter.py    # Main application
├── requirements.txt             # Python dependencies
├── install.bat                 # Windows installation script
├── install.sh                  # Linux/Mac installation script
└── README.md                   # This file
```

## Troubleshooting

### Common Issues

1. **Import Error**: Make sure you're using the virtual environment
2. **OpenCV Error**: Try reinstalling opencv-python: `pip install --force-reinstall opencv-python`
3. **Memory Error**: Reduce image size or close other applications
4. **Slow Performance**: Use smaller images or reduce detection parameters

### Performance Tips

- Use images with resolution under 2000x2000 pixels for best performance
- For template matching, use smaller reference objects
- For color segmentation, start with conservative tolerance values
- Disable watershed segmentation if not needed

## License

This project is for educational purposes. Please ensure you have proper licenses for any images you process.