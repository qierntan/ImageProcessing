# Smart Object Counter with Rotation and Scaling

This is a smart object counting system that combines rotation detection, scaling detection, and YOLO-based object detection for intelligent object counting.

## Features

### ROI Selection
- **Auto Detection**: Automatic object detection using YOLO or classical methods (default)
- **Rectangle (drag)**: Manual rectangle selection by dragging

### Appearance Options
- **Grayscale Matching**: Detect objects using grayscale pattern matching
- **Color Matching**: Detect objects using color-based matching

### Detection Flexibility
- **Detect Rotated Objects**: Enable/disable rotation detection (0-360° in 8° increments)
- **Detect Objects of Different Sizes**: Enable/disable scaling detection (0.6x to 1.4x scale variations)

## How It Works

1. **Load Image**: Load an image file (supports JPG, PNG, BMP, TIFF, WebP, AVIF)
   - Automatically resets to Auto Detection mode
   - Automatically resets to Grayscale matching
   - Automatically enables rotation and scaling detection
2. **Choose ROI Method**: 
   - Select "Auto Detection" to automatically detect objects using YOLO (default)
   - Select "Rectangle" to manually draw a selection box (clears YOLO detections)
3. **Configure Detection**: Choose between grayscale or color matching
4. **Set Flexibility**: Enable/disable rotation and scaling detection
5. **Count Objects**: Click "Count Objects" to analyze the image
6. **Save Results**: Save the annotated result image

## Detection Methods

### Template Matching
- Uses correlation-based template matching
- Automatically tunes thresholds based on image content
- Supports rotation and scaling variations
- Includes horizontal flipping for better detection

### ORB (Oriented FAST and Rotated BRIEF)
- Feature-based detection using ORB keypoints
- FLANN-based matching with automatic ratio threshold selection
- DBSCAN clustering for object grouping
- Homography-based validation

## Requirements

- Python 3.7+
- OpenCV 4.5+
- NumPy 1.19+
- Pillow 8.0+
- scikit-learn 0.24+
- ultralytics 8.0+ (for YOLO detection)

## Installation

1. Install the required packages:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
python smart_object_counter.py
```

## Usage Examples

### Basic Object Counting
1. Load an image
2. Select "Rectangle" ROI method
3. Draw a selection box around a reference object
4. Enable rotation and scaling detection
5. Click "Count Objects"

### Auto-Detection with YOLO (Recommended)
1. Load an image
2. Select "Auto Detection" ROI method (default)
3. Objects will be automatically detected and highlighted using YOLO
4. Click on a highlighted object to select it as reference
5. Click "Count Objects"



### Custom Detection Settings
- **Grayscale Matching**: Best for objects with distinct patterns
- **Color Matching**: Best for objects with distinct colors
- **Rotation Detection**: Enable for objects that may be rotated
- **Scaling Detection**: Enable for objects of varying sizes

## Technical Details

- **Rotation Detection**: 8-degree increments from 0° to 360°
- **Scaling Detection**: Scale factors from 0.6x to 1.4x
- **Threshold Tuning**: Automatic threshold selection based on correlation map analysis
- **Object Validation**: Edge-based validation with adaptive thresholds
- **Non-Maximum Suppression**: IoU-based overlap removal
- **YOLO Integration**: Automatic fallback to classical methods if YOLO fails

## Notes

- The system automatically chooses between ORB and template matching based on feature availability
- For best results with rotation detection, use objects with distinct features
- Scaling detection works best with objects that have consistent aspect ratios
- YOLO detection requires the `yolov8n.pt` model file in the same directory
