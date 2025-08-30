# Smart Object Counter

A Python application that allows users to count objects in images by selecting a reference object and automatically detecting similar objects, categorizing them by size (small, same, large).

## Features

- **Interactive GUI**: User-friendly interface with image display and controls
- **Rectangle Selection**: Click and drag to select a reference object
- **Automatic Object Detection**: Uses computer vision techniques to find similar objects
- **Size Classification**: Categorizes objects as small, same size, or large compared to the reference
- **Visual Results**: Displays bounding boxes and labels on detected objects
- **Real-time Counting**: Shows counts for each category

## Installation

1. Install Python 3.7 or higher
2. Install required dependencies:
   ```bash
   python -m pip install -r requirements.txt
   ```
   pip install ultralytics

## Usage

1. **Run the application**:
   ```bash
   python applyscaling.py
   ```

2. **Load an image**:
   - Click "Load Image" button
   - Select an image file (supports JPG, PNG, BMP, TIFF)

3. **Select reference object**:
   - Click and drag to draw a rectangle around the main object you want to count
   - The selected area will be highlighted in red

4. **Count objects**:
   - Click "Count Objects" button
   - The application will analyze the image and detect similar objects

5. **View results**:
   - Results are displayed in the left panel showing counts for each category
   - Detected objects are highlighted on the image with different colors:
     - **Green**: Reference object
     - **Red**: Small objects (S)
     - **Blue**: Same size objects (M)
     - **Yellow**: Large objects (L)

## Controls

- **Load Image**: Open an image file for analysis
- **Clear Selection**: Remove the current rectangle selection
- **Count Objects**: Analyze the image and count objects
- **Reset**: Clear all results and return to original image

## How It Works

1. **Image Preprocessing**: Converts image to grayscale and applies Gaussian blur
2. **Object Segmentation**: Uses adaptive thresholding to separate objects from background
3. **Morphological Operations**: Cleans up the segmented image using opening and closing operations
4. **Contour Detection**: Finds all object boundaries in the image
5. **Size Classification**: Compares each detected object's area to the reference object:
   - Small: < 50% of reference area
   - Same: 50% - 150% of reference area
   - Large: > 150% of reference area

## Tips for Best Results

- Use images with good contrast between objects and background
- Ensure the reference object is clearly visible and well-defined
- Avoid selecting very small areas as reference objects
- For better accuracy, use images with consistent lighting
- Objects should have distinct boundaries for better detection

## Troubleshooting

- **No objects detected**: Try adjusting the image contrast or selecting a larger reference area
- **Too many false positives**: The image may have too much noise; try using a cleaner image
- **Application crashes**: Ensure all dependencies are properly installed

## Dependencies

- OpenCV (cv2): Computer vision operations
- NumPy: Numerical operations
- PIL (Pillow): Image processing
- Tkinter: GUI framework (included with Python)

## License

This project is open source and available under the MIT License. 



use this to run
"C:\Users\User\AppData\Local\Programs\Python\Python312\python.exe" applyscaling.py


Steps to run (in cmd): 

1. python -m pip install -r requirements.txt
2. pip install ultralytics
3. python applyscaling.py