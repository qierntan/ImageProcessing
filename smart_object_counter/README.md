# Smart Object Counter with Rotation and Scaling

A comprehensive object counting system that combines advanced computer vision techniques including YOLO-based detection, rotation detection, scaling detection, and intelligent pattern matching for accurate object counting in various scenarios.

## 🚀 Features

### 🔍 ROI Selection Methods
- **Auto Detection**: Automatic object detection using YOLO (default)
- **Rectangle Selection**: Manual rectangle selection by dragging

### 🎨 Appearance Detection Options
- **Grayscale Matching**: Detect objects using grayscale pattern matching
- **Color Matching**: Detect objects using color-based segmentation

### 🔄 Advanced Detection Capabilities
- **Rotation Detection**: Enable/disable rotation detection (0-360° in 8° increments)
- **Scaling Detection**: Enable/disable scaling detection (0.6x to 1.4x scale variations)
- **Intelligent Thresholding**: Automatic threshold selection based on image content

## 🛠️ How It Works

1. **Load Image**: 
   - Supports JPG, PNG, BMP, TIFF, WebP, AVIF formats
   - Automatically resets to optimal detection settings
   - Enables rotation and scaling detection by default

2. **Choose ROI Method**: 
   - **Auto Detection**: Automatically detects objects using YOLO
   - **Rectangle**: Manually draw selection box (clears YOLO detections)

3. **Configure Detection**: 
   - Choose between grayscale or color matching
   - Adjust detection sensitivity and parameters

4. **Set Flexibility**: 
   - Enable/disable rotation and scaling detection
   - Fine-tune detection parameters

5. **Count Objects**: 
   - Click "Count Objects" to analyze the image
   - View real-time detection results

6. **Save Results**: 
   - Save annotated result images
   - Export detection data

## 🔬 Detection Methods

### Template Matching
- **Correlation-based**: Uses advanced correlation algorithms
- **Auto-thresholding**: Automatically tunes thresholds based on image content
- **Multi-scale**: Supports rotation and scaling variations
- **Edge validation**: Includes horizontal flipping for better detection

### ORB (Oriented FAST and Rotated BRIEF)
- **Feature-based**: Uses ORB keypoints for robust detection
- **FLANN matching**: Fast approximate nearest neighbor matching
- **DBSCAN clustering**: Intelligent object grouping
- **Homography validation**: Geometric consistency checking

### YOLO Integration
- **Automatic detection**: Pre-trained YOLO model for common objects
- **Fallback system**: Seamlessly switches to classical methods if needed
- **Real-time processing**: Fast inference for live applications

## 📋 Requirements

### System Requirements
- **Python**: 3.8+ (recommended 3.9+)
- **OS**: Windows 10+, macOS 10.14+, or Linux
- **Memory**: 4GB+ RAM (8GB+ recommended)
- **GPU**: Optional but recommended for YOLO acceleration

### Python Dependencies
```
# Core image processing and computer vision
opencv-python>=4.8.0
numpy>=1.21.0
Pillow>=9.0.0

# Machine learning and clustering
scikit-learn>=1.0.0

# YOLO object detection
ultralytics>=8.0.0

# Additional useful packages for image processing
matplotlib>=3.5.0
seaborn>=0.11.0
```

## 🚀 Installation

### Quick Start
1. **Clone or download** the project files
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the application**:
   ```bash
   python smart_object_counter.py
   ```

### Detailed Installation
1. **Ensure Python 3.8+ is installed**
2. **Create virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install requirements**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Download YOLO model** (if not included):
   - The `yolov8n.pt` file should be in the same directory
   - Or it will be automatically downloaded on first run

## 📖 Usage Examples

### Basic Object Counting
1. **Load an image** from the file menu
2. **Select "Rectangle"** ROI method
3. **Draw selection box** around a reference object
4. **Enable rotation and scaling** detection
5. **Click "Count Objects"** to analyze

### Auto-Detection with YOLO (Recommended)
1. **Load an image**
2. **Select "Auto Detection"** ROI method (default)
3. **Objects are automatically detected** and highlighted
4. **Click on highlighted object** to select as reference
5. **Click "Count Objects"** for analysis

### Advanced Detection Settings
- **Grayscale Matching**: Best for objects with distinct patterns/textures
- **Color Matching**: Best for objects with distinct colors
- **Rotation Detection**: Enable for objects that may be rotated
- **Scaling Detection**: Enable for objects of varying sizes

## ⚙️ Technical Details

### Detection Parameters
- **Rotation Detection**: 8-degree increments from 0° to 360°
- **Scaling Detection**: Scale factors from 0.6x to 1.4x
- **Threshold Tuning**: Automatic threshold selection based on correlation analysis
- **Object Validation**: Edge-based validation with adaptive thresholds

### Performance Features
- **Non-Maximum Suppression**: IoU-based overlap removal
- **Intelligent Method Selection**: Automatic choice between ORB and template matching
- **Memory Optimization**: Efficient image processing and caching
- **Real-time Updates**: Live preview and parameter adjustment

### YOLO Integration
- **Model**: YOLOv8 nano (yolov8n.pt)
- **Classes**: 80 common object categories
- **Fallback**: Automatic switch to classical methods if YOLO fails
- **Performance**: Optimized for real-time processing

## 🎯 Best Practices

### For Best Results
1. **Use high-quality images** with good lighting
2. **Select representative reference objects** that are clearly visible
3. **Enable rotation detection** for objects that may be oriented differently
4. **Enable scaling detection** for objects of varying sizes
5. **Use grayscale matching** for textured objects
6. **Use color matching** for objects with distinct colors

### Troubleshooting
- **Poor detection**: Try adjusting thresholds or changing detection method
- **Slow performance**: Disable rotation/scaling detection if not needed
- **Memory issues**: Close other applications or use smaller images
- **YOLO errors**: Ensure yolov8n.pt file is present and accessible

## 📁 File Structure
```
combine2/
├── smart_object_counter.py    # Main application
├── requirements.txt           # Python dependencies
├── yolov8n.pt               # YOLO model file
├── README.md                # This file
└── sample_images/           # Example images for testing
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- **OpenCV** for computer vision capabilities
- **Ultralytics** for YOLO implementation
- **scikit-learn** for machine learning algorithms
- **PIL/Pillow** for image processing

---

**Note**: The system automatically chooses between ORB and template matching based on feature availability. For best results with rotation detection, use objects with distinct features. Scaling detection works best with objects that have consistent aspect ratios.
