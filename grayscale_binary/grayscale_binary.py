import cv2 as cv
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading, os

class ObjectCounterGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Smart Object Counter")
        self.root.geometry("1200x800")
        
        # Variables
        self.image = None
        self.original_image = None
        self.display_image = None
        self.photo = None
        self.rect_start = None
        self.rect_end = None
        self.drawing = False
        self.selected_roi = None
        self.results = {}
        self.detected_objects = []  # Store detected object bounding boxes
        self.object_highlighted = False  # Track if objects are highlighted
        self.gray_image = None
        self.binary_image = None
        self.template = None
        self.detections = []
        self.image_path = None
        self.roi_selecting = False
        self.start_x = None
        self.start_y = None
        self.rect_id = None
        self.canvas_image_id = None
        self.scale = 1.0
        self.image_x = 0
        self.image_y = 0

        # GUI Setup
        self.setup_gui()
        
    def setup_gui(self):
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel for controls
        left_panel = ttk.Frame(main_frame, width=300)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # Right panel for image display
        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Control buttons
        ttk.Button(left_panel, text="Load Image", command=self.load_image).pack(fill=tk.X, pady=5)
        ttk.Button(left_panel, text="Count Objects", command=self.count_objects).pack(fill=tk.X, pady=5)
        ttk.Button(left_panel, text="Reset", command=self.reset).pack(fill=tk.X, pady=5)

        # Instructions
        instruction_frame = ttk.LabelFrame(left_panel, text="Instructions")
        instruction_frame.pack(fill=tk.X, pady=10)
        
        instructions = """
1. Load an image (objects will be detected automatically)
2. Click on a highlighted object to select it as reference
3. Click "Count Objects" to analyze
4. View results below
        """
        ttk.Label(instruction_frame, text=instructions, justify=tk.LEFT).pack(padx=10, pady=10)
        
        # Results display
        results_frame = ttk.LabelFrame(left_panel, text="Results")
        results_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.results_text = tk.Text(results_frame, height=10, width=35)
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Image display
        self.canvas = tk.Canvas(right_panel, bg="white")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Bind mouse events
        self.canvas.bind("<Button-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        
    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp")]
        )
        
        if file_path:
            self.original_image = cv.imread(file_path)
            if self.original_image is not None:
                self.image = self.original_image.copy()
                self.display_image = self.original_image.copy()
                # Convert to RGB for display
                self.image = cv.cvtColor(self.image, cv.COLOR_BGR2RGB)
                self.display_image = cv.cvtColor(self.display_image, cv.COLOR_BGR2RGB)
                # Convert to grayscale and binary for processing
                self.gray_image = cv.cvtColor(self.image, cv.COLOR_RGB2GRAY)
                _, self.binary_image = cv.threshold(self.gray_image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
                self.image_path = file_path
                self.display_image_on_canvas()
                self.clear_results()
                # Automatically detect objects after loading
                self.auto_detect_objects()
            else:
                messagebox.showerror("Error", "Failed to load image")
    
    def display_image_on_canvas(self):
        if self.display_image is None:
            return
            
        # Get canvas dimensions
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            # Canvas not yet sized, schedule for later
            self.root.after(100, self.display_image_on_canvas)
            return
        
        # Resize image to fit canvas while maintaining aspect ratio
        h, w = self.display_image.shape[:2]
        scale = min(canvas_width / w, canvas_height / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        resized_image = cv.resize(self.display_image, (new_w, new_h))
        
        # Convert to PIL Image
        pil_image = Image.fromarray(resized_image)
        self.photo = ImageTk.PhotoImage(pil_image)
        
        # Clear canvas and display image
        self.canvas.delete("all")
        self.canvas.create_image(
            canvas_width // 2, canvas_height // 2,
            image=self.photo, anchor=tk.CENTER
        )
        
        # Store scale for coordinate conversion
        self.scale_factor = scale
        self.canvas_offset_x = (canvas_width - new_w) // 2
        self.canvas_offset_y = (canvas_height - new_h) // 2
    
    def on_mouse_down(self, event):
        if self.image is None:
            return
        
        # If objects are highlighted, try to select one
        if self.object_highlighted and self.detected_objects:
            # Convert canvas coordinates to image coordinates
            canvas_x = (event.x - self.canvas_offset_x) / self.scale_factor
            canvas_y = (event.y - self.canvas_offset_y) / self.scale_factor
            
            # Check if click is within any detected object
            for i, (x, y, w, h) in enumerate(self.detected_objects):
                if x <= canvas_x <= x + w and y <= canvas_y <= y + h:
                    # Object selected
                    self.selected_roi = (x, y, x + w, y + h)
                    self.highlight_selected_object(i)
                    return
        
        # Fall back to rectangle selection if no objects highlighted
        self.rect_start = (event.x, event.y)
        self.drawing = True
        self.canvas.delete("selection_rect")
    
    def on_mouse_drag(self, event):
        if not self.drawing or self.rect_start is None:
            return
        
        self.rect_end = (event.x, event.y)
        self.canvas.delete("selection_rect")
        
        # Draw rectangle
        x1, y1 = self.rect_start
        x2, y2 = self.rect_end
        self.canvas.create_rectangle(
            x1, y1, x2, y2,
            outline="red", width=2,
            tags="selection_rect"
        )
    
    def on_mouse_up(self, event):
        if not self.drawing:
            return
        
        self.drawing = False
        self.rect_end = (event.x, event.y)
        
        if self.rect_start and self.rect_end:
            # Convert canvas coordinates to image coordinates
            x1 = (min(self.rect_start[0], self.rect_end[0]) - self.canvas_offset_x) / self.scale_factor
            y1 = (min(self.rect_start[1], self.rect_end[1]) - self.canvas_offset_y) / self.scale_factor
            x2 = (max(self.rect_start[0], self.rect_end[0]) - self.canvas_offset_x) / self.scale_factor
            y2 = (max(self.rect_start[1], self.rect_end[1]) - self.canvas_offset_y) / self.scale_factor
            
            # Ensure coordinates are within image bounds
            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(self.image.shape[1], int(x2))
            y2 = min(self.image.shape[0], int(y2))
            
            if x2 > x1 and y2 > y1:
                self.selected_roi = (x1, y1, x2, y2)
                print(f"Selected ROI: {self.selected_roi}")
    
    def auto_detect_objects(self):
        """Automatically detect objects after loading an image"""
        try:
            if self.gray_image is None:
                return
                
            # Enhanced preprocessing for better object detection
            # Apply Gaussian blur to reduce noise
            blurred = cv.GaussianBlur(self.gray_image, (5, 5), 0)
            
            # Use adaptive thresholding for better segmentation
            binary = cv.adaptiveThreshold(blurred, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv.THRESH_BINARY_INV, 11, 2)
            
            # Morphological operations to clean up the binary image
            kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
            cleaned = cv.morphologyEx(binary, cv.MORPH_CLOSE, kernel)
            cleaned = cv.morphologyEx(cleaned, cv.MORPH_OPEN, kernel)
            
            # Find contours
            contours, _ = cv.findContours(cleaned, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            
            # Enhanced filtering criteria with stricter parameters
            min_area = 500  # Increased minimum area to reduce false positives
            max_area = 30000  # Reduced maximum area to avoid detecting large regions
            self.detected_objects = []
            
            for contour in contours:
                area = cv.contourArea(contour)
                if min_area < area < max_area:
                    # Get bounding rectangle
                    x, y, w, h = cv.boundingRect(contour)
                    
                    # Stricter aspect ratio filtering
                    aspect_ratio = h / w if w > 0 else 0
                    if 0.3 < aspect_ratio < 3.0:  # Tighter aspect ratio range
                        # Calculate contour properties for better filtering
                        perimeter = cv.arcLength(contour, True)
                        if perimeter > 0:
                            circularity = 4 * np.pi * area / (perimeter * perimeter)
                            # Stricter circularity filter
                            if 0.2 < circularity < 0.9:
                                # Additional filter: check if the contour is reasonably compact
                                # Calculate the ratio of contour area to bounding rectangle area
                                rect_area = w * h
                                if rect_area > 0:
                                    compactness = area / rect_area
                                    # Objects should be reasonably compact (not too sparse)
                                    if compactness > 0.3:
                                        self.detected_objects.append((x, y, w, h))
            
            if len(self.detected_objects) == 0:
                messagebox.showinfo("Info", "No objects detected in the image.")
                return
            
            # Highlight detected objects
            self.highlight_detected_objects()
            self.object_highlighted = True
            
            messagebox.showinfo("Info", f"Found {len(self.detected_objects)} objects. Click on one to select it as reference.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error detecting objects: {str(e)}")
    
    def highlight_detected_objects(self):
        if not self.detected_objects:
            return
        
        # Create a copy of the original image for highlighting
        highlight_image = self.image.copy()
        
        # Draw bounding boxes around detected objects
        for i, (x, y, w, h) in enumerate(self.detected_objects):
            # Draw rectangle with different color for each object
            color = (255, 0, 255)  # Magenta for detected objects
            cv.rectangle(highlight_image, (x, y), (x+w, y+h), color, 2)
            cv.putText(highlight_image, f"Object {i+1}", (x, y-10), 
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Update display
        self.display_image = highlight_image
        self.display_image_on_canvas()
    
    def highlight_selected_object(self, selected_index):
        if not self.detected_objects:
            return
        
        # Create a copy of the original image for highlighting
        highlight_image = self.image.copy()
        
        # Draw bounding boxes around all detected objects
        for i, (x, y, w, h) in enumerate(self.detected_objects):
            if i == selected_index:
                # Highlight selected object in green
                color = (0, 255, 0)  # Green for selected object
                thickness = 3
                cv.putText(highlight_image, f"Reference (Object {i+1})", (x, y-10), 
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            else:
                # Other objects in magenta
                color = (255, 0, 255)  # Magenta for other objects
                thickness = 2
                cv.putText(highlight_image, f"Object {i+1}", (x, y-10), 
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            cv.rectangle(highlight_image, (x, y), (x+w, y+h), color, thickness)
        
        # Update display
        self.display_image = highlight_image
        self.display_image_on_canvas()
    
    def non_maximum_suppression(self, detections, overlap_thresh=0.3):
        """Remove overlapping detections using non-maximum suppression"""
        if len(detections) == 0:
            return []
        
        # Convert to numpy array for easier processing
        boxes = np.array([[d['x'], d['y'], d['x'] + d['w'], d['y'] + d['h']] for d in detections])
        confidences = np.array([d['confidence'] for d in detections])
        
        # Sort by confidence
        indices = np.argsort(confidences)[::-1]
        
        keep = []
        while len(indices) > 0:
            # Pick the detection with highest confidence
            current = indices[0]
            keep.append(current)
            
            if len(indices) == 1:
                break
            
            # Remove current from indices
            indices = indices[1:]
            
            # Calculate IoU with remaining detections
            current_box = boxes[current]
            remaining_boxes = boxes[indices]
            
            # Calculate intersection
            x1 = np.maximum(current_box[0], remaining_boxes[:, 0])
            y1 = np.maximum(current_box[1], remaining_boxes[:, 1])
            x2 = np.minimum(current_box[2], remaining_boxes[:, 2])
            y2 = np.minimum(current_box[3], remaining_boxes[:, 3])
            
            intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
            
            # Calculate union
            current_area = (current_box[2] - current_box[0]) * (current_box[3] - current_box[1])
            remaining_areas = (remaining_boxes[:, 2] - remaining_boxes[:, 0]) * (remaining_boxes[:, 3] - remaining_boxes[:, 1])
            union = current_area + remaining_areas - intersection
            
            # Calculate IoU
            iou = intersection / union
            
            # Keep detections with IoU below threshold
            indices = indices[iou < overlap_thresh]
        
        # Return filtered detections
        return [detections[i] for i in keep]
    
    def clear_selection(self):
        self.selected_roi = None
        self.canvas.delete("selection_rect")
        self.rect_start = None
        self.rect_end = None
        self.detected_objects = []
        self.object_highlighted = False
        # Reset to original image
        if self.image is not None:
            self.display_image = self.image.copy()
            self.display_image_on_canvas()

    def count_objects(self):
        if self.image is None:
            messagebox.showwarning("Warning", "Please load an image first")
            return
        
        if self.selected_roi is None:
            # No reference object selected - show clear message and return zero counts
            self.results = {
                'objects': [],
                'reference_area': 0,
                'no_reference_selected': True
            }
            self.display_results()
            return
        
        try:
            # Extract the selected ROI
            x1, y1, x2, y2 = self.selected_roi
            reference_object = self.gray_image[y1:y2, x1:x2]
            
            # Check if the selected area is too small (empty selection)
            if (x2 - x1) < 5 or (y2 - y1) < 5:
                # Empty or very small selection - return zero counts
                self.results = {
                    'objects': [],
                    'reference_area': 0,
                    'no_objects_found': True
                }
                self.display_results()
                self.draw_results_on_image()
                return
            
            # Enhanced object detection using multiple approaches
            
            # Convert reference to edges for color-invariant matching
            # Use adaptive Canny parameters based on reference object brightness
            ref_mean_brightness = np.mean(reference_object)
            if ref_mean_brightness > 200:  # Very bright/white objects
                ref_edges = cv.Canny(reference_object, 30, 100)  # Lower thresholds for white objects
            elif ref_mean_brightness > 150:  # Light colored objects
                ref_edges = cv.Canny(reference_object, 40, 120)  # Medium thresholds for light objects
            else:  # Dark colored objects
                ref_edges = cv.Canny(reference_object, 50, 150)  # Standard thresholds for dark objects
            
            # 1. Template matching with multiple scales (more conservative)
            template = reference_object
            scales = [0.8, 0.9, 1.0, 1.1, 1.2]  # Reduced scale range for more precise detection
            all_detections = []
            
            for scale in scales:
                # Resize template
                new_w = int(reference_object.shape[1] * scale)
                new_h = int(reference_object.shape[0] * scale)
                if new_w > 0 and new_h > 0:
                    resized_template = cv.resize(reference_object, (new_w, new_h))
                    
                    # Use adaptive Canny parameters for resized template
                    resized_mean_brightness = np.mean(resized_template)
                    if resized_mean_brightness > 200:  # Very bright/white objects
                        resized_edges = cv.Canny(resized_template, 30, 100)
                    elif resized_mean_brightness > 150:  # Light colored objects
                        resized_edges = cv.Canny(resized_template, 40, 120)
                    else:  # Dark colored objects
                        resized_edges = cv.Canny(resized_template, 50, 150)
                    
                    # Template matching on edges (color-invariant)
                    res = cv.matchTemplate(self.gray_image, resized_template, cv.TM_CCOEFF_NORMED)
                    # Adjust threshold based on reference brightness
                    if ref_mean_brightness > 200:  # White objects need lower threshold
                        threshold = 0.5
                    elif ref_mean_brightness > 150:  # Light objects
                        threshold = 0.55
                    else:  # Dark objects
                        threshold = 0.6
                    loc = np.where(res >= threshold)
                    
                    for pt in zip(*loc[::-1]):
                        all_detections.append({
                            'x': pt[0], 'y': pt[1],
                            'w': new_w, 'h': new_h,
                            'confidence': float(res[pt[1], pt[0]]),
                            'scale': scale
                        })
            
            # 2. Enhanced contour-based detection with color-invariant similarity
            # Use multiple preprocessing approaches for better detection
            all_contours = []
            
            # Method 1: Adaptive thresholding
            blurred = cv.GaussianBlur(self.gray_image, (5, 5), 0)
            binary1 = cv.adaptiveThreshold(blurred, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv.THRESH_BINARY_INV, 11, 2)
            
            # Method 2: Otsu thresholding
            _, binary2 = cv.threshold(blurred, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
            
            # Method 3: Edge-based detection with adaptive parameters
            if ref_mean_brightness > 200:  # Very bright/white objects
                edges = cv.Canny(blurred, 30, 100)
            elif ref_mean_brightness > 150:  # Light colored objects
                edges = cv.Canny(blurred, 40, 120)
            else:  # Dark colored objects
                edges = cv.Canny(blurred, 50, 150)
            
            # Combine all methods
            combined = cv.bitwise_or(binary1, binary2)
            combined = cv.bitwise_or(combined, edges)
            
            # Morphological operations
            kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
            cleaned = cv.morphologyEx(combined, cv.MORPH_CLOSE, kernel)
            cleaned = cv.morphologyEx(cleaned, cv.MORPH_OPEN, kernel)
            
            contours, _ = cv.findContours(cleaned, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            
            # Filter contours similar to reference object
            ref_area = (x2 - x1) * (y2 - y1)
            ref_aspect = (y2 - y1) / (x2 - x1) if (x2 - x1) > 0 else 1
            
            for contour in contours:
                area = cv.contourArea(contour)
                # More lenient area range for color-invariant detection
                if 0.2 * ref_area < area < 5.0 * ref_area:
                    x, y, w, h = cv.boundingRect(contour)
                    aspect = h / w if w > 0 else 1
                    
                    # More lenient aspect ratio check
                    if 0.5 * ref_aspect < aspect < 2.0 * ref_aspect:
                        # Calculate shape similarity using contour matching
                        roi = self.gray_image[y:y+h, x:x+w]
                        if roi.shape[0] > 0 and roi.shape[1] > 0:
                            # Resize ROI to match reference size for comparison
                            resized_roi = cv.resize(roi, (x2-x1, y2-y1))
                            
                            # Use multiple similarity measures
                            # 1. Template matching on grayscale
                            similarity1 = cv.matchTemplate(resized_roi, reference_object, cv.TM_CCOEFF_NORMED)[0][0]
                            
                            # 2. Edge-based similarity with adaptive parameters
                            roi_mean_brightness = np.mean(resized_roi)
                            if roi_mean_brightness > 200:  # Very bright/white objects
                                roi_edges = cv.Canny(resized_roi, 30, 100)
                            elif roi_mean_brightness > 150:  # Light colored objects
                                roi_edges = cv.Canny(resized_roi, 40, 120)
                            else:  # Dark colored objects
                                roi_edges = cv.Canny(resized_roi, 50, 150)
                            edge_similarity = cv.matchTemplate(roi_edges, ref_edges, cv.TM_CCOEFF_NORMED)[0][0]
                            
                            # 3. Contour shape similarity with adaptive parameters
                            if roi_mean_brightness > 200:  # Very bright/white objects
                                roi_contours, _ = cv.findContours(cv.Canny(resized_roi, 30, 100), 
                                                                cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
                            elif roi_mean_brightness > 150:  # Light colored objects
                                roi_contours, _ = cv.findContours(cv.Canny(resized_roi, 40, 120), 
                                                                cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
                            else:  # Dark colored objects
                                roi_contours, _ = cv.findContours(cv.Canny(resized_roi, 50, 150), 
                                                                cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
                            if len(roi_contours) > 0:
                                shape_similarity = cv.matchShapes(contour, roi_contours[0], 
                                                                cv.CONTOURS_MATCH_I2, 0.0)
                                # Convert shape similarity to 0-1 range (lower is better)
                                shape_similarity = 1.0 / (1.0 + shape_similarity)
                            else:
                                shape_similarity = 0.0
                            
                            # Combined similarity score
                            combined_similarity = (similarity1 + edge_similarity + shape_similarity) / 3.0
                            
                            # Adjust similarity threshold based on reference brightness
                            if ref_mean_brightness > 200:  # White objects need lower threshold
                                similarity_threshold = 0.35
                            elif ref_mean_brightness > 150:  # Light objects
                                similarity_threshold = 0.4
                            else:  # Dark objects
                                similarity_threshold = 0.4
                            
                            if combined_similarity > similarity_threshold:  # Adaptive threshold for color-invariant detection
                                all_detections.append({
                                    'x': x, 'y': y, 'w': w, 'h': h,
                                    'confidence': combined_similarity,
                                    'scale': 1.0
                                })
            
            # Remove duplicate detections using non-maximum suppression (more aggressive)
            detections = self.non_maximum_suppression(all_detections, overlap_thresh=0.2)
            
            # Additional filter: remove detections that are too close to each other
            filtered_detections = []
            min_distance = min(x2-x1, y2-y1) * 0.8  # Minimum distance between object centers
            
            for i, detection in enumerate(detections):
                center_x = detection['x'] + detection['w'] // 2
                center_y = detection['y'] + detection['h'] // 2
                too_close = False
                
                for j, other_detection in enumerate(detections):
                    if i != j:
                        other_center_x = other_detection['x'] + other_detection['w'] // 2
                        other_center_y = other_detection['y'] + other_detection['h'] // 2
                        
                        distance = np.sqrt((center_x - other_center_x)**2 + (center_y - other_center_y)**2)
                        if distance < min_distance:
                            too_close = True
                            break
                
                if not too_close:
                    filtered_detections.append(detection)
            
            detections = filtered_detections
            
            # Check if any objects were found
            if len(detections) == 0:
                # No objects found in the image
                self.results = {
                    'objects': [],
                    'reference_area': 0,
                    'no_objects_found': True
                }
                self.display_results()
                self.draw_results_on_image()
                return
            
            # Calculate reference object area
            reference_area = (x2 - x1) * (y2 - y1)
            
            # Store all detected objects in one category
            objects = []
            
            for detection in detections:
                x, y, w, h = detection['x'], detection['y'], detection['w'], detection['h']
                area = w * h
                confidence = detection['confidence']
                scale = detection['scale']
                
                # Store all objects without size classification
                objects.append((x, y, w, h, area))
            
            # Store results
            self.results = {
                'objects': objects,
                'reference_area': reference_area,
                'no_objects_found': False
            }
            
            # Display results
            self.display_results()
            
            # Draw results on image
            self.draw_results_on_image()
            
        except Exception as e:
            messagebox.showerror("Error", f"Error counting objects: {str(e)}")
    
    def display_results(self):
        self.results_text.delete(1.0, tk.END)
        
        if not self.results:
            self.results_text.insert(tk.END, "No results available.\nPlease count objects first.")
            return
        
        # Check if no reference object was selected
        if self.results.get('no_reference_selected', False):
            results_text = "No reference object selected.\n\n"
            results_text += f"Objects Detected : 0\n\n"
            results_text += "Note: Please click on a highlighted object to select it as reference."
        # Check if no objects were found
        elif self.results.get('no_objects_found', False):
            results_text = "No objects found in the image.\n\n"
            results_text += f"Objects Detected : 0\n\n"
            results_text += "Note: No valid objects detected.\n"
            results_text += "Please ensure you have selected an area with objects."
        # Check if reference area is 0 (empty selection)
        elif self.results['reference_area'] == 0:
            results_text = "Reference Object Area: 0 pixels (Empty selection)\n\n"
            results_text += f"Objects Detected : 0\n\n"
            results_text += "Note: Please select a valid reference object."
        else:
            results_text = f"Reference Object Area: {self.results['reference_area']:.1f} pixels\n\n"
            results_text += f"Objects Detected : {len(self.results['objects'])}\n\n"
            
            # Add detailed information
            results_text += "Detection Settings:\n"
            results_text += "- Adaptive color-invariant edge-based matching\n"
            results_text += "- Brightness-adaptive Canny parameters\n"
            results_text += "- Multi-method contour detection (adaptive + Otsu + edges)\n"
            results_text += "- Combined similarity scoring (template + edge + shape)\n"
            results_text += "- Adaptive similarity thresholds based on brightness\n"
            results_text += "- Aggressive NMS (IoU < 0.2) + distance filtering\n"
            results_text += "- Single category detection (no size classification)\n"
        
        self.results_text.insert(tk.END, results_text)
    
    def draw_results_on_image(self):
        if not self.results:
            return
        
        # Create a copy of the original image for drawing
        result_image = self.image.copy()
        
        # Only draw reference object if objects were found
        if self.selected_roi and not self.results.get('no_objects_found', False):
            x1, y1, x2, y2 = self.selected_roi
            cv.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv.putText(result_image, "Reference", (x1, y1-10), 
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Draw all detected objects
        for x, y, w, h, area in self.results['objects']:
            cv.rectangle(result_image, (x, y), (x+w, y+h), (255, 0, 0), 1)
            cv.putText(result_image, "O", (x, y-5), 
                       cv.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
        
        # Update display
        self.display_image = result_image
        self.display_image_on_canvas()
    
    def clear_results(self):
        self.results = {}
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "No results available.\nPlease count objects first.")
    
    def reset(self):
        # Clear the counting results and selected reference
        self.clear_results()
        self.selected_roi = None
        
        # If objects were detected and highlighted, show them again (without selection)
        if self.detected_objects and self.object_highlighted:
            # Just highlight all detected objects (no reference selected)
            self.highlight_detected_objects()
        elif self.image is not None:
            # If no objects were detected, just show the original image
            self.display_image = self.image.copy()
        self.display_image_on_canvas()

def main():
    root = tk.Tk()
    app = ObjectCounterGUI(root)
    
    # Handle window resize
    def on_resize(event):
        if app.image is not None:
            app.display_image_on_canvas()
    
    root.bind("<Configure>", on_resize)
    
    root.mainloop()

if __name__ == "__main__":
    main()
