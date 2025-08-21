import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import os

class SmartObjectCounter:
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
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        
        if file_path:
            self.original_image = cv2.imread(file_path)
            if self.original_image is not None:
                self.image = self.original_image.copy()
                self.display_image = self.original_image.copy()
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
        
        resized_image = cv2.resize(self.display_image, (new_w, new_h))
        
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(rgb_image)
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
            # Convert to grayscale for processing
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Use Otsu's thresholding for better segmentation
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Morphological operations to connect object parts
            kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 15))  # Vertical kernel to connect screw parts
            kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            
            # Close operation to connect head and shaft
            closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close)
            # Open operation to remove small noise
            cleaned = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)
            
            # Find contours
            contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours by area and aspect ratio (screws are typically tall and narrow)
            min_area = 500  # Increased minimum area
            self.detected_objects = []
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > min_area:
                    # Get bounding rectangle
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = h / w if w > 0 else 0
                    
                    # Filter by aspect ratio (screws are typically taller than wide)
                    if aspect_ratio > 1.2:  # Height should be at least 1.2x width
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
        highlight_image = self.original_image.copy()
        
        # Draw bounding boxes around detected objects
        for i, (x, y, w, h) in enumerate(self.detected_objects):
            # Draw rectangle with different color for each object
            color = (255, 0, 255)  # Magenta for detected objects
            cv2.rectangle(highlight_image, (x, y), (x+w, y+h), color, 2)
            cv2.putText(highlight_image, f"Object {i+1}", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Update display
        self.display_image = highlight_image
        self.display_image_on_canvas()
    
    def highlight_selected_object(self, selected_index):
        if not self.detected_objects:
            return
        
        # Create a copy of the original image for highlighting
        highlight_image = self.original_image.copy()
        
        # Draw bounding boxes around all detected objects
        for i, (x, y, w, h) in enumerate(self.detected_objects):
            if i == selected_index:
                # Highlight selected object in green
                color = (0, 255, 0)  # Green for selected object
                thickness = 3
                cv2.putText(highlight_image, f"Reference (Object {i+1})", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            else:
                # Other objects in magenta
                color = (255, 0, 255)  # Magenta for other objects
                thickness = 2
                cv2.putText(highlight_image, f"Object {i+1}", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            cv2.rectangle(highlight_image, (x, y), (x+w, y+h), color, thickness)
        
        # Update display
        self.display_image = highlight_image
        self.display_image_on_canvas()
    
    def clear_selection(self):
        self.selected_roi = None
        self.canvas.delete("selection_rect")
        self.rect_start = None
        self.rect_end = None
        self.detected_objects = []
        self.object_highlighted = False
        # Reset to original image
        if self.original_image is not None:
            self.display_image = self.original_image.copy()
            self.display_image_on_canvas()
    
    def count_objects(self):
        if self.image is None:
            messagebox.showwarning("Warning", "Please load an image first")
            return
        
        if self.selected_roi is None:
            # No reference object selected - show clear message and return zero counts
            self.results = {
                'small': [],
                'same': [],
                'large': [],
                'reference_area': 0,
                'no_reference_selected': True
            }
            self.display_results()
            return
        
        try:
            # Extract the selected ROI
            x1, y1, x2, y2 = self.selected_roi
            reference_object = self.image[y1:y2, x1:x2]
            
            # Check if the selected area is too small (empty selection)
            if (x2 - x1) < 5 or (y2 - y1) < 5:
                # Empty or very small selection - return zero counts
                self.results = {
                    'small': [],
                    'same': [],
                    'large': [],
                    'reference_area': 0,
                    'no_objects_found': True
                }
                self.display_results()
                self.draw_results_on_image()
                return
            
            # Convert to grayscale for processing
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Use Otsu's thresholding for better segmentation
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Morphological operations to connect object parts
            kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 15))  # Vertical kernel to connect screw parts
            kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            
            # Close operation to connect head and shaft
            closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close)
            # Open operation to remove small noise
            cleaned = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)
            
            # Find contours
            contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours by area and aspect ratio (screws are typically tall and narrow)
            min_area = 500  # Increased minimum area
            valid_contours = []
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > min_area:
                    # Get bounding rectangle
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = h / w if w > 0 else 0
                    
                    # Filter by aspect ratio (screws are typically taller than wide)
                    if aspect_ratio > 1.2:  # Height should be at least 1.2x width
                        valid_contours.append(contour)
            
            # Check if any valid objects were found
            if len(valid_contours) == 0:
                # No objects found in the image
                self.results = {
                    'small': [],
                    'same': [],
                    'large': [],
                    'reference_area': 0,
                    'no_objects_found': True
                }
                self.display_results()
                self.draw_results_on_image()
                return
            
            # Calculate reference object area using bounding rectangle
            ref_contours = cv2.findContours(
                cv2.cvtColor(reference_object, cv2.COLOR_BGR2GRAY),
                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )[0]
            
            if len(ref_contours) > 0:
                ref_x, ref_y, ref_w, ref_h = cv2.boundingRect(ref_contours[0])
                reference_area = ref_w * ref_h
            else:
                # No contours found in reference area - use selection rectangle
                reference_area = (x2 - x1) * (y2 - y1)
            
            # Check if reference area is valid
            if reference_area <= 0:
                # Invalid reference area - return zero counts
                self.results = {
                    'small': [],
                    'same': [],
                    'large': [],
                    'reference_area': 0,
                    'no_objects_found': True
                }
                self.display_results()
                self.draw_results_on_image()
                return
            
            # Classify objects by size using bounding rectangle area
            small_objects = []
            same_size_objects = []
            large_objects = []
            
            # More sensitive thresholds
            small_threshold = 0.6  # Objects smaller than 60% of reference
            large_threshold = 1.05  # Objects larger than 105% of reference
            
            for contour in valid_contours:
                x, y, w, h = cv2.boundingRect(contour)
                area = w * h  # Use bounding rectangle area instead of contour area
                ratio = area / reference_area
                
                if ratio < small_threshold:
                    small_objects.append((x, y, w, h, area))
                elif ratio > large_threshold:
                    large_objects.append((x, y, w, h, area))
                else:
                    same_size_objects.append((x, y, w, h, area))
            
            # Store results
            self.results = {
                'small': small_objects,
                'same': same_size_objects,
                'large': large_objects,
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
            results_text += f"Small Objects : 0\n"
            results_text += f"Same Size Objects : 0\n"
            results_text += f"Large Objects : 0\n\n"
            results_text += f"Total Objects : 0\n\n"
            results_text += "Note: Please click on a highlighted object to select it as reference."
        # Check if no objects were found
        elif self.results.get('no_objects_found', False):
            results_text = "No objects found in the image.\n\n"
            results_text += f"Small Objects : 0\n"
            results_text += f"Same Size Objects : 0\n"
            results_text += f"Large Objects : 0\n\n"
            results_text += f"Total Objects : 0\n\n"
            results_text += "Note: No valid objects detected.\n"
            results_text += "Please ensure you have selected an area with objects."
        # Check if reference area is 0 (empty selection)
        elif self.results['reference_area'] == 0:
            results_text = "Reference Object Area: 0 pixels (Empty selection)\n\n"
            results_text += f"Small Objects : 0\n"
            results_text += f"Same Size Objects : 0\n"
            results_text += f"Large Objects : 0\n\n"
            results_text += f"Total Objects : 0\n\n"
            results_text += "Note: Please select a valid reference object."
        else:
            results_text = f"Reference Object Area: {self.results['reference_area']:.1f} pixels\n\n"
            results_text += f"Small Objects : {len(self.results['small'])}\n"
            results_text += f"Same Size Objects : {len(self.results['same'])}\n"
            results_text += f"Large Objects : {len(self.results['large'])}\n\n"
            results_text += f"Total Objects : {len(self.results['small']) + len(self.results['same']) + len(self.results['large'])}\n\n"
            
            # Add detailed information
            results_text += "Detection Settings:\n"
            results_text += "- Min area : 500 pixels\n"
            results_text += "- Aspect ratio > 1.2\n"
            results_text += "- Uses bounding rectangle area\n"
        
        self.results_text.insert(tk.END, results_text)
    
    def draw_results_on_image(self):
        if not self.results:
            return
        
        # Create a copy of the original image for drawing
        result_image = self.original_image.copy()
        
        # Only draw reference object if objects were found
        if self.selected_roi and not self.results.get('no_objects_found', False):
            x1, y1, x2, y2 = self.selected_roi
            cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(result_image, "Reference", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Draw small objects
        for x, y, w, h, area in self.results['small']:
            cv2.rectangle(result_image, (x, y), (x+w, y+h), (0, 0, 255), 1)
            cv2.putText(result_image, "S", (x, y-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
        
        # Draw same size objects
        for x, y, w, h, area in self.results['same']:
            cv2.rectangle(result_image, (x, y), (x+w, y+h), (255, 0, 0), 1)
            cv2.putText(result_image, "M", (x, y-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
        
        # Draw large objects
        for x, y, w, h, area in self.results['large']:
            cv2.rectangle(result_image, (x, y), (x+w, y+h), (255, 255, 0), 1)
            cv2.putText(result_image, "L", (x, y-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)
        
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
        elif self.original_image is not None:
            # If no objects were detected, just show the original image
            self.display_image = self.original_image.copy()
            self.display_image_on_canvas()

def main():
    root = tk.Tk()
    app = SmartObjectCounter(root)
    
    # Handle window resize
    def on_resize(event):
        if app.image is not None:
            app.display_image_on_canvas()
    
    root.bind("<Configure>", on_resize)
    
    root.mainloop()

if __name__ == "__main__":
    main()
