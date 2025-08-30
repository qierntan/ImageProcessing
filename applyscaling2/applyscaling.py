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
        # YOLO model (lazy loaded)
        self.yolo_model = None
        self.yolo_names = None
        self.use_yolo = True
        
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
    
    def _ensure_yolo(self):
        """Lazy-load YOLO model if available. Return True if loaded."""
        if not self.use_yolo:
            print("YOLO disabled by user")
            return False
        if self.yolo_model is not None:
            print("YOLO already loaded")
            return True
        try:
            from ultralytics import YOLO
            print("Loading YOLO model...")
            # Small model for speed; user can swap to yolov8s.pt or better
            self.yolo_model = YOLO("yolov8n.pt")
            # names mapping lives on model
            self.yolo_names = self.yolo_model.model.names if hasattr(self.yolo_model, "model") else None
            print(f"YOLO loaded successfully with {len(self.yolo_names) if self.yolo_names else 0} classes")
            return True
        except Exception as e:
            print(f"YOLO failed to load: {e}")
            # If import or load fails, disable YOLO for this session
            self.use_yolo = False
            self.yolo_model = None
            self.yolo_names = None
            return False

    def _norm_box(self, obj):
        """Return (x,y,w,h,label,conf) regardless of storage format."""
        if isinstance(obj, dict):
            return obj.get('x', 0), obj.get('y', 0), obj.get('w', 0), obj.get('h', 0), obj.get('label', ''), obj.get('conf', 0.0)
        if isinstance(obj, (tuple, list)):
            if len(obj) >= 4:
                return obj[0], obj[1], obj[2], obj[3], obj[4] if len(obj) > 4 else '', obj[5] if len(obj) > 5 else 0.0
        return 0, 0, 0, 0, '', 0.0
        
    def _extract_bboxes_from_mask(self, binary_mask):
        """Return bounding boxes from a binary foreground mask after splitting and filtering.
        Splits touching objects with watershed and removes text-like regions.
        """
        try:
            # Ensure mask is 8-bit single channel with foreground=255
            mask = (binary_mask > 0).astype(np.uint8) * 255

            # Gently erode to break thin bridges before distance transform
            pre_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask_eroded = cv2.erode(mask, pre_kernel, iterations=1)

            # Split touching objects using watershed on distance transform
            dist = cv2.distanceTransform(mask_eroded, cv2.DIST_L2, 5)
            # Higher threshold for seeds so clusters split better
            _, sure_fg = cv2.threshold(dist, 0.60 * dist.max(), 255, 0)
            sure_fg = sure_fg.astype(np.uint8)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            sure_bg = cv2.dilate(mask_eroded, kernel, iterations=2)
            unknown = cv2.subtract(sure_bg, sure_fg)
            # Markers
            num_labels, markers = cv2.connectedComponents(sure_fg)
            markers = markers + 1
            markers[unknown == 255] = 0
            # Watershed requires 3-channel image; create dummy
            color = cv2.cvtColor(mask_eroded, cv2.COLOR_GRAY2BGR)
            cv2.watershed(color, markers)

            # Each region id > 1 is an object
            h_img, w_img = mask.shape[:2]
            bboxes = []
            # Dynamic minimums scale with image size
            min_area = max(150, int(0.00005 * h_img * w_img))
            for label in range(2, num_labels + 2):
                component = (markers == label).astype(np.uint8) * 255
                if cv2.countNonZero(component) == 0:
                    continue
                cnts, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not cnts:
                    continue
                contour = max(cnts, key=cv2.contourArea)
                area = cv2.contourArea(contour)
                if area < min_area:
                    continue
                x, y, w, h = cv2.boundingRect(contour)

                # Heuristics to remove text-like regions
                aspect_ratio = w / h if h > 0 else 0
                bbox_area = w * h
                extent = area / bbox_area if bbox_area > 0 else 0
                # Rules: very wide-thin regions, very small height, or low fill ratio are likely text
                if aspect_ratio > 6.0 and h < int(0.18 * h_img):
                    continue
                if h < max(10, int(0.02 * h_img)) or w < max(10, int(0.02 * w_img)):
                    continue
                if extent < 0.22:
                    continue

                bboxes.append((x, y, w, h))

            # Merge adjacent/overlapping boxes to prevent over-segmentation
            return self._merge_adjacent_boxes(bboxes, (h_img, w_img))
        except Exception:
            # Fallback to simple contour extraction if watershed fails
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            h_img, w_img = mask.shape[:2]
            bboxes = []
            min_area = max(150, int(0.00005 * h_img * w_img))
            for c in contours:
                area = cv2.contourArea(c)
                if area < min_area:
                    continue
                x, y, w, h = cv2.boundingRect(c)
                aspect_ratio = w / h if h > 0 else 0
                bbox_area = w * h
                extent = area / bbox_area if bbox_area > 0 else 0
                if aspect_ratio > 6.0 and h < int(0.18 * h_img):
                    continue
                if h < max(10, int(0.02 * h_img)) or w < max(10, int(0.02 * w_img)):
                    continue
                if extent < 0.22:
                    continue
                bboxes.append((x, y, w, h))
            return self._merge_adjacent_boxes(bboxes, (h_img, w_img))

    def _merge_adjacent_boxes(self, bboxes, image_shape):
        """Merge boxes that significantly overlap or are separated by tiny gaps.
        Prevents large objects from being split into multiple parts.
        """
        if not bboxes:
            return []
        h_img, w_img = image_shape
        gap_thresh = int(0.005 * max(h_img, w_img))
        image_area = h_img * w_img
        small_area_thresh = 0.002 * image_area

        def iou(a, b):
            ax1, ay1, aw, ah = a
            bx1, by1, bw, bh = b
            ax2, ay2 = ax1 + aw, ay1 + ah
            bx2, by2 = bx1 + bw, by1 + bh
            inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
            inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
            inter_w = max(0, inter_x2 - inter_x1)
            inter_h = max(0, inter_y2 - inter_y1)
            inter = inter_w * inter_h
            union = aw * ah + bw * bh - inter
            return inter / union if union > 0 else 0.0

        def gap(a, b):
            ax1, ay1, aw, ah = a
            bx1, by1, bw, bh = b
            ax2, ay2 = ax1 + aw, ay1 + ah
            bx2, by2 = bx1 + bw, by1 + bh
            # Horizontal and vertical gaps (negative means overlap)
            gx = max(0, max(ax1, bx1) - min(ax2, bx2))
            gy = max(0, max(ay1, by1) - min(ay2, by2))
            return max(gx, gy)

        def overlap_lengths(a, b):
            ax1, ay1, aw, ah = a
            bx1, by1, bw, bh = b
            ax2, ay2 = ax1 + aw, ay1 + ah
            bx2, by2 = bx1 + bw, by1 + bh
            ovx = max(0, min(ax2, bx2) - max(ax1, bx1))
            ovy = max(0, min(ay2, by2) - max(ay1, by1))
            return ovx, ovy

        merged = True
        boxes = bboxes[:]
        while merged:
            merged = False
            new_boxes = []
            used = [False] * len(boxes)
            for i in range(len(boxes)):
                if used[i]:
                    continue
                a = boxes[i]
                ax1, ay1, aw, ah = a
                ax2, ay2 = ax1 + aw, ay1 + ah
                for j in range(i + 1, len(boxes)):
                    if used[j]:
                        continue
                    b = boxes[j]
                    ovx, ovy = overlap_lengths(a, b)
                    min_w = min(a[2], b[2])
                    min_h = min(a[3], b[3])
                    area_a = a[2] * a[3]
                    area_b = b[2] * b[3]
                    close_aligned = gap(a, b) <= gap_thresh and (ovx >= 0.75 * min_w or ovy >= 0.75 * min_h)
                    # Small-object safeguard: if both are small, require high IoU to merge
                    if (area_a < small_area_thresh and area_b < small_area_thresh):
                        should_merge = iou(a, b) > 0.6
                    else:
                        should_merge = iou(a, b) > 0.4 or close_aligned
                    if should_merge:
                        # merge
                        bx1, by1, bw, bh = b
                        bx2, by2 = bx1 + bw, by1 + bh
                        nx1, ny1 = min(ax1, bx1), min(ay1, by1)
                        nx2, ny2 = max(ax2, bx2), max(ay2, by2)
                        a = (nx1, ny1, nx2 - nx1, ny2 - ny1)
                        ax1, ay1, aw, ah = a
                        ax2, ay2 = ax1 + aw, ay1 + ah
                        used[j] = True
                        merged = True
                used[i] = True
                new_boxes.append(a)
            boxes = new_boxes
        return boxes

    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp")]
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
            for i, obj in enumerate(self.detected_objects):
                x, y, w, h, _, _ = self._norm_box(obj)
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
        """Automatically detect generic objects after loading an image"""
        try:
            # Try YOLO first
            if self._ensure_yolo():
                print("Running YOLO detection...")
                results = self.yolo_model(self.original_image, verbose=False)[0]
                yolo_boxes = []
                for b in results.boxes:
                    x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                    cls_id = int(b.cls[0]) if hasattr(b, 'cls') else -1
                    label = results.names[cls_id] if hasattr(results, 'names') and cls_id in results.names else ''
                    conf = float(b.conf[0]) if hasattr(b, 'conf') else 0.0
                    x, y, w, h = x1, y1, x2 - x1, y2 - y1
                    yolo_boxes.append({'x': x, 'y': y, 'w': w, 'h': h, 'label': label, 'conf': conf})
                print(f"YOLO found {len(yolo_boxes)} raw detections")
                # Keep only reasonably confident boxes
                self.detected_objects = [d for d in yolo_boxes if d['conf'] >= 0.1]
                print(f"After confidence filtering: {len(self.detected_objects)} objects")
                if len(self.detected_objects) == 0:
                    print("YOLO found no confident objects, falling back to classical method")
                    # Fall back to classical pipeline if YOLO found nothing
                    pass
                else:
                    self.highlight_detected_objects()
                    self.object_highlighted = True
                    messagebox.showinfo("Info", f"Found {len(self.detected_objects)} objects (YOLO). Click one to select as reference.")
                    return

            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            # Auto polarity: decide whether foreground should be white or black
            _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            white_ratio = np.mean(otsu == 255)
            if white_ratio > 0.5:
                # Background likely white; use inverse to get foreground white
                _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            else:
                _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Morphology: use small kernels to remove noise and close small gaps
            kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
            opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_open, iterations=1)
            cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close, iterations=1)

            # Extract bounding boxes with splitting and text filtering
            boxes = self._extract_bboxes_from_mask(cleaned)
            # Store as simple tuples for fallback, label empty
            self.detected_objects = [(x, y, w, h) for (x, y, w, h) in boxes]

            if len(self.detected_objects) == 0:
                messagebox.showinfo("Info", "No objects detected in the image.")
                return

            self.highlight_detected_objects()
            self.object_highlighted = True
            messagebox.showinfo("Info", f"Found {len(self.detected_objects)} objects. Click one to select as reference.")

        except Exception as e:
            messagebox.showerror("Error", f"Error detecting objects: {str(e)}")
    
    def highlight_detected_objects(self):
        if not self.detected_objects:
            return
        
        # Create a copy of the original image for highlighting
        highlight_image = self.original_image.copy()
        
        # Draw bounding boxes around detected objects
        for i, obj in enumerate(self.detected_objects):
            x, y, w, h, label, conf = self._norm_box(obj)
            # Draw rectangle with different color for each object
            color = (255, 0, 255)  # Magenta for detected objects
            cv2.rectangle(highlight_image, (x, y), (x+w, y+h), color, 2)
            name = label if label else f"Object {i+1}"
            cv2.putText(highlight_image, name, (x, y-10), 
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
        for i, obj in enumerate(self.detected_objects):
            x, y, w, h, label, conf = self._norm_box(obj)
            if i == selected_index:
                # Highlight selected object in green
                color = (0, 255, 0)  # Green for selected object
                thickness = 3
                ref_text = f"Reference: {label}" if label else f"Reference (Object {i+1})"
                cv2.putText(highlight_image, ref_text, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            else:
                # Other objects in magenta
                color = (255, 0, 255)  # Magenta for other objects
                thickness = 2
                name = label if label else f"Object {i+1}"
                cv2.putText(highlight_image, name, (x, y-10), 
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
            
            # Convert to grayscale and threshold with auto-polarity
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            white_ratio = np.mean(otsu == 255)
            if white_ratio > 0.5:
                _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            else:
                _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Generic morphology
            kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
            opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_open, iterations=1)
            cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close, iterations=1)

            # Bboxes as contours-like proxies (fallback)
            bboxes = self._extract_bboxes_from_mask(cleaned)

            # If YOLO is available, prefer YOLO detections for counting
            yolo_used = False
            ref_label = ''
            if self._ensure_yolo():
                # Determine reference class by intersecting click ROI with YOLO boxes
                results = self.yolo_model(self.original_image, verbose=False)[0]
                yolo_boxes = []
                for b in results.boxes:
                    x1b, y1b, x2b, y2b = map(int, b.xyxy[0].tolist())
                    cls_id = int(b.cls[0]) if hasattr(b, 'cls') else -1
                    label = results.names[cls_id] if hasattr(results, 'names') and cls_id in results.names else ''
                    conf = float(b.conf[0]) if hasattr(b, 'conf') else 0.0
                    yolo_boxes.append((x1b, y1b, x2b - x1b, y2b - y1b, label, conf))
                # Pick the YOLO box that overlaps the selected ROI the most
                def iou_roi(box):
                    bx, by, bw, bh, _, _ = box
                    bx2, by2 = bx + bw, by + bh
                    ix1, iy1 = max(bx, x1), max(by, y1)
                    ix2, iy2 = min(bx2, x2), min(by2, y2)
                    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
                    inter = iw * ih
                    union = (bw * bh) + ((x2 - x1) * (y2 - y1)) - inter
                    return inter / union if union > 0 else 0.0
                if len(yolo_boxes) > 0:
                    ref_idx = max(range(len(yolo_boxes)), key=lambda i: iou_roi(yolo_boxes[i]))
                    if iou_roi(yolo_boxes[ref_idx]) > 0.1:
                        ref_label = yolo_boxes[ref_idx][4]
                        # Use only same-class YOLO boxes for counting
                        sel = [b for b in yolo_boxes if b[4] == ref_label and b[5] >= 0.25]
                        if len(sel) > 0:
                            bboxes = [(bx, by, bw, bh) for (bx, by, bw, bh, _, _) in sel]
                            yolo_used = True
            
            # Check if any valid objects were found
            if len(bboxes) == 0:
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
            
            # Thresholds relative to selected reference
            small_threshold = 0.7
            large_threshold = 1.3
            
            for (x, y, w, h) in bboxes:
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
                'no_objects_found': False,
                'yolo_used': yolo_used,
                'reference_label': ref_label
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
            if self.results.get('yolo_used', False):
                results_text += f"Filtered by class: {self.results.get('reference_label','')} (YOLO)\n\n"
            results_text += "Detection Settings:\n"
            results_text += "- Dynamic min area & watershed splitting\n"
            results_text += "- Auto foreground polarity (Otsu)\n"
            results_text += "- Morphology: open(3x3) + close(7x7)\n"
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
