import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import numpy as np
from PIL import Image, ImageTk
from sklearn.cluster import DBSCAN
from dataclasses import dataclass
from typing import Tuple
import time

# ----------------------- Utilities for Color Segmentation -----------------------
@dataclass
class Tolerances:
    hue: int = 15
    sat: int = 60
    val: int = 60
    r: int = 30
    g: int = 30
    b: int = 30

def clamp_hsv_range(center: Tuple[int, int, int], tol: Tolerances) -> Tuple[np.ndarray, np.ndarray]:
    h, s, v = center
    lower = np.array([max(0, h - tol.hue), max(0, s - tol.sat), max(0, v - tol.val)], dtype=np.uint8)
    upper = np.array([min(179, h + tol.hue), min(255, s + tol.sat), min(255, v + tol.val)], dtype=np.uint8)
    return lower, upper

def clamp_bgr_range(center: Tuple[int, int, int], tol: Tolerances) -> Tuple[np.ndarray, np.ndarray]:
    b, g, r = center
    lower = np.array([max(0, b - tol.b), max(0, g - tol.g), max(0, r - tol.r)], dtype=np.uint8)
    upper = np.array([min(255, b + tol.b), min(255, g + tol.g), min(255, r + tol.r)], dtype=np.uint8)
    return lower, upper

def get_most_frequent_color(roi: np.ndarray, color_space: str = 'bgr') -> Tuple[int, int, int]:
    """Find the most frequent color in the ROI, excluding white/background colors."""
    if roi.size == 0:
        return (0, 0, 0)
    
    if color_space == 'hsv':
        roi_converted = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    else:
        roi_converted = roi.copy()
    
    pixels = roi_converted.reshape(-1, 3)
    
    if len(pixels) > 10000:
        step = len(pixels) // 5000
        pixels = pixels[::step]
    
    if color_space == 'hsv':
        mask = (pixels[:, 1] > 30) & (pixels[:, 2] < 240)
        filtered_pixels = pixels[mask]
    else:
        brightness = np.mean(pixels, axis=1)
        mask = brightness < 200
        filtered_pixels = pixels[mask]
    
    if len(filtered_pixels) < len(pixels) * 0.1:
        filtered_pixels = pixels
    
    unique_colors, counts = np.unique(filtered_pixels, axis=0, return_counts=True)
    most_frequent_idx = np.argmax(counts)
    most_frequent_color = unique_colors[most_frequent_idx]
    
    return tuple(map(int, most_frequent_color))

def compute_mask(img_bgr: np.ndarray, roi_mean_bgr: Tuple[int, int, int], 
                roi_mean_hsv: Tuple[int, int, int], tol: Tolerances, 
                use_hsv: bool, morph_kernel: int) -> np.ndarray:
    if use_hsv:
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        h, s, v = roi_mean_hsv
        h_low = h - tol.hue
        h_high = h + tol.hue
        s_low = max(0, s - tol.sat)
        s_high = min(255, s + tol.sat)
        v_low = max(0, v - tol.val)
        v_high = min(255, v + tol.val)
        if h_low < 0 or h_high > 179:
            h_low_1 = (h_low + 180) if h_low < 0 else h_low
            h_high_1 = 179
            h_low_2 = 0
            h_high_2 = (h_high - 180) if h_high > 179 else h_high
            low1 = np.array([h_low_1, s_low, v_low], dtype=np.uint8)
            high1 = np.array([h_high_1, s_high, v_high], dtype=np.uint8)
            low2 = np.array([h_low_2, s_low, v_low], dtype=np.uint8)
            high2 = np.array([h_high_2, s_high, v_high], dtype=np.uint8)
            mask = cv2.inRange(hsv, low1, high1) | cv2.inRange(hsv, low2, high2)
        else:
            low = np.array([max(0, h_low), s_low, v_low], dtype=np.uint8)
            high = np.array([min(179, h_high), s_high, v_high], dtype=np.uint8)
            mask = cv2.inRange(hsv, low, high)
    else:
        low, high = clamp_bgr_range(roi_mean_bgr, tol)
        mask = cv2.inRange(img_bgr, low, high)

    if morph_kernel > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return mask

def count_and_draw(mask: np.ndarray, img_rgb: np.ndarray, min_area: int):
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    vis = img_rgb.copy()
    count = 0
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        count += 1
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cx, cy = centroids[i]
        cv2.circle(vis, (int(cx), int(cy)), 3, (0, 0, 255), -1)
    return vis, count

# ----------------------- Main Application Class -----------------------
class UnifiedObjectCounter:
    def __init__(self, root):
        self.root = root
        self.root.title("Unified Smart Object Counter")
        self.root.geometry("1400x900")

        # Images/state
        self.image = None               # RGB copy for display
        self.original_image = None      # BGR original for processing
        self.display_image = None       # RGB display image
        self.photo = None
        self.scale_factor = 1.0
        self.canvas_offset_x = 0
        self.canvas_offset_y = 0

        # ROI
        self.selected_roi = None

        # Template matching settings
        self.template_threshold = 0.7
        self.orb_ratio = 0.75
        self.postproc_mode = "none"
        
        # Color segmentation settings
        self.tol = Tolerances()
        self.kernel = 3
        self.min_area = 100
        self.use_hsv = True
        self.use_watershed = False
        self.ws_sensitivity = 35

        # Detection mode
        self.detection_mode = "template"  # "template" or "color"
        self.appearance_method = "grayscale"  # "grayscale", "color", "template"
        self.detect_rotated = False
        self.detect_different_sizes = False
        
        # YOLO detection
        self.detected_objects = []
        
        # YOLO model (lazy loaded)
        self.yolo_model = None
        self.yolo_names = None
        self.use_yolo = True

        self.setup_gui()

    def setup_gui(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        left_panel = ttk.Frame(main_frame, width=400)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_panel.pack_propagate(False)

        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Mode Selection
        mode_frame = ttk.LabelFrame(left_panel, text="Detection Mode")
        mode_frame.pack(fill=tk.X, pady=10)
        
        self.mode_var = tk.StringVar(value="template")
        ttk.Radiobutton(mode_frame, text="Template Matching", variable=self.mode_var, 
                       value="template", command=self.on_mode_change).pack(anchor=tk.W)
        ttk.Radiobutton(mode_frame, text="Color Segmentation", variable=self.mode_var, 
                       value="color", command=self.on_mode_change).pack(anchor=tk.W)

        # Template Matching Options
        self.template_frame = ttk.LabelFrame(left_panel, text="Appearance Options")
        self.template_frame.pack(fill=tk.X, pady=10)
        
        # Appearance method selection
        self.appearance_var = tk.StringVar(value="grayscale")
        ttk.Radiobutton(self.template_frame, text="Detect by grayscale matching", 
                       variable=self.appearance_var, value="grayscale").pack(anchor=tk.W)
        ttk.Radiobutton(self.template_frame, text="Detect by color matching", 
                       variable=self.appearance_var, value="color").pack(anchor=tk.W)

        # Detection flexibility options
        flexibility_frame = ttk.LabelFrame(self.template_frame, text="Detection Flexibility")
        flexibility_frame.pack(fill=tk.X, pady=10)
        
        self.rotated_var = tk.BooleanVar(value=False)
        self.sizes_var = tk.BooleanVar(value=False)
        
        ttk.Checkbutton(flexibility_frame, text="Detect rotated objects (slower)", 
                       variable=self.rotated_var).pack(anchor=tk.W)
        ttk.Checkbutton(flexibility_frame, text="Detect objects of different sizes", 
                       variable=self.sizes_var).pack(anchor=tk.W)

        # Color Segmentation Options
        self.color_frame = ttk.LabelFrame(left_panel, text="Color Segmentation Options")
        self.color_frame.pack(fill=tk.X, pady=10)
        
        # Color space selection
        color_mode_frame = ttk.Frame(self.color_frame)
        color_mode_frame.pack(fill=tk.X, pady=5)
        self.color_mode_var = tk.IntVar(value=1)
        ttk.Radiobutton(color_mode_frame, text="HSV", variable=self.color_mode_var, 
                       value=1, command=self.on_color_mode_change).pack(side=tk.LEFT)
        ttk.Radiobutton(color_mode_frame, text="BGR", variable=self.color_mode_var, 
                       value=0, command=self.on_color_mode_change).pack(side=tk.LEFT)
        
        # Sliders
        self.hue_label = ttk.Label(self.color_frame, text="Hue tol")
        self.hue_label.pack(anchor=tk.W, padx=6)
        self.hue_slider = tk.Scale(self.color_frame, from_=0, to=90, orient=tk.HORIZONTAL,
                                   command=lambda v: self.on_slider_change())
        self.hue_slider.set(self.tol.hue)
        self.hue_slider.pack(fill=tk.X)

        self.sat_label = ttk.Label(self.color_frame, text="Sat tol")
        self.sat_label.pack(anchor=tk.W, padx=6)
        self.sat_slider = tk.Scale(self.color_frame, from_=0, to=127, orient=tk.HORIZONTAL,
                                   command=lambda v: self.on_slider_change())
        self.sat_slider.set(self.tol.sat)
        self.sat_slider.pack(fill=tk.X)

        self.val_label = ttk.Label(self.color_frame, text="Val tol")
        self.val_label.pack(anchor=tk.W, padx=6)
        self.val_slider = tk.Scale(self.color_frame, from_=0, to=127, orient=tk.HORIZONTAL,
                                   command=lambda v: self.on_slider_change())
        self.val_slider.set(self.tol.val)
        self.val_slider.pack(fill=tk.X)

        ttk.Label(self.color_frame, text="Morph kernel").pack(anchor=tk.W, padx=6)
        self.kernel_slider = tk.Scale(self.color_frame, from_=0, to=25, orient=tk.HORIZONTAL,
                                      command=lambda v: self.on_slider_change())
        self.kernel_slider.set(self.kernel)
        self.kernel_slider.pack(fill=tk.X)

        ttk.Label(self.color_frame, text="Min area").pack(anchor=tk.W, padx=6)
        self.min_slider = tk.Scale(self.color_frame, from_=1, to=20000, orient=tk.HORIZONTAL,
                                   command=lambda v: self.on_slider_change())
        self.min_slider.set(self.min_area)
        self.min_slider.pack(fill=tk.X)

        # Watershed options
        ws_frame = ttk.Frame(self.color_frame)
        ws_frame.pack(fill=tk.X, pady=5)
        self.ws_var = tk.IntVar(value=0)
        ttk.Checkbutton(ws_frame, text="Split touching objects (watershed)", 
                       variable=self.ws_var, command=self.on_ws_toggle).pack(anchor=tk.W)
        ttk.Label(ws_frame, text="Sensitivity").pack(anchor=tk.W, padx=6)
        self.ws_slider = tk.Scale(ws_frame, from_=1, to=80, orient=tk.HORIZONTAL,
                                  command=lambda v: self.on_ws_slider_change())
        self.ws_slider.set(self.ws_sensitivity)
        self.ws_slider.pack(fill=tk.X)

        # Controls
        ttk.Button(left_panel, text="Load Image", command=self.load_image).pack(fill=tk.X, pady=4)
        self.count_button = ttk.Button(left_panel, text="Count Objects", command=self.count_objects)
        self.count_button.pack(fill=tk.X, pady=4)
        ttk.Button(left_panel, text="Reset", command=self.reset).pack(fill=tk.X, pady=4)
        ttk.Button(left_panel, text="Save Result", command=self.save_result).pack(fill=tk.X, pady=4)

        # Instructions
        instruction_frame = ttk.LabelFrame(left_panel, text="Instructions")
        instruction_frame.pack(fill=tk.X, pady=10)
        instructions = """
1. Select detection mode (Template or Color)
2. Choose appearance options and flexibility
3. Load an image (YOLO will auto-detect objects)
4. Click on a detected object or drag a box to select reference
5. Click "Count Objects" to analyze
        """
        ttk.Label(instruction_frame, text=instructions, justify=tk.LEFT).pack(padx=10, pady=8)

        # Results Display
        results_frame = ttk.LabelFrame(left_panel, text="Results")
        results_frame.pack(fill=tk.BOTH, pady=10, expand=True)
        self.results_text = tk.Text(results_frame, height=12, width=45)
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Image canvas
        self.canvas = tk.Canvas(right_panel, bg="white")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)

        # Initialize UI state
        self.on_mode_change()
        self.clear_results()

    def on_mode_change(self):
        """Handle detection mode change"""
        mode = self.mode_var.get()
        if mode == "template":
            self.template_frame.pack(fill=tk.X, pady=10)
            self.color_frame.pack_forget()
        else:
            self.template_frame.pack_forget()
            self.color_frame.pack(fill=tk.X, pady=10)

    def on_color_mode_change(self):
        """Handle color space mode change"""
        self.use_hsv = (self.color_mode_var.get() == 1)
        
        if self.use_hsv:
            self.hue_label.config(text="Hue tol")
            self.sat_label.config(text="Sat tol")
            self.val_label.config(text="Val tol")
            self.hue_slider.config(from_=0, to=90)
            self.sat_slider.config(from_=0, to=127)
            self.val_slider.config(from_=0, to=127)
            self.hue_slider.set(self.tol.hue)
            self.sat_slider.set(self.tol.sat)
            self.val_slider.set(self.tol.val)
        else:
            self.hue_label.config(text="Blue tol")
            self.sat_label.config(text="Green tol")
            self.val_label.config(text="Red tol")
            self.hue_slider.config(from_=0, to=255)
            self.sat_slider.config(from_=0, to=255)
            self.val_slider.config(from_=0, to=255)
            self.hue_slider.set(self.tol.b)
            self.sat_slider.set(self.tol.g)
            self.val_slider.set(self.tol.r)

    def on_slider_change(self):
        """Handle slider changes"""
        if self.use_hsv:
            self.tol.hue = int(self.hue_slider.get())
            self.tol.sat = int(self.sat_slider.get())
            self.tol.val = int(self.val_slider.get())
        else:
            self.tol.b = int(self.hue_slider.get())
            self.tol.g = int(self.sat_slider.get())
            self.tol.r = int(self.val_slider.get())
        
        self.kernel = int(self.kernel_slider.get())
        self.min_area = int(self.min_slider.get())

    def on_ws_toggle(self):
        """Handle watershed toggle"""
        self.use_watershed = (self.ws_var.get() == 1)

    def on_ws_slider_change(self):
        """Handle watershed sensitivity change"""
        self.ws_sensitivity = int(self.ws_slider.get())

    # ----------------------- Image / ROI handlers -----------------------
    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp *.avif")]
        )
        if not file_path:
            return
        
        img_bgr = cv2.imread(file_path)
        if img_bgr is None:
            messagebox.showerror("Error", "Failed to load image")
            return

        self.original_image = img_bgr.copy()
        self.image = cv2.cvtColor(img_bgr.copy(), cv2.COLOR_BGR2RGB)
        self.display_image = self.image.copy()

        self.selected_roi = None
        self.display_image_on_canvas()
        self.clear_results()
        
        # Auto-detect objects using YOLOv8
        self.auto_detect_objects()

    def display_image_on_canvas(self):
        if self.display_image is None:
            return
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        if canvas_width <= 1 or canvas_height <= 1:
            self.root.after(100, self.display_image_on_canvas)
            return

        h, w = self.display_image.shape[:2]
        scale = min(canvas_width / w, canvas_height / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(self.display_image, (new_w, new_h))
        pil = Image.fromarray(resized)
        self.photo = ImageTk.PhotoImage(pil)

        self.canvas.delete("all")
        self.canvas.create_image(canvas_width // 2, canvas_height // 2, image=self.photo, anchor=tk.CENTER)
        self.scale_factor = scale
        self.canvas_offset_x = (canvas_width - new_w) // 2
        self.canvas_offset_y = (canvas_height - new_h) // 2

    def on_mouse_down(self, event):
        if self.image is None:
            return
        
        # If objects are highlighted, try to select one
        if hasattr(self, 'detected_objects') and self.detected_objects:
            # Convert canvas coordinates to image coordinates
            canvas_x = (event.x - self.canvas_offset_x) / self.scale_factor
            canvas_y = (event.y - self.canvas_offset_y) / self.scale_factor
            
            # Check if click is within any detected object
            for i, obj in enumerate(self.detected_objects):
                x, y, w, h = obj['x'], obj['y'], obj['w'], obj['h']
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
        if not getattr(self, "drawing", False):
            return
        self.rect_end = (event.x, event.y)
        self.canvas.delete("selection_rect")
        x1, y1 = self.rect_start
        x2, y2 = self.rect_end
        self.canvas.create_rectangle(x1, y1, x2, y2, outline="red", width=2, tags="selection_rect")

    def on_mouse_up(self, event):
        if not getattr(self, "drawing", False):
            return
        self.drawing = False
        self.rect_end = (event.x, event.y)
        if self.rect_start and self.rect_end:
            x1 = (min(self.rect_start[0], self.rect_end[0]) - self.canvas_offset_x) / self.scale_factor
            y1 = (min(self.rect_start[1], self.rect_end[1]) - self.canvas_offset_y) / self.scale_factor
            x2 = (max(self.rect_start[0], self.rect_end[0]) - self.canvas_offset_x) / self.scale_factor
            y2 = (max(self.rect_start[1], self.rect_end[1]) - self.canvas_offset_y) / self.scale_factor
            x1 = max(0, int(round(x1))); y1 = max(0, int(round(y1)))
            x2 = min(self.image.shape[1], int(round(x2))); y2 = min(self.image.shape[0], int(round(y2)))
            if x2 > x1 and y2 > y1:
                self.selected_roi = (x1, y1, x2, y2)
                messagebox.showinfo("Info", f"ROI selected: {self.selected_roi}")

    # ----------------------- Template Matching Methods -----------------------
    def rotate_image(self, img, angle):
        """Rotate a BGR numpy array about its center, keep same size"""
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
        return rotated

    def non_max_suppression(self, boxes, overlapThresh=0.3):
        """Simple IoU-based NMS for boxes in format [x,y,w,h]."""
        if len(boxes) == 0:
            return []
        boxes = np.array(boxes).astype(float)
        x1 = boxes[:,0]
        y1 = boxes[:,1]
        x2 = boxes[:,0] + boxes[:,2]
        y2 = boxes[:,1] + boxes[:,3]
        areas = (x2 - x1) * (y2 - y1)
        idxs = np.argsort(areas)[::-1]
        picked = []
        while len(idxs) > 0:
            i = idxs[0]
            picked.append(i)
            xx1 = np.maximum(x1[i], x1[idxs[1:]])
            yy1 = np.maximum(y1[i], y1[idxs[1:]])
            xx2 = np.minimum(x2[i], x2[idxs[1:]])
            yy2 = np.minimum(y2[i], y2[idxs[1:]])
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            inter = w * h
            overlap = inter / (areas[idxs[1:]] + 1e-6)
            idxs = idxs[np.where(overlap <= overlapThresh)[0] + 1]
        return boxes[picked].astype(int).tolist()

    def preprocess_gray(self, bgr_img):
        """Convert BGR to grayscale and enhance contrast using CLAHE."""
        gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        return gray

    def compute_auto_template_threshold(self, corr_map: np.ndarray) -> float:
        """Estimate a suitable threshold for cv2.matchTemplate correlation map."""
        flat_corr = corr_map.flatten()
        p99 = float(np.percentile(flat_corr, 99.0))
        p95 = float(np.percentile(flat_corr, 95.0))
        p90 = float(np.percentile(flat_corr, 90.0))
        
        std_dev = float(np.std(flat_corr))
        
        if std_dev > 0.3:
            threshold = 0.6 * p99 + 0.4 * p95
        else:
            threshold = 0.5 * p99 + 0.3 * p95 + 0.2 * p90
        
        threshold = float(np.clip(threshold, 0.45, 0.85))
        return threshold

    def template_matching_detection(self):
        """Perform template matching based on selected options"""
        if self.selected_roi is None:
            return [], "No ROI selected"
        
        x1, y1, x2, y2 = self.selected_roi
        roi_bgr = self.original_image[y1:y2, x1:x2]
        if roi_bgr.size == 0:
            return [], "Invalid ROI"
        
        appearance_method = self.appearance_var.get()
        detect_rotated = self.rotated_var.get()
        detect_different_sizes = self.sizes_var.get()
        
        # Show progress in results area
        if detect_rotated:
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, "⚠️ WARNING: Rotation detection is enabled.\n")
            self.results_text.insert(tk.END, "This will process 24 different angles and may take 10-30 seconds.\n")
            self.results_text.insert(tk.END, "Processing rotations... Please wait.\n")
            self.results_text.see(tk.END)
            self.root.update()
        
        rectangles = []
        method_info = f"Template Matching ({appearance_method})"
        
        # Determine preprocessing based on appearance method
        if appearance_method == "grayscale":
            img_gray = self.preprocess_gray(self.original_image)
            roi_gray = self.preprocess_gray(roi_bgr)
        elif appearance_method == "color":
            img_gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        else:  # template
            # For template matching, use edge-based detection for better accuracy
            img_gray = cv2.Canny(self.preprocess_gray(self.original_image), 50, 150)
            roi_gray = cv2.Canny(self.preprocess_gray(roi_bgr), 50, 150)
        
        # Base detection without rotation/scaling
        if not detect_rotated and not detect_different_sizes:
            res = cv2.matchTemplate(img_gray, roi_gray, cv2.TM_CCOEFF_NORMED)
            auto_thr = self.compute_auto_template_threshold(res)
            self.template_threshold = auto_thr
            
            loc = np.where(res >= auto_thr)
            for pt in zip(*loc[::-1]):
                rectangles.append([pt[0], pt[1], roi_gray.shape[1], roi_gray.shape[0]])
        
        # Rotation detection
        if detect_rotated:
            # Use coarser angle increments for better performance, but still accurate
            angles = range(0, 360, 15)  # 15-degree increments for better performance
            
            # Update GUI to show progress
            self.root.title("Unified Smart Object Counter - Processing rotations...")
            self.root.update()
            
            # Add timeout protection
            start_time = time.time()
            max_time = 30  # Maximum 30 seconds for rotation detection
            
            for i, angle in enumerate(angles):
                # Check timeout
                if time.time() - start_time > max_time:
                    self.results_text.insert(tk.END, f"\nTimeout reached after {max_time} seconds. Processing stopped.\n")
                    break
                
                # Update progress every few angles
                if i % 4 == 0:
                    progress = (i / len(angles)) * 100
                    self.root.title(f"Unified Smart Object Counter - Processing rotations... {progress:.0f}%")
                    self.root.update()
                
                rotated_roi = self.rotate_image(roi_bgr, angle)
                if appearance_method == "template":
                    # For template matching, use edge-based detection for better rotation handling
                    rotated_gray = cv2.Canny(self.preprocess_gray(rotated_roi), 50, 150)
                else:
                    rotated_gray = cv2.cvtColor(rotated_roi, cv2.COLOR_BGR2GRAY)
                
                res = cv2.matchTemplate(img_gray, rotated_gray, cv2.TM_CCOEFF_NORMED)
                auto_thr = self.compute_auto_template_threshold(res)
                
                # Use a slightly lower threshold for rotation detection to catch more variations
                rotation_thr = max(0.4, auto_thr - 0.1)
                
                loc = np.where(res >= rotation_thr)
                for pt in zip(*loc[::-1]):
                    rectangles.append([pt[0], pt[1], rotated_gray.shape[1], rotated_gray.shape[0]])
            
            # Reset title
            self.root.title("Unified Smart Object Counter")
        
        # Size scaling detection
        if detect_different_sizes:
            scales = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]
            
            for scale in scales:
                scaled_w = int(roi_gray.shape[1] * scale)
                scaled_h = int(roi_gray.shape[0] * scale)
                if scaled_w < 10 or scaled_h < 10:
                    continue
                
                scaled_roi = cv2.resize(roi_gray, (scaled_w, scaled_h))
                res = cv2.matchTemplate(img_gray, scaled_roi, cv2.TM_CCOEFF_NORMED)
                auto_thr = self.compute_auto_template_threshold(res)
                
                loc = np.where(res >= auto_thr)
                for pt in zip(*loc[::-1]):
                    rectangles.append([pt[0], pt[1], scaled_w, scaled_h])
        
        # Apply NMS to remove overlapping detections
        if rectangles:
            rectangles, _ = cv2.groupRectangles(rectangles, groupThreshold=1, eps=0.5)
            rectangles = self.non_max_suppression(rectangles, overlapThresh=0.3)
        
        return rectangles, method_info

    # ----------------------- Color Segmentation Methods -----------------------
    def color_segmentation_detection(self):
        """Perform color-based segmentation detection"""
        if self.selected_roi is None:
            return [], "No ROI selected"
        
        x1, y1, x2, y2 = self.selected_roi
        roi = self.original_image[y1:y2, x1:x2]
        
        # Get color information from ROI
        roi_mean_bgr = get_most_frequent_color(roi, 'bgr')
        roi_mean_hsv = get_most_frequent_color(roi, 'hsv')
        
        # Compute mask using current settings
        mask = compute_mask(self.original_image, roi_mean_bgr, roi_mean_hsv,
                           self.tol, self.use_hsv, self.kernel)
        
        # Optional watershed split
        if self.use_watershed:
            mask = self.apply_watershed_split(mask)
        
        # Extract objects
        num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        objects = []
        for i in range(1, num):
            x, y, w, h, area = stats[i]
            if area >= self.min_area:
                objects.append((x, y, w, h, area))
        
        method_info = f"Color Segmentation ({'HSV' if self.use_hsv else 'BGR'})"
        return objects, method_info

    def apply_watershed_split(self, mask):
        """Split touching objects using watershed"""
        if mask is None:
            return mask
        
        bin_mask = (mask > 0).astype(np.uint8) * 255
        ksize = max(3, self.kernel | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        
        sure_bg = cv2.dilate(bin_mask, kernel, iterations=1)
        dist = cv2.distanceTransform(bin_mask, cv2.DIST_L2, 5)
        
        if dist.max() > 0:
            dist_norm = dist / (dist.max() + 1e-6)
        else:
            dist_norm = dist
        
        thr = 0.2 + (min(max(self.ws_sensitivity, 1), 80) / 80.0) * (0.7 - 0.2)
        sure_fg = (dist_norm > thr).astype(np.uint8) * 255
        
        unknown = cv2.subtract(sure_bg, sure_fg)
        num_markers, markers = cv2.connectedComponents((sure_fg > 0).astype(np.uint8))
        markers = markers + 1
        markers[unknown > 0] = 0
        
        ws_markers = cv2.watershed(self.original_image.copy(), markers)
        refined = np.zeros_like(bin_mask)
        refined[ws_markers > 1] = 255
        
        return refined

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

    def auto_detect_objects(self):
        """Automatically detect objects using YOLOv8 after loading an image"""
        try:
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
                if len(self.detected_objects) > 0:
                    self.highlight_detected_objects()
                    messagebox.showinfo("Info", f"Found {len(self.detected_objects)} objects using YOLO. Click one to select as reference.")
                    return
                else:
                    print("YOLO found no confident objects")
            else:
                print("YOLO not available, skipping auto-detection")
        except Exception as e:
            print(f"Error in auto-detection: {e}")

    def highlight_detected_objects(self):
        """Highlight detected objects on the image"""
        if not hasattr(self, 'detected_objects') or not self.detected_objects:
            return
        
        # Create a copy of the original image for highlighting
        highlight_image = self.original_image.copy()
        
        # Draw bounding boxes around detected objects
        for i, obj in enumerate(self.detected_objects):
            x, y, w, h = obj['x'], obj['y'], obj['w'], obj['h']
            label = obj.get('label', f'Object {i+1}')
            # Draw rectangle with different color for each object
            color = (255, 0, 255)  # Magenta for detected objects
            cv2.rectangle(highlight_image, (x, y), (x+w, y+h), color, 2)
            cv2.putText(highlight_image, label, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Update display
        self.display_image = cv2.cvtColor(highlight_image, cv2.COLOR_BGR2RGB)
        self.display_image_on_canvas()

    def highlight_selected_object(self, selected_index):
        """Highlight the selected object and show others"""
        if not hasattr(self, 'detected_objects') or not self.detected_objects:
            return
        
        # Create a copy of the original image for highlighting
        highlight_image = self.original_image.copy()
        
        # Draw bounding boxes around all detected objects
        for i, obj in enumerate(self.detected_objects):
            x, y, w, h = obj['x'], obj['y'], obj['w'], obj['h']
            label = obj.get('label', f'Object {i+1}')
            
            if i == selected_index:
                # Highlight selected object in green
                color = (0, 255, 0)  # Green for selected object
                thickness = 3
                cv2.putText(highlight_image, f"Reference: {label}", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            else:
                # Other objects in magenta
                color = (255, 0, 255)  # Magenta for other objects
                thickness = 2
                cv2.putText(highlight_image, label, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            cv2.rectangle(highlight_image, (x, y), (x+w, y+h), color, thickness)
        
        # Update display
        self.display_image = cv2.cvtColor(highlight_image, cv2.COLOR_BGR2RGB)
        self.display_image_on_canvas()

    # ----------------------- Main Detection Method -----------------------
    def count_objects(self):
        if self.original_image is None:
            messagebox.showwarning("Warning", "Please load an image first")
            return
        if self.selected_roi is None:
            messagebox.showwarning("Warning", "Please select a reference ROI")
            return

        try:
            # Disable the count button to prevent multiple clicks
            self.count_button.config(state='disabled')
            self.root.title("Unified Smart Object Counter - Processing...")
            self.root.update()
            
            mode = self.mode_var.get()
            
            if mode == "template":
                # Template matching detection
                rectangles, method_info = self.template_matching_detection()
                
                if not rectangles:
                    self.results = {
                        'objects': [],
                        'reference_area': 0,
                        'no_objects_found': True,
                        'method': method_info
                    }
                else:
                    # Convert rectangles to objects format
                    objects = []
                    for x, y, w, h in rectangles:
                        area = w * h
                        objects.append((x, y, w, h, area))
                    
                    self.results = {
                        'objects': objects,
                        'reference_area': (self.selected_roi[2] - self.selected_roi[0]) * 
                                        (self.selected_roi[3] - self.selected_roi[1]),
                        'no_objects_found': False,
                        'method': method_info
                    }
                
            else:
                # Color segmentation detection
                objects, method_info = self.color_segmentation_detection()
                
                if not objects:
                    self.results = {
                        'objects': [],
                        'reference_area': 0,
                        'no_objects_found': True,
                        'method': method_info
                    }
                else:
                    self.results = {
                        'objects': objects,
                        'reference_area': (self.selected_roi[2] - self.selected_roi[0]) * 
                                        (self.selected_roi[3] - self.selected_roi[1]),
                        'no_objects_found': False,
                        'method': method_info
                    }
            
            # Display results
            self.display_results()
            self.draw_results_on_image()
            
        except Exception as e:
            messagebox.showerror("Error", f"Error counting objects: {str(e)}")
        finally:
            # Re-enable the count button and reset title
            self.count_button.config(state='normal')
            self.root.title("Unified Smart Object Counter")

    def display_results(self):
        self.results_text.delete(1.0, tk.END)
        
        if not self.results:
            self.results_text.insert(tk.END, "No results available.\nPlease count objects first.")
            return
        
        if self.results.get('no_objects_found', False):
            results_text = "No objects found in the image.\n\n"
            results_text += f"Objects Detected : 0\n\n"
            results_text += f"Method: {self.results.get('method', 'Unknown')}\n"
            results_text += "Note: No valid objects detected.\n"
            results_text += "Please ensure you have selected an area with objects."
        else:
            results_text = f"Reference Object Area: {self.results['reference_area']:.1f} pixels\n\n"
            results_text += f"Objects Detected : {len(self.results['objects'])}\n\n"
            results_text += f"Method: {self.results.get('method', 'Unknown')}\n\n"
            
            if self.mode_var.get() == "template":
                results_text += "Template Matching Settings:\n"
                results_text += f"- Appearance Method: {self.appearance_var.get()}\n"
                results_text += f"- Detect Rotated: {'Yes' if self.rotated_var.get() else 'No'}\n"
                results_text += f"- Detect Different Sizes: {'Yes' if self.sizes_var.get() else 'No'}\n"
                results_text += f"- Auto Threshold: {self.template_threshold:.2f}\n"
            else:
                results_text += "Color Segmentation Settings:\n"
                if self.use_hsv:
                    results_text += f"- Mode: HSV (H:{self.tol.hue}, S:{self.tol.sat}, V:{self.tol.val})\n"
                else:
                    results_text += f"- Mode: BGR (B:{self.tol.b}, G:{self.tol.g}, R:{self.tol.r})\n"
                results_text += f"- Kernel: {self.kernel}, Min Area: {self.min_area}\n"
                results_text += f"- Watershed: {'On' if self.use_watershed else 'Off'}"
                if self.use_watershed:
                    results_text += f" (Sensitivity: {self.ws_sensitivity})"
        
        self.results_text.insert(tk.END, results_text)

    def draw_results_on_image(self):
        if not self.results:
            return
        
        # Create a copy of the original image for drawing
        result_image = self.original_image.copy()
        
        # Draw reference object
        if self.selected_roi and not self.results.get('no_objects_found', False):
            x1, y1, x2, y2 = self.selected_roi
            cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(result_image, "Reference", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Draw detected objects
        for x, y, w, h, area in self.results.get('objects', []):
            cv2.rectangle(result_image, (x, y), (x+w, y+h), (255, 0, 0), 1)
            cv2.putText(result_image, "O", (x, y-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
        
        # Update display
        self.display_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
        self.display_image_on_canvas()

    # ----------------------- Save & Reset -----------------------
    def save_result(self):
        if self.display_image is None:
            messagebox.showwarning("Warning", "No result image to save")
            return
        save_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                 filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg;*.jpeg")])
        if not save_path:
            return
        # display_image is RGB; convert to BGR 
        bgr = cv2.cvtColor(self.display_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, bgr)
        messagebox.showinfo("Saved", f"Result image saved to:\n{save_path}")

    def reset(self):
        self.image = None
        self.original_image = None
        self.display_image = None
        self.photo = None
        self.selected_roi = None
        self.detected_objects = []
        self.canvas.delete("all")
        self.clear_results()

    def clear_results(self):
        self.results = {}
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "No results yet. Load image and select ROI.\n")


def main():
    root = tk.Tk()
    app = UnifiedObjectCounter(root)

    def on_resize(event):
        if app.image is not None:
            app.display_image_on_canvas()

    root.bind("<Configure>", on_resize)
    root.mainloop()

if __name__ == "__main__":
    main()
