import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import numpy as np
from PIL import Image, ImageTk
from sklearn.cluster import DBSCAN
import os
from dataclasses import dataclass
from typing import Tuple

class SmartObjectCounter:
    def __init__(self, root):
        self.root = root
        self.root.title("Smart Object Counter with Rotation and Scaling")
        self.root.geometry("1200x800")

        # images/state
        self.image = None               # RGB copy for display
        self.original_image = None      # BGR original for processing
        self.display_image = None       # RGB display image
        self.photo = None
        self.scale_factor = 1.0
        self.canvas_offset_x = 0
        self.canvas_offset_y = 0

        # ROI
        self.selected_roi = None
        self.detected_objects = []  # Store detected object bounding boxes
        self.object_highlighted = False  # Track if objects are highlighted

        # App mode: template (existing) or color (segmentation)
        self.app_mode = tk.StringVar(value="template")

        # YOLO model (lazy loaded)
        self.yolo_model = None
        self.yolo_names = None
        self.use_yolo = True

        # settings (auto-tuned)
        self.template_threshold = 0.7
        self.orb_ratio = 0.75
        self.postproc_mode = "none"

        # ------------ Color Segmentation state ------------
        self.use_hsv = True
        self.kernel = 3
        self.min_area = 100
        self.ws_enabled = False
        self.ws_sensitivity = 35
        self.roi_photo = None
        # tolerance storage
        self.tol_hue = 15
        self.tol_sat = 60
        self.tol_val = 60
        self.tol_b = 30
        self.tol_g = 30
        self.tol_r = 30

        self.setup_gui()

    # ----------------------- GUI -----------------------
    def setup_gui(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left panel with scrollbar
        left_container = ttk.Frame(main_frame, width=380)
        left_container.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_container.pack_propagate(False)
        # Keep a fixed visual height so overflow content uses the scrollbar
        left_container.configure(height=600)

        left_canvas = tk.Canvas(left_container, borderwidth=0, highlightthickness=0)
        # Use classic Tk scrollbar with larger width for visibility
        left_scrollbar = tk.Scrollbar(left_container, orient="vertical", command=left_canvas.yview, width=16)
        left_panel = ttk.Frame(left_canvas)

        left_panel.bind("<Configure>", lambda e: left_canvas.configure(scrollregion=left_canvas.bbox("all")))
        left_panel_window = left_canvas.create_window((0, 0), window=left_panel, anchor="nw")
        left_canvas.configure(yscrollcommand=left_scrollbar.set)

        left_canvas.pack(side="left", fill=tk.BOTH, expand=True)
        left_scrollbar.pack(side="right", fill="y")

        # Ensure the embedded frame matches canvas width
        def _on_left_canvas_configure(event):
            try:
                left_canvas.itemconfig(left_panel_window, width=event.width)
            except Exception:
                pass
        left_canvas.bind("<Configure>", _on_left_canvas_configure)

        # Keep left sidebar height synced with the window height for scrolling
        def _sync_left_height(event):
            try:
                left_container.configure(height=event.height)
            except Exception:
                pass
        main_frame.bind("<Configure>", _sync_left_height)

        # Mouse wheel scrolling for left panel
        def _on_left_mousewheel(event):
            try:
                delta = int(-1 * (event.delta / 120))
            except Exception:
                delta = -1
            left_canvas.yview_scroll(delta, "units")
        left_canvas.bind_all("<MouseWheel>", _on_left_mousewheel)

        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Controls
        ttk.Button(left_panel, text="Load Image", command=self.load_image).pack(fill=tk.X, pady=4)
        ttk.Button(left_panel, text="Count Objects", command=self.count_objects).pack(fill=tk.X, pady=4)
        ttk.Button(left_panel, text="Reset", command=self.reset).pack(fill=tk.X, pady=4)
        ttk.Button(left_panel, text="Save Result", command=self.save_result).pack(fill=tk.X, pady=4)

        # Mode switch
        mode_frame = ttk.LabelFrame(left_panel, text="Mode")
        mode_frame.pack(fill=tk.X, pady=10)
        ttk.Radiobutton(mode_frame, text="Template Matching", variable=self.app_mode, value="template", command=self._on_mode_switch).pack(anchor=tk.W, padx=10, pady=2)
        ttk.Radiobutton(mode_frame, text="Color Segmentation", variable=self.app_mode, value="color", command=self._on_mode_switch).pack(anchor=tk.W, padx=10, pady=2)

        # ROI Selection Section
        self.roi_frame = ttk.LabelFrame(left_panel, text="ROI Selection")
        self.roi_frame.pack(fill=tk.X, pady=10)
        
        self.roi_method = tk.StringVar(value="auto")
        ttk.Radiobutton(self.roi_frame, text="Auto Detection", variable=self.roi_method, 
                       value="auto", command=self.on_roi_method_change).pack(anchor=tk.W, padx=10, pady=2)
        ttk.Radiobutton(self.roi_frame, text="Rectangle (drag)", variable=self.roi_method, 
                       value="rectangle", command=self.on_roi_method_change).pack(anchor=tk.W, padx=10, pady=2)
        
        ttk.Button(self.roi_frame, text="Clear", command=self.clear_roi).pack(fill=tk.X, padx=10, pady=5)
        
        # Ready to detect status
        ttk.Label(left_panel, text="Ready to detect", font=("Arial", 10, "bold")).pack(pady=5)

        # Always use grayscale detection without showing an appearance options section
        self.detection_method = tk.StringVar(value="grayscale")

        # Detection Flexibility Section
        self.flexibility_frame = ttk.LabelFrame(left_panel, text="Detection Flexibility")
        self.flexibility_frame.pack(fill=tk.X, pady=10)
        
        self.detect_rotated = tk.BooleanVar(value=True)
        self.detect_scaled = tk.BooleanVar(value=True)
        
        ttk.Checkbutton(self.flexibility_frame, text="Detect rotated objects", 
                       variable=self.detect_rotated).pack(anchor=tk.W, padx=10, pady=2)
        ttk.Checkbutton(self.flexibility_frame, text="Detect objects of different sizes", 
                       variable=self.detect_scaled).pack(anchor=tk.W, padx=10, pady=2)

        # Ensure order: ROI -> Flexibility (appearance options removed)
        try:
            self.flexibility_frame.pack_forget()
        except Exception:
            pass
        try:
            self.flexibility_frame.pack(fill=tk.X, pady=10, after=self.roi_frame)
        except Exception:
            self.flexibility_frame.pack(fill=tk.X, pady=10)

        # ---------------- Color Segmentation Section (Scrollable) ----------------
        self.color_frame = ttk.LabelFrame(left_panel, text="Color Segmentation Settings", padding=(8,8,8,8))
        self.color_frame.pack(fill=tk.BOTH, pady=10, expand=False)

        # Create a canvas + scrollbar inside color_frame
        self.color_canvas = tk.Canvas(self.color_frame, borderwidth=0, highlightthickness=0)
        # Make the color section scrollbar more visible
        self.color_scrollbar = tk.Scrollbar(self.color_frame, orient="vertical", command=self.color_canvas.yview, width=16)
        self.color_inner = ttk.Frame(self.color_canvas)

        self.color_inner.bind("<Configure>", lambda e: self.color_canvas.configure(scrollregion=self.color_canvas.bbox("all")))
        self.color_window = self.color_canvas.create_window((0, 0), window=self.color_inner, anchor="nw")
        self.color_canvas.configure(yscrollcommand=self.color_scrollbar.set)

        self.color_canvas.pack(side="left", fill="both", expand=True)
        self.color_scrollbar.pack(side="right", fill="y")

        # Bind mouse wheel to scroll when cursor is over the canvas
        def _color_on_mousewheel(event):
            try:
                delta = int(-1 * (event.delta / 120))
            except Exception:
                delta = -1
            self.color_canvas.yview_scroll(delta, "units")
        self.color_canvas.bind_all("<MouseWheel>", _color_on_mousewheel)

        # ROI Preview
        roi_prev_frame = ttk.LabelFrame(self.color_inner, text="Selected ROI")
        roi_prev_frame.pack(fill=tk.X, pady=6)
        self.roi_preview_size = 200
        self.roi_preview_label = tk.Label(roi_prev_frame, width=self.roi_preview_size, height=self.roi_preview_size, bg="#f0f0f0")
        self.roi_preview_label.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        # Color space
        cs_frame = ttk.LabelFrame(self.color_inner, text="Color Space")
        cs_frame.pack(fill=tk.X, pady=6)
        self.color_space_var = tk.IntVar(value=1)  # 1 HSV, 0 BGR
        ttk.Radiobutton(cs_frame, text="HSV", variable=self.color_space_var, value=1, command=self._on_color_space_change).pack(anchor=tk.W)
        ttk.Radiobutton(cs_frame, text="BGR", variable=self.color_space_var, value=0, command=self._on_color_space_change).pack(anchor=tk.W)

        # Sliders
        sliders = ttk.LabelFrame(self.color_inner, text="Detection Settings")
        sliders.pack(fill=tk.X, pady=6)

        self.hue_label = ttk.Label(sliders, text="Hue tol")
        self.hue_label.pack(anchor=tk.W, padx=6)
        self.hue_slider = tk.Scale(sliders, from_=0, to=90, orient=tk.HORIZONTAL, length=320, command=lambda v: self._on_color_slider_change())
        self.hue_slider.set(self.tol_hue)
        self.hue_slider.pack(fill=tk.X)

        self.sat_label = ttk.Label(sliders, text="Sat tol")
        self.sat_label.pack(anchor=tk.W, padx=6)
        self.sat_slider = tk.Scale(sliders, from_=0, to=127, orient=tk.HORIZONTAL, length=320, command=lambda v: self._on_color_slider_change())
        self.sat_slider.set(self.tol_sat)
        self.sat_slider.pack(fill=tk.X)

        self.val_label = ttk.Label(sliders, text="Val tol")
        self.val_label.pack(anchor=tk.W, padx=6)
        self.val_slider = tk.Scale(sliders, from_=0, to=127, orient=tk.HORIZONTAL, length=320, command=lambda v: self._on_color_slider_change())
        self.val_slider.set(self.tol_val)
        self.val_slider.pack(fill=tk.X)

        ttk.Label(sliders, text="Morph kernel").pack(anchor=tk.W, padx=6)
        self.kernel_slider = tk.Scale(sliders, from_=0, to=25, orient=tk.HORIZONTAL, length=320, command=lambda v: self._on_color_slider_change())
        self.kernel_slider.set(self.kernel)
        self.kernel_slider.pack(fill=tk.X)

        ttk.Label(sliders, text="Min area").pack(anchor=tk.W, padx=6)
        self.min_slider = tk.Scale(sliders, from_=1, to=20000, orient=tk.HORIZONTAL, length=320, command=lambda v: self._on_color_slider_change())
        self.min_slider.set(self.min_area)
        self.min_slider.pack(fill=tk.X)

        # Watershed
        ws_frame = ttk.LabelFrame(self.color_inner, text="Segmentation Refinement")
        ws_frame.pack(fill=tk.X, pady=6)
        self.ws_var = tk.IntVar(value=0)
        ttk.Checkbutton(ws_frame, text="Split touching objects (watershed)", variable=self.ws_var, command=self._on_ws_toggle).pack(anchor=tk.W)
        ttk.Label(ws_frame, text="Sensitivity").pack(anchor=tk.W, padx=6)
        self.ws_slider = tk.Scale(ws_frame, from_=1, to=80, orient=tk.HORIZONTAL, length=320, command=lambda v: self._on_ws_slider_change())
        self.ws_slider.set(self.ws_sensitivity)
        self.ws_slider.pack(fill=tk.X)



        # Instructions
        instruction_frame = ttk.LabelFrame(left_panel, text="Instructions")
        instruction_frame.pack(fill=tk.X, pady=10)
        instructions = """
1. Load an image 
2. Choose ROI method (auto detection or drag rectangle)
3. Optionally adjust detection flexibility (rotation/scale)
4. Click \"Count Objects\" to analyze
5. Save annotated result if needed
        """
        ttk.Label(instruction_frame, text=instructions, justify=tk.LEFT).pack(padx=10, pady=8)

        # Results Display
        results_frame = ttk.LabelFrame(left_panel, text="Results")
        results_frame.pack(fill=tk.BOTH, pady=10, expand=False)
        self.results_text = tk.Text(results_frame, height=12, width=36)
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Image canvas
        self.canvas = tk.Canvas(right_panel, bg="white")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)

        self.clear_results()

        # Initialize visibility per mode
        self._apply_mode_visibility()
        # Ensure default color slider visibility and values
        self._on_color_space_change()

    # ----------------------- Helpers -----------------------
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

    def rotate_image(self, img, angle):
        # rotate a BGR numpy array about its center, keep same size
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

    def preprocess_color(self, bgr_img):
        """Convert BGR to HSV for color-based detection."""
        hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
        return hsv

    def validate_detection(self, img_gray, roi_gray, x, y, w, h, roi_area, is_uniform=False):
        """Enhanced validation for detected objects using multiple criteria."""
        if x < 0 or y < 0 or x + w > img_gray.shape[1] or y + h > img_gray.shape[0]:
            return False
        
        # Area ratio validation - more lenient for uniform objects and expanded range for all objects
        area_ratio = (w * h) / roi_area
        if is_uniform:
            if area_ratio < 0.15 or area_ratio > 7.0:  # Much more flexible for uniform objects
                return False
        else:
            if area_ratio < 0.2 or area_ratio > 6.0:  # Expanded range for all objects
                return False
        
        # Edge-based validation
        img_edges = cv2.Canny(img_gray, 50, 150)
        roi_edges_base = cv2.Canny(roi_gray, 50, 150)
        roi_edges = cv2.resize(roi_edges_base, (w, h))
        patch_edges = img_edges[y:y+h, x:x+w]
        
        if patch_edges.size == 0:
            return False
            
        edge_corr = cv2.matchTemplate(patch_edges, roi_edges, cv2.TM_CCOEFF_NORMED)
        edge_score = float(edge_corr.max()) if edge_corr.size > 0 else 0.0
        
        # Adaptive edge threshold based on area ratio and uniformity
        if is_uniform:
            min_edge_score = max(0.20, 0.40 - (area_ratio - 1.0) * 0.06)  # Lower threshold for uniform objects
        else:
            min_edge_score = max(0.25, 0.45 - (area_ratio - 1.0) * 0.08)
        
        if edge_score < min_edge_score:
            return False
        
        # Additional texture validation - more lenient for uniform objects
        patch = img_gray[y:y+h, x:x+w]
        if patch.size == 0:
            return False
            
        # Check if the patch has sufficient texture (not too uniform)
        patch_std = np.std(patch)
        if is_uniform:
            if patch_std < 8:  # More lenient for uniform objects
                return False
        else:
            if patch_std < 10:  # Original threshold for distinct objects
                return False
        
        return True

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

        self.original_image = img_bgr.copy()       # keep BGR for processing
        # Also prepare grayscale immediately for downstream processing
        try:
            self.original_gray = self.preprocess_gray(self.original_image)
        except Exception:
            self.original_gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        self.image = cv2.cvtColor(img_bgr.copy(), cv2.COLOR_BGR2RGB)  # RGB for display
        self.display_image = self.image.copy()

        # Reset to default settings for new image
        self.roi_method.set("auto")
        self.detection_method.set("grayscale")
        self.detect_rotated.set(True)
        self.detect_scaled.set(True)
        try:
            self.detect_scaled_safe.set(False)
        except Exception:
            pass
        
        # Clear previous state
        self.selected_roi = None
        self.detected_objects = []
        self.object_highlighted = False
        self.clear_roi()
        self.display_image_on_canvas()
        self.clear_results()
        
        # Auto-detect objects with YOLO if in template mode
        if self.app_mode.get() == "template":
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
        resized = cv2.resize(self.display_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
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
        
        # In template mode, allow auto YOLO selection
        if self.app_mode.get() == "template" and self.roi_method.get() == "auto" and self.object_highlighted and self.detected_objects:
            canvas_x = (event.x - self.canvas_offset_x) / self.scale_factor
            canvas_y = (event.y - self.canvas_offset_y) / self.scale_factor
            
            for i, obj in enumerate(self.detected_objects):
                x, y, w, h, _, _ = self._norm_box(obj)
                if x <= canvas_x <= x + w and y <= canvas_y <= y + h:
                    self.selected_roi = (x, y, x + w, y + h)
                    self.highlight_selected_object(i)
                    return
        
        # Fall back to rectangle selection
        if self.roi_method.get() == "rectangle":
            self.rect_start = (event.x, event.y)
            self.drawing = True
            self.canvas.delete("selection_rect")

    def on_mouse_drag(self, event):
        if not getattr(self, "drawing", False) or self.roi_method.get() != "rectangle":
            return
        self.rect_end = (event.x, event.y)
        self.canvas.delete("selection_rect")
        x1, y1 = self.rect_start
        x2, y2 = self.rect_end
        self.canvas.create_rectangle(x1, y1, x2, y2, outline="red", width=2, tags="selection_rect")

    def on_mouse_up(self, event):
        if not getattr(self, "drawing", False) or self.roi_method.get() != "rectangle":
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
                # Update ROI preview for color mode
                if self.app_mode.get() == "color":
                    self._update_roi_preview()
                    self._update_color_preview()

    def clear_roi(self):
        """Clear the current ROI selection"""
        self.selected_roi = None
        self.canvas.delete("selection_rect")
        if hasattr(self, 'drawing'):
            self.drawing = False
        if hasattr(self, 'rect_start'):
            self.rect_start = None
        if hasattr(self, 'rect_end'):
            self.rect_end = None

    def on_roi_method_change(self):
        """Handle ROI method change between auto detection and rectangle selection"""
        # Force rectangle mode for color segmentation
        if self.app_mode.get() == "color":
            self.roi_method.set("rectangle")
        if self.roi_method.get() == "auto":
            # Switch to auto detection - run YOLO detection
            if self.original_image is not None:
                self.clear_roi()
                if self.app_mode.get() == "template":
                    self.auto_detect_objects()
        else:
            # Switch to rectangle selection - clear YOLO detections
            self.detected_objects = []
            self.object_highlighted = False
            self.clear_roi()
            if self.original_image is not None:
                # Reset to original image without any color space conversion issues
                self.display_image = cv2.cvtColor(self.original_image.copy(), cv2.COLOR_BGR2RGB)
                self.display_image_on_canvas()

    def auto_detect_objects(self):
        """Automatically detect objects using YOLO or classical methods"""
        try:
            if self.app_mode.get() != "template":
                return
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
                # Keep only reasonably confident boxes - optimized for butterfly detection
                self.detected_objects = [d for d in yolo_boxes if d['conf'] >= 0.08]  # Lowered to 0.08 for better detection
                print(f"After confidence filtering: {len(self.detected_objects)} objects")
                
                # Debug: Show all detections for troubleshooting
                for i, obj in enumerate(self.detected_objects):
                    print(f"Object {i}: {obj['label']} at ({obj['x']}, {obj['y']}) with conf {obj['conf']:.3f}")
                
                if len(self.detected_objects) == 0:
                    print("YOLO found no confident objects, falling back to classical method")
                    # Fall back to classical pipeline if YOLO found nothing
                    pass
                else:
                    self.highlight_detected_objects()
                    self.object_highlighted = True
                    messagebox.showinfo("Info", f"Found {len(self.detected_objects)} objects using YOLO (confidence ≥0.08). Click one to select as reference. Only objects of the same class will be counted.")
                    return

            # Fall back to classical detection if YOLO fails
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
            messagebox.showinfo("Info", f"Found {len(self.detected_objects)} objects using classical detection. Click one to select as reference.")

        except Exception as e:
            messagebox.showerror("Error", f"Error in auto-detection: {str(e)}")


    def highlight_detected_objects(self):
        """Highlight all detected objects on the image"""
        if not self.detected_objects:
            return
        
        highlight_image = self.original_image.copy()
        
        for i, obj in enumerate(self.detected_objects):
            x, y, w, h, label, conf = self._norm_box(obj)
            color = (255, 0, 255)  # Magenta for detected objects
            cv2.rectangle(highlight_image, (x, y), (x+w, y+h), color, 2)
            name = label if label else f"Object {i+1}"
            cv2.putText(highlight_image, name, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        self.display_image = cv2.cvtColor(highlight_image, cv2.COLOR_BGR2RGB)
        self.display_image_on_canvas()

    def highlight_selected_object(self, selected_index):
        """Highlight the selected object in green, others in magenta"""
        if not self.detected_objects:
            return
        
        highlight_image = self.original_image.copy()
        
        for i, obj in enumerate(self.detected_objects):
            x, y, w, h, label, conf = self._norm_box(obj)
            if i == selected_index:
                color = (0, 255, 0)  # Green for selected object
                thickness = 3
                ref_text = f"Reference: {label}" if label else f"Reference (Object {i+1})"
                cv2.putText(highlight_image, ref_text, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            else:
                color = (255, 0, 255)  # Magenta for other objects
                thickness = 2
                name = label if label else f"Object {i+1}"
                cv2.putText(highlight_image, name, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            cv2.rectangle(highlight_image, (x, y), (x+w, y+h), color, thickness)
        
        self.display_image = cv2.cvtColor(highlight_image, cv2.COLOR_BGR2RGB)
        self.display_image_on_canvas()

    # ----------------------- Detection -----------------------
    def count_objects(self):
        if self.original_image is None:
            messagebox.showwarning("Warning", "Please load an image first")
            return
        # Route to active mode
        if self.app_mode.get() == "color":
            self._count_objects_color()
            return
        # template flow from here
        if self.selected_roi is None:
            messagebox.showwarning("Warning", "Please select a reference ROI")
            return

        x1, y1, x2, y2 = self.selected_roi
        roi_bgr = self.original_image[y1:y2, x1:x2]
        if roi_bgr.size == 0:
            messagebox.showwarning("Warning", "Invalid ROI")
            return

        # Get detection method
        detection_method = self.detection_method.get()
        
        # If using auto detection with YOLO, use YOLO results for counting
        if self.roi_method.get() == "auto" and self.detected_objects and self.object_highlighted:
            self.count_objects_with_yolo()
            return
                
        # Check which detection methods are enabled
        detect_rotated = self.detect_rotated.get()
        detect_scaled = self.detect_scaled.get()
        
        # Apply different logic based on checkbox selections
        if detect_rotated and detect_scaled:
            self.count_objects_combined_rotation_scaling(roi_bgr, detection_method)
            return
        elif detect_rotated:
             self.count_objects_rotation_only(roi_bgr, detection_method)
             return
        elif detect_scaled:
             self.count_objects_scaling_only(roi_bgr, detection_method)
             return
        else:
            self.count_objects_basic(roi_bgr, detection_method)
            return

    def count_objects_with_yolo(self):
        """Count objects using YOLO detection results"""
        if not self.detected_objects or not self.selected_roi:
            return
        
        try:
            # Get reference object class from selected ROI
            ref_x1, ref_y1, ref_x2, ref_y2 = self.selected_roi
            ref_label = ''
            
            # Find the YOLO object that overlaps with the selected ROI
            best_overlap = 0
            best_obj = None
            
            for obj in self.detected_objects:
                x, y, w, h, label, conf = self._norm_box(obj)
                # Calculate overlap with selected ROI
                overlap_x1 = max(x, ref_x1)
                overlap_y1 = max(y, ref_y1)
                overlap_x2 = min(x + w, ref_x2)
                overlap_y2 = min(y + h, ref_y2)
                
                if overlap_x2 > overlap_x1 and overlap_y2 > overlap_y1:
                    overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
                    roi_area = (ref_x2 - ref_x1) * (ref_y2 - ref_y1)
                    overlap_ratio = overlap_area / roi_area if roi_area > 0 else 0
                    
                    if overlap_ratio > best_overlap:
                        best_overlap = overlap_ratio
                        best_obj = obj
            
            if best_obj is None or best_overlap < 0.1:
                messagebox.showwarning("Warning", "Selected ROI doesn't overlap with any detected object")
                return
            
            ref_label = best_obj.get('label', '') if isinstance(best_obj, dict) else best_obj[4] if len(best_obj) > 4 else ''
            
            # Filter objects by the same class
            same_class_objects = []
            for obj in self.detected_objects:
                x, y, w, h, label, conf = self._norm_box(obj)
                if label == ref_label and conf >= 0.25:  # Same class and confident
                    same_class_objects.append((x, y, w, h, label, conf))
            
            if not same_class_objects:
                messagebox.showwarning("Warning", f"No confident objects found for class '{ref_label}'")
                return
            
            # Calculate reference area
            ref_area = (ref_x2 - ref_x1) * (ref_y2 - ref_y1)
            
            # Augment YOLO with ORB based on selected ROI to catch misses
            orb_rects = []
            try:
                roi_sel = self.original_image[ref_y1:ref_y2, ref_x1:ref_x2]
                orb = cv2.ORB_create(nfeatures=1500)
                kp_roi, des_roi = orb.detectAndCompute(roi_sel, None)
                img_gray_orb = self.preprocess_gray(self.original_image)
                kp_img, des_img = orb.detectAndCompute(img_gray_orb, None)
                if des_roi is not None and des_img is not None and kp_roi is not None and len(kp_roi) >= 20:
                    index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=2)
                    search_params = dict(checks=80)
                    flann = cv2.FlannBasedMatcher(index_params, search_params)
                    try:
                        matches_knn = flann.knnMatch(des_roi, des_img, k=2)
                    except Exception:
                        matches_knn = []
                    def rects_for_ratio(ratio):
                        good_local = []
                        for m_n in matches_knn:
                            if len(m_n) != 2:
                                continue
                            m, n = m_n
                            if m.distance < ratio * n.distance:
                                good_local.append(m)
                        rects = []
                        if len(good_local) >= 8:
                            dst_pts = np.float32([kp_img[m.trainIdx].pt for m in good_local])
                            roi_diag = float(np.hypot(roi_sel.shape[1], roi_sel.shape[0]))
                            eps = max(8.0, roi_diag * 0.20)
                            clustering = DBSCAN(eps=eps, min_samples=6).fit(dst_pts)
                            labels = clustering.labels_
                            uniq = set(labels)
                            if -1 in uniq:
                                uniq.remove(-1)
                            for lbl in uniq:
                                inds = [i for i, lab in enumerate(labels) if lab == lbl]
                                if len(inds) < 6:
                                    continue
                                cluster_matches = [good_local[i] for i in inds]
                                src_pts = np.float32([kp_roi[m.queryIdx].pt for m in cluster_matches]).reshape(-1, 1, 2)
                                dst_pts_cluster = np.float32([kp_img[m.trainIdx].pt for m in cluster_matches]).reshape(-1, 1, 2)
                                M, mask = cv2.findHomography(src_pts, dst_pts_cluster, cv2.RANSAC, 3.0)
                                if M is None or mask is None:
                                    continue
                                inliers = int(mask.ravel().sum())
                                if inliers < 12:
                                    continue
                                h_roi, w_roi = roi_sel.shape[:2]
                                pts = np.float32([[0,0],[w_roi,0],[w_roi,h_roi],[0,h_roi]]).reshape(-1,1,2)
                                dst = cv2.perspectiveTransform(pts, M)
                                x, y, w, h = cv2.boundingRect(np.int32(dst))
                                rects.append((x, y, w, h))
                        return rects
                    best = []
                    for r in [0.60,0.65,0.70,0.75,0.80]:
                        rects = rects_for_ratio(r)
                        if len(rects) > len(best):
                            best = rects
                    orb_rects = self.non_max_suppression(best, overlapThresh=0.45) if best else []
            except Exception:
                orb_rects = []

            # Merge YOLO class-filtered boxes with ORB boxes
            merged = [(x, y, w, h) for (x, y, w, h, _, _) in same_class_objects]
            merged.extend(orb_rects)
            if merged:
                merged = self.non_max_suppression(merged, overlapThresh=0.45)

            # Display results
            results_info = f"Method used: YOLO + ORB Augmentation\n"
            results_info += f"Detection: {self.detection_method.get()}\n"
            results_info += f"Rotation: {'Enabled' if self.detect_rotated.get() else 'Disabled'}\n"
            results_info += f"Scaling: {'Enabled' if self.detect_scaled.get() else 'Disabled'}\n"
            results_info += f"Reference Class: {ref_label}\n"
            results_info += f"Reference Area: {ref_area:.1f} pixels\n\n"
            results_info += f"YOLO matches: {len(same_class_objects)} | ORB matches: {len(orb_rects)} | Merged: {len(merged)}\n"

            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, results_info)

            vis = self.original_image.copy()
            for (x, y, w, h) in merged:
                cv2.rectangle(vis, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.rectangle(vis, (ref_x1, ref_y1), (ref_x2, ref_y2), (0,255,0), 2)
            self.display_image = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
            self.display_image_on_canvas()
            
        except Exception as e:
            messagebox.showerror("Error", f"Error counting objects with YOLO: {str(e)}")

    def count_objects_rotation_only(self, roi_bgr, detection_method):
        """Apply only rotation detection logic (from apply_rotation.py)"""
        try:
            # Check if ROI is too small (blank selection)
            if roi_bgr.shape[0] < 5 or roi_bgr.shape[1] < 5:
                self.results_text.delete(1.0, tk.END)
                self.results_text.insert(tk.END, "No objects found in the image.")
                # Update display to show original image without highlighting
                self.display_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
                self.display_image_on_canvas()
                return
                
            # Convert original image and ROI based on detection method
            if detection_method == "grayscale":
                img_gray = self.preprocess_gray(self.original_image)
                roi_gray = self.preprocess_gray(roi_bgr)
            else:  # color matching
                img_hsv = self.preprocess_color(self.original_image)
                roi_hsv = self.preprocess_color(roi_bgr)
                img_gray = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
                roi_gray = cv2.cvtColor(roi_hsv, cv2.COLOR_HSV2BGR)
                img_gray = cv2.cvtColor(img_gray, cv2.COLOR_BGR2GRAY)
                roi_gray = cv2.cvtColor(roi_hsv, cv2.COLOR_HSV2BGR)
                
            # Decide method: try ORB first, fallback to template matching
            orb = cv2.ORB_create(nfeatures=1000)
            kp_roi, des_roi = orb.detectAndCompute(roi_bgr, None)
            method_used = "ORB" if (kp_roi is not None and len(kp_roi) >= 20 and des_roi is not None) else "Template Matching"
            
            results_info = ""
            vis_bgr = self.original_image.copy()  # draw on BGR copy
            
            if method_used == "Template Matching":
                # Convert original image and ROI to enhanced grayscale
                img_gray = self.preprocess_gray(self.original_image)
                roi_gray = self.preprocess_gray(roi_bgr)

                # Template Matching section in count_objects method
                rectangles = []
                angles = range(0, 360, 8)  # Use 8-degree increments for finer rotation detection
                scales = [0.9, 1.0, 1.1]  

                used_thresholds = []
                for angle in angles:
                    rotated_roi = self.rotate_image(roi_bgr, angle)  # rotate ROI in BGR
                    rotated_gray = self.preprocess_gray(rotated_roi)  # convert rotated ROI to gray

                    for scale in scales:
                        scaled_w = int(rotated_gray.shape[1] * scale)
                        scaled_h = int(rotated_gray.shape[0] * scale)
                        if scaled_w < 10 or scaled_h < 10:
                            continue

                        scaled_roi_gray = cv2.resize(rotated_gray, (scaled_w, scaled_h))
                        res = cv2.matchTemplate(img_gray, scaled_roi_gray, cv2.TM_CCOEFF_NORMED)
                        
                        # Check if we're likely dealing with uniform objects
                        res_mean = np.mean(res)
                        res_std = np.std(res)
                        is_uniform = res_mean > 0.15 and res_std < 0.2
                        
                        auto_thr = self.compute_auto_template_threshold(res)
                        
                        # For uniform objects, be more aggressive
                        if is_uniform:
                            auto_thr = max(0.35, auto_thr - 0.12)  # Even lower threshold for better detection
                        
                        self.template_threshold = auto_thr
                        used_thresholds.append(auto_thr)
                        
                        # Local maxima to avoid clusters of overlapping candidates
                        # Use smaller kernel for uniform objects to catch more instances
                        kh = max(2, int(round(scaled_h * 0.06))) if is_uniform else max(3, int(round(scaled_h * 0.08)))
                        kw = max(2, int(round(scaled_w * 0.06))) if is_uniform else max(3, int(round(scaled_w * 0.08)))
                        kernel = np.ones((kh, kw), np.uint8)
                        res_dil = cv2.dilate(res, kernel)
                        maxima = (res >= auto_thr) & (res == res_dil)
                        ys, xs = np.where(maxima)
                        for y, x in zip(ys, xs):
                            rectangles.append([x, y, scaled_w, scaled_h])
                    
                    # Also try horizontally flipped version to detect left-right mirrored objects
                    flipped_roi = cv2.flip(rotated_roi, 1)  # 1 = horizontal flip
                    flipped_gray = self.preprocess_gray(flipped_roi)
                    
                    for scale in scales:
                        scaled_w = int(flipped_gray.shape[1] * scale)
                        scaled_h = int(flipped_gray.shape[0] * scale)
                        if scaled_w < 10 or scaled_h < 10:
                            continue

                        scaled_flipped_gray = cv2.resize(flipped_gray, (scaled_w, scaled_h))
                        res_flipped = cv2.matchTemplate(img_gray, scaled_flipped_gray, cv2.TM_CCOEFF_NORMED)
                        
                        # Use same threshold calculation
                        auto_thr_flipped = self.compute_auto_template_threshold(res_flipped)
                        
                        # For uniform objects, be much more aggressive
                        if is_uniform:
                            auto_thr_flipped = max(0.35, auto_thr_flipped - 0.12)
                        
                        used_thresholds.append(auto_thr_flipped)
                        
                        # Local maxima for flipped version
                        kh = max(2, int(round(scaled_h * 0.06))) if is_uniform else max(3, int(round(scaled_h * 0.08)))
                        kw = max(2, int(round(scaled_w * 0.06))) if is_uniform else max(3, int(round(scaled_w * 0.08)))
                        kernel = np.ones((kh, kw), np.uint8)
                        res_dil = cv2.dilate(res_flipped, kernel)
                        maxima = (res_flipped >= auto_thr_flipped) & (res_flipped == res_dil)
                        ys, xs = np.where(maxima)
                        
                        for y, x in zip(ys, xs):
                            rectangles.append([x, y, scaled_w, scaled_h])

                # After collecting rectangles, modify grouping and NMS
                if rectangles:
                    # For uniform objects, use more lenient grouping
                    is_uniform = len(rectangles) > 20  # Many candidates suggest uniform objects
                    
                    if is_uniform:
                        # More lenient grouping for uniform objects
                        rectangles, _ = cv2.groupRectangles(rectangles, groupThreshold=1, eps=0.7)
                        rectangles = self.non_max_suppression(rectangles, overlapThresh=0.45)  # More lenient overlap
                    else:
                        # Original approach for distinct objects
                        rectangles, _ = cv2.groupRectangles(rectangles, groupThreshold=1, eps=0.5)
                        rectangles = self.non_max_suppression(rectangles, overlapThresh=0.3)
                else:
                    rectangles = []

                # Secondary validation using edges and size consistency
                if rectangles:
                    img_edges = cv2.Canny(img_gray, 50, 150)
                    roi_edges_base = cv2.Canny(roi_gray, 50, 150)

                    validated = []
                    roi_area = float(roi_gray.shape[0] * roi_gray.shape[1] + 1e-6)
                    
                    # Check if we're dealing with uniform objects 
                    is_uniform = len(rectangles) > 10
                    
                    for (x, y, w, h) in rectangles:
                        if not self.validate_detection(img_gray, roi_gray, x, y, w, h, roi_area, is_uniform):
                            continue
                                
                        validated.append((x, y, w, h))
                    rectangles = validated

                # If nothing validated, fallback to edge-only template matching with flipping
                if not rectangles:
                    img_edges = cv2.Canny(img_gray, 50, 150)
                    roi_edges_base = cv2.Canny(roi_gray, 50, 150)
                    rectangles_fallback = []
                    fine_angles = range(0, 360, 15)  # Use finer angles for better detection
                    scales_fb = [0.7, 0.85, 1.0, 1.15, 1.3]  # More scale variations
                    
                    for ang in fine_angles:
                        # Process normal rotated edges
                        rot_edges = self.rotate_image(cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB), ang)
                        rot_edges = cv2.cvtColor(rot_edges, cv2.COLOR_RGB2GRAY)
                        rot_edges = cv2.Canny(rot_edges, 50, 150)
                        
                        # Also process flipped version
                        flipped_edges = cv2.flip(rot_edges, 1)
                        
                        for sc in scales_fb:
                            w_fb = int(rot_edges.shape[1] * sc)
                            h_fb = int(rot_edges.shape[0] * sc)
                            if w_fb < 15 or h_fb < 15:
                                continue
                                
                            # Process normal edges
                            tmpl = cv2.resize(rot_edges, (w_fb, h_fb))
                            res = cv2.matchTemplate(img_edges, tmpl, cv2.TM_CCOEFF_NORMED)
                            thr_fb = max(0.35, float(np.percentile(res, 97.5)))  # More aggressive threshold
                            loc = np.where(res >= thr_fb)
                            for pt in zip(*loc[::-1]):
                                rectangles_fallback.append([pt[0], pt[1], w_fb, h_fb])
                            
                            # Process flipped edges
                            tmpl_flipped = cv2.resize(flipped_edges, (w_fb, h_fb))
                            res_flipped = cv2.matchTemplate(img_edges, tmpl_flipped, cv2.TM_CCOEFF_NORMED)
                            loc_flipped = np.where(res_flipped >= thr_fb)
                            for pt in zip(*loc_flipped[::-1]):
                                rectangles_fallback.append([pt[0], pt[1], w_fb, h_fb])
                    
                    if rectangles_fallback:
                        rectangles_fallback = self.non_max_suppression(rectangles_fallback, overlapThresh=0.4)
                    rectangles = rectangles_fallback

                # Draw results
                vis_bgr = self.original_image.copy()
                for (x, y, w, h) in rectangles:
                    cv2.rectangle(vis_bgr, (x, y), (x + w, y + h), (255, 255, 150), 2)

                self.display_image = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)
                self.display_image_on_canvas()
                avg_thr = (np.mean(used_thresholds) if used_thresholds else self.template_threshold)
                results_info = (f"Method used: {method_used}\n"
                                f"Objects Found: {len(rectangles)}\n"
                                f"Auto Threshold: {avg_thr:.2f}\n")

            else:
                # ORB-based multi-instance detection (FLANN + ratio test)
                orb = cv2.ORB_create(nfeatures=1200)
                roi_gray = self.preprocess_gray(roi_bgr)
                img_gray = self.preprocess_gray(self.original_image)
                kp_roi, des_roi = orb.detectAndCompute(roi_gray, None)
                kp_img, des_img = orb.detectAndCompute(img_gray, None)
                if des_roi is None or des_img is None:
                    messagebox.showwarning("Warning", "Not enough features for ORB")
                    return

                # FLANN matcher parameters for ORB (LSH index)
                index_params = dict(algorithm=6,  # FLANN_INDEX_LSH
                                    table_number=6,
                                    key_size=12,
                                    multi_probe_level=2)
                search_params = dict(checks=50)
                flann = cv2.FlannBasedMatcher(index_params, search_params)

                try:
                    matches_knn = flann.knnMatch(des_roi, des_img, k=2)
                except Exception:
                    matches_knn = []

                def evaluate_with_ratio(ratio):
                    good_local = []
                    for m_n in matches_knn:
                        if len(m_n) != 2:
                            continue
                        m, n = m_n
                        if m.distance < ratio * n.distance:
                            good_local.append(m)

                    boxes_local = []
                    objects_local = 0
                    if len(good_local) >= 8:
                        dst_pts_loc = np.float32([kp_img[m.trainIdx].pt for m in good_local])
                        roi_diag = np.hypot(roi_bgr.shape[1], roi_bgr.shape[0])
                        eps = max(10.0, roi_diag * 0.22)
                        clustering_loc = DBSCAN(eps=eps, min_samples=6).fit(dst_pts_loc)
                        labels_loc = clustering_loc.labels_
                        uniq = set(labels_loc)
                        if -1 in uniq:
                            uniq.remove(-1)
                        for lbl in uniq:
                            inds = [i for i, lab in enumerate(labels_loc) if lab == lbl]
                            if len(inds) < 6:
                                continue
                            cluster_matches = [good_local[i] for i in inds]
                            src_pts = np.float32([kp_roi[m.queryIdx].pt for m in cluster_matches]).reshape(-1, 1, 2)
                            dst_pts_cluster = np.float32([kp_img[m.trainIdx].pt for m in cluster_matches]).reshape(-1, 1, 2)
                            M, mask = cv2.findHomography(src_pts, dst_pts_cluster, cv2.RANSAC, 4.0)
                            if M is None or mask is None:
                                continue
                            inliers = int(mask.ravel().sum())
                            if inliers < 12:
                                continue
                            h_roi, w_roi = roi_bgr.shape[:2]
                            pts = np.float32([[0,0],[w_roi,0],[w_roi,h_roi],[0,h_roi]]).reshape(-1,1,2)
                            dst = cv2.perspectiveTransform(pts, M)
                            x, y, w, h = cv2.boundingRect(np.int32(dst))
                            roi_area = float(w_roi * h_roi + 1e-6)
                            scale_sq = (w * h) / roi_area
                            if scale_sq < 0.30 or scale_sq > 3.8:
                                continue
                            aspect_roi = w_roi / (h_roi + 1e-6)
                            aspect_box = w / (h + 1e-6)
                            if abs(aspect_box - aspect_roi) / aspect_roi > 0.65:
                                continue
                            boxes_local.append([x, y, w, h])
                            objects_local += 1
                    if boxes_local:
                        boxes_local = self.non_max_suppression(boxes_local, overlapThresh=0.45)

                    objects_local = len(boxes_local)
                    return objects_local, boxes_local, good_local

                # Sweep several ratio candidates and keep the best outcome
                candidate_ratios = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
                best_score = -1
                best = (0, [], [], 0.75)
                for r in candidate_ratios:
                    objs, boxes, good_local = evaluate_with_ratio(r)
                    # score favors more objects and fewer overlaps (via boxes length)
                    score = objs * 1000 + len(good_local)
                    if score > best_score:
                        best_score = score
                        best = (objs, boxes, good_local, r)

                objects_found, accepted_boxes, good, chosen_ratio = best
                # record chosen ratio
                self.orb_ratio = float(chosen_ratio)

                for (x, y, w, h) in accepted_boxes:
                    cv2.rectangle(vis_bgr, (x, y), (x+w, y+h), (0,255,0), 2)

                self.display_image = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)
                self.display_image_on_canvas()
                results_info = (f"Method used: ORB (FLANN)\n"
                                f"Keypoints in ROI: {len(kp_roi) if kp_roi is not None else 0}\n"
                                f"Good Matches: {len(good)}\n"
                                f"Objects Found: {objects_found}\n"
                                f"Auto Ratio Threshold: {self.orb_ratio:.2f}\n")

            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, results_info)
        except Exception as e:
            messagebox.showerror("Error", f"No matches found with selected rotation settings.")

    def count_objects_scaling_only(self, roi_bgr, detection_method):
        """Apply only scaling detection logic"""
        try:
            # Convert original image and ROI based on detection method
            if detection_method == "grayscale":
                img_gray = self.preprocess_gray(self.original_image)
                roi_gray = self.preprocess_gray(roi_bgr)
            else:
                 raise ValueError("Unsupported detection method for this function.")

            all_rectangles = []
            used_thresholds = []

            # Use a comprehensive range of scales
            scales = np.linspace(0.6, 1.4, 25) # Increased number of steps for better coverage

            for scale in scales:
                scaled_w = int(roi_gray.shape[1] * scale)
                scaled_h = int(roi_gray.shape[0] * scale)
                
                # Skip if the scaled ROI is too small or too large
                if scaled_w < 10 or scaled_h < 10 or scaled_w > img_gray.shape[1] or scaled_h > img_gray.shape[0]:
                    continue
                    
                scaled_roi_gray = cv2.resize(roi_gray, (scaled_w, scaled_h))
                res = cv2.matchTemplate(img_gray, scaled_roi_gray, cv2.TM_CCOEFF_NORMED)
                
                # Use robust auto threshold, but with a minimum value
                auto_thr = self.compute_auto_template_threshold(res)
                final_thr = max(0.5, auto_thr) # Ensure threshold is not too high
                used_thresholds.append(final_thr)

                # Find all locations above the dynamic threshold
                loc = np.where(res >= final_thr)
                for pt in zip(*loc[::-1]):
                    x, y = pt[0], pt[1]
                    all_rectangles.append([x, y, scaled_w, scaled_h])

            # Apply Non-Maximum Suppression (NMS) on all combined rectangles
            if all_rectangles:
                rectangles = self.non_max_suppression(all_rectangles, overlapThresh=0.3)
            else:
                rectangles = []

            # Display results
            vis_bgr = self.original_image.copy()
            for (x, y, w, h) in rectangles:
                cv2.rectangle(vis_bgr, (x, y), (x + w, y + h), (255, 255, 150), 2)

            self.display_image = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)
            self.display_image_on_canvas()
            
            avg_thr = (np.mean(used_thresholds) if used_thresholds else 0.5)
            results_info = (f"Method used: Template Matching\n"
                            f"Detection: {detection_method}\n"
                            f"Rotation: Disabled\n"
                            f"Scaling: Enabled (0.6x to 1.4x)\n"
                            f"Objects Found: {len(rectangles)}\n"
                            f"Average Threshold: {avg_thr:.2f}\n")
            
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, results_info)
            
        except Exception as e:
            messagebox.showerror("Error", f"No matches found with selected scaling settings.")


    def count_objects_combined_rotation_scaling(self, roi_bgr, detection_method):
        try:
            # Check if ROI is too small (blank selection)
            if roi_bgr.shape[0] < 5 or roi_bgr.shape[1] < 5:
                self.results_text.delete(1.0, tk.END)
                self.results_text.insert(tk.END, "No objects found in the image.")
                # Update display to show original image without highlighting
                self.display_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
                self.display_image_on_canvas()
                return
                
            # Convert original image and ROI based on detection method
            if detection_method == "grayscale":
                img_gray = self.preprocess_gray(self.original_image)
                roi_gray = self.preprocess_gray(roi_bgr)
            
            # Decide method: try ORB first, fallback to template matching
            orb = cv2.ORB_create(nfeatures=1000)
            kp_roi, des_roi = orb.detectAndCompute(roi_bgr, None)
            method_used = "ORB" if (kp_roi is not None and len(kp_roi) >= 20 and des_roi is not None) else "Template Matching"
            
            results_info = ""
            vis_bgr = self.original_image.copy()  # draw on BGR copy

            # Process based on the automatically selected method
            if method_used == "ORB":
                # Process with ORB feature matching
                try:
                    img_gray_orb = self.preprocess_gray(self.original_image)
                    kp_img, des_img = orb.detectAndCompute(img_gray_orb, None)
                    
                    index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=2)
                    search_params = dict(checks=80)
                    flann = cv2.FlannBasedMatcher(index_params, search_params)
                    matches_knn = flann.knnMatch(des_roi, des_img, k=2)
                except Exception:
                    matches_knn = []

                def rects_for_ratio(ratio: float):
                    good_local = []
                    for m_n in matches_knn:
                        if len(m_n) != 2:
                            continue
                        m, n = m_n
                        if m.distance < ratio * n.distance:
                            good_local.append(m)
                    rects = []
                    if len(good_local) >= 8:
                        dst_pts = np.float32([kp_img[m.trainIdx].pt for m in good_local])
                        roi_diag = float(np.hypot(roi_bgr.shape[1], roi_bgr.shape[0]))
                        eps = max(8.0, roi_diag * 0.20)
                        clustering = DBSCAN(eps=eps, min_samples=6).fit(dst_pts)
                        labels = clustering.labels_
                        uniq = set(labels)
                        if -1 in uniq:
                            uniq.remove(-1)
                        for lbl in uniq:
                            inds = [i for i, lab in enumerate(labels) if lab == lbl]
                            if len(inds) < 6:
                                continue
                            cluster_matches = [good_local[i] for i in inds]
                            src_pts = np.float32([kp_roi[m.queryIdx].pt for m in cluster_matches]).reshape(-1, 1, 2)
                            dst_pts_cluster = np.float32([kp_img[m.trainIdx].pt for m in cluster_matches]).reshape(-1, 1, 2)
                            M, mask = cv2.findHomography(src_pts, dst_pts_cluster, cv2.RANSAC, 3.0)
                            if M is None or mask is None:
                                continue
                            inliers = int(mask.ravel().sum())
                            if inliers < 12:
                                continue
                            h_roi, w_roi = roi_bgr.shape[:2]
                            pts = np.float32([[0,0],[w_roi,0],[w_roi,h_roi],[0,h_roi]]).reshape(-1,1,2)
                            dst = cv2.perspectiveTransform(pts, M)
                            x, y, w, h = cv2.boundingRect(np.int32(dst))
                            # sanity checks for scale/aspect
                            roi_area = float(max(1, w_roi * h_roi))
                            scale_sq = (w * h) / roi_area
                            if scale_sq < 0.15 or scale_sq > 7.0:  # Expanded range for better size detection
                                continue
                            aspect_roi = w_roi / (h_roi + 1e-6)
                            aspect_box = w / (h + 1e-6)
                            if abs(aspect_box - aspect_roi) / max(1e-6, aspect_roi) > 0.85:  # More lenient aspect ratio check
                                continue
                            rects.append((x, y, w, h))
                    return rects, len(good_local)

                candidate_ratios = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
                best_rects = []
                best_score = -1
                for r in candidate_ratios:
                    rects, good_count = rects_for_ratio(r)
                    score = len(rects) * 1000 + good_count
                    if score > best_score:
                        best_score = score
                        best_rects = rects

                if best_rects:
                    best_rects = self.non_max_suppression(best_rects, overlapThresh=0.45)
                    vis_bgr = self.original_image.copy()
                    for (x, y, w, h) in best_rects:
                        cv2.rectangle(vis_bgr, (x, y), (x + w, y + h), (0,255,0), 2)
                    self.display_image = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)
                    self.display_image_on_canvas()
                    results_info = (f"Method used: ORB (FLANN)\n"
                                    f"Detection: {detection_method}\n"
                                    f"Rotation: {'Enabled' if self.detect_rotated.get() else 'Disabled'}\n"
                                    f"Scaling: {'Enabled' if self.detect_scaled.get() else 'Disabled'}\n"
                                    f"Objects Found: {len(best_rects)}\n")
                    self.results_text.delete(1.0, tk.END)
                    self.results_text.insert(tk.END, results_info)
                    return

            # Template Matching with BOTH rotation AND scaling
            elif method_used == "Template Matching":
                rectangles = []
                used_thresholds = []
                
                # Configure rotation and scaling based on user settings
                if self.detect_rotated.get():
                    angles = range(0, 360, 8)
                else:
                    angles = [0]  # No rotation detection
                    
                if self.detect_scaled.get() or self.detect_scaled_safe.get():
                    scales = [0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.7, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
                else:
                    scales = [1.0]  # No scaling detection
            
            for angle in angles:
                if angle != 0:
                    rotated_roi = self.rotate_image(roi_bgr, angle)
                    rotated_gray = cv2.cvtColor(rotated_roi, cv2.COLOR_BGR2GRAY)
                else:
                    rotated_gray = roi_gray

                for scale in scales:
                    scaled_w = int(rotated_gray.shape[1] * scale)
                    scaled_h = int(rotated_gray.shape[0] * scale)
                    if scaled_w < 10 or scaled_h < 10:
                        continue
                    
                    scaled_roi_gray = cv2.resize(rotated_gray, (scaled_w, scaled_h))
                    res = cv2.matchTemplate(img_gray, scaled_roi_gray, cv2.TM_CCOEFF_NORMED)
                    
                    # Check if we're likely dealing with uniform objects
                    res_mean = np.mean(res)
                    res_std = np.std(res)
                    is_uniform = res_mean > 0.15 and res_std < 0.2
                    
                    # Use robust map-based threshold
                    auto_thr = self.compute_auto_template_threshold(res)
                    
                    # For uniform objects, be more aggressive
                    if is_uniform:
                        auto_thr = max(0.35, auto_thr - 0.12)
                    
                    used_thresholds.append(auto_thr)
                    
                    # Local maxima to avoid clusters
                    kh = max(2, int(round(scaled_h * 0.06))) if is_uniform else max(3, int(round(scaled_h * 0.08)))
                    kw = max(2, int(round(scaled_w * 0.06))) if is_uniform else max(3, int(round(scaled_w * 0.08)))
                    kernel = np.ones((kh, kw), np.uint8)
                    res_dil = cv2.dilate(res, kernel)
                    maxima = (res >= auto_thr) & (res == res_dil)
                    ys, xs = np.where(maxima)
                    for y, x in zip(ys, xs):
                        rectangles.append([x, y, scaled_w, scaled_h])
                
                # Also try horizontally flipped version
                flipped_roi = cv2.flip(rotated_roi if angle != 0 else roi_bgr, 1)
                flipped_gray = cv2.cvtColor(flipped_roi, cv2.COLOR_BGR2GRAY)

                for scale in scales:
                        scaled_w = int(flipped_gray.shape[1] * scale)
                        scaled_h = int(flipped_gray.shape[0] * scale)
                        if scaled_w < 10 or scaled_h < 10:
                            continue
                        
                        scaled_flipped_gray = cv2.resize(flipped_gray, (scaled_w, scaled_h))
                        res_flipped = cv2.matchTemplate(img_gray, scaled_flipped_gray, cv2.TM_CCOEFF_NORMED)
                        
                        auto_thr_flipped = self.compute_auto_template_threshold(res_flipped)
                        if is_uniform:
                            auto_thr_flipped = max(0.35, auto_thr_flipped - 0.12)
                        
                        used_thresholds.append(auto_thr_flipped)
                        
                        kh = max(2, int(round(scaled_h * 0.06))) if is_uniform else max(3, int(round(scaled_h * 0.08)))
                        kw = max(2, int(round(scaled_w * 0.06))) if is_uniform else max(3, int(round(scaled_w * 0.08)))
                        kernel = np.ones((kh, kw), np.uint8)
                        res_dil = cv2.dilate(res_flipped, kernel)
                        maxima = (res_flipped >= auto_thr_flipped) & (res_flipped == res_dil)
                        ys, xs = np.where(maxima)
                        
                        for y, x in zip(ys, xs):
                            rectangles.append([x, y, scaled_w, scaled_h])

            # Group and filter rectangles
            if rectangles:
                is_uniform = len(rectangles) > 20
                
                if is_uniform:
                    rectangles, _ = cv2.groupRectangles(rectangles, groupThreshold=1, eps=0.7)
                    rectangles = self.non_max_suppression(rectangles, overlapThresh=0.45)
                else:
                    rectangles, _ = cv2.groupRectangles(rectangles, groupThreshold=1, eps=0.5)
                    rectangles = self.non_max_suppression(rectangles, overlapThresh=0.3)
            else:
                rectangles = []

            # Secondary validation
            if rectangles:
                img_edges = cv2.Canny(img_gray, 50, 150)
                roi_edges_base = cv2.Canny(roi_gray, 50, 150)

                validated = []
                roi_area = float(roi_gray.shape[0] * roi_gray.shape[1] + 1e-6)
                is_uniform = len(rectangles) > 10
                
                for (x, y, w, h) in rectangles:
                    if not self.validate_detection(img_gray, roi_gray, x, y, w, h, roi_area, is_uniform):
                        continue
                    validated.append((x, y, w, h))
                rectangles = validated

            # Display results
            vis_bgr = self.original_image.copy()
            for (x, y, w, h) in rectangles:
                cv2.rectangle(vis_bgr, (x, y), (x + w, y + h), (255, 255, 150), 2)

            self.display_image = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)
            self.display_image_on_canvas()
            
            avg_thr = (np.mean(used_thresholds) if used_thresholds else 0.7)
            results_info = (f"Method used: {method_used} (Rotation + Scaling)\n"
                            f"Detection: {detection_method}\n"
                            f"Rotation: Enabled (0-360° in 8° increments)\n"
                            f"Scaling: Enabled (0.15x to 5.0x)\n"
                            f"Objects Found: {len(rectangles)}\n"
                            f"Auto Threshold: {avg_thr:.2f}\n"
                            f"Method Selection: Automatic\n")
            
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, results_info)
            
        except Exception as e:
            messagebox.showerror("Error", f"No matches found with selected rotation and scaling settings.")

    def count_objects_basic(self, roi_bgr, detection_method):
        """Apply basic template matching without rotation or scaling"""
        try:
            # Check if ROI is too small (blank selection)
            if roi_bgr.shape[0] < 5 or roi_bgr.shape[1] < 5:
                self.results_text.delete(1.0, tk.END)
                self.results_text.insert(tk.END, "No objects found in the image.")
                # Update display to show original image without highlighting
                self.display_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
                self.display_image_on_canvas()
                return
                
            # Convert original image and ROI based on detection method
            if detection_method == "grayscale":
                img_gray = self.preprocess_gray(self.original_image)
                roi_gray = self.preprocess_gray(roi_bgr)

            # Basic template matching (no rotation, no scaling)
            rectangles = []
            used_thresholds = []
            
            # No rotation, but improved scaling range
            angles = [0]
            scales = [0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.7, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
            
            for angle in angles:
                rotated_gray = roi_gray  # No rotation
                
                for scale in scales:
                    scaled_w = int(rotated_gray.shape[1] * scale)
                    scaled_h = int(rotated_gray.shape[0] * scale)
                    
                    scaled_roi_gray = cv2.resize(rotated_gray, (scaled_w, scaled_h))
                    res = cv2.matchTemplate(img_gray, scaled_roi_gray, cv2.TM_CCOEFF_NORMED)
                    
                    # Use robust map-based threshold
                    auto_thr = self.compute_auto_template_threshold(res)
                    used_thresholds.append(auto_thr)
                    
                    # Local maxima to avoid clusters
                    kh = max(3, int(round(scaled_h * 0.08)))
                    kw = max(3, int(round(scaled_w * 0.08)))
                    kernel = np.ones((kh, kw), np.uint8)
                    res_dil = cv2.dilate(res, kernel)
                    maxima = (res >= auto_thr) & (res == res_dil)
                    ys, xs = np.where(maxima)
                    for y, x in zip(ys, xs):
                        rectangles.append([x, y, scaled_w, scaled_h])

            # Group and filter rectangles
            if rectangles:
                rectangles, _ = cv2.groupRectangles(rectangles, groupThreshold=1, eps=0.5)
                rectangles = self.non_max_suppression(rectangles, overlapThresh=0.3)
            else:
                rectangles = []

            # Display results
            vis_bgr = self.original_image.copy()
            for (x, y, w, h) in rectangles:
                cv2.rectangle(vis_bgr, (x, y), (x + w, y + h), (255, 255, 150), 2)

            self.display_image = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)
            self.display_image_on_canvas()
            
            avg_thr = (np.mean(used_thresholds) if used_thresholds else 0.7)
            results_info = (f"Method used: Template Matching (Basic)\n"
                            f"Detection: {detection_method}\n"
                            f"Rotation: Disabled\n"
                            f"Scaling: Disabled\n"
                            f"Objects Found: {len(rectangles)}\n"
                            f"Auto Threshold: {avg_thr:.2f}\n")
            
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, results_info)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error in basic detection: {str(e)}")

    def draw_yolo_results(self, small_objects, same_size_objects, large_objects):
        """Draw the counting results on the image"""
        result_image = self.original_image.copy()
        
        # Draw reference object
        if self.selected_roi:
            x1, y1, x2, y2 = self.selected_roi
            cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(result_image, "Reference", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Draw small objects
        for x, y, w, h, area in small_objects:
            cv2.rectangle(result_image, (x, y), (x+w, y+h), (0, 0, 255), 1)
            cv2.putText(result_image, "S", (x, y-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
        
        # Draw same size objects
        for x, y, w, h, area in same_size_objects:
            cv2.rectangle(result_image, (x, y), (x+w, y+h), (255, 0, 0), 1)
            cv2.putText(result_image, "M", (x, y-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
        
        # Draw large objects
        for x, y, w, h, area in large_objects:
            cv2.rectangle(result_image, (x, y), (x+w, y+h), (255, 255, 0), 1)
            cv2.putText(result_image, "L", (x, y-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)
        
        # Update display
        self.display_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
        self.display_image_on_canvas()

    def draw_simple_results(self, objects):
        """Draw simple results highlighting all objects of the same class"""
        result_image = self.original_image.copy()
        
        # Draw reference object
        if self.selected_roi:
            x1, y1, x2, y2 = self.selected_roi
            cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(result_image, "Reference", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Draw all detected objects of the same class
        for x, y, w, h, label, conf in objects:
            cv2.rectangle(result_image, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(result_image, f"{label}", (x, y-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        
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
        self.object_highlighted = False
        self.canvas.delete("all")
        self.clear_results()
        # Reset color controls
        if hasattr(self, 'color_space_var'):
            self.color_space_var.set(1)
        if hasattr(self, 'hue_slider'):
            self.hue_slider.set(15)
        if hasattr(self, 'sat_slider'):
            self.sat_slider.set(60)
        if hasattr(self, 'val_slider'):
            self.val_slider.set(60)
        if hasattr(self, 'kernel_slider'):
            self.kernel_slider.set(3)
        if hasattr(self, 'min_slider'):
            self.min_slider.set(100)
        if hasattr(self, 'ws_var'):
            self.ws_var.set(0)
        if hasattr(self, 'ws_slider'):
            self.ws_slider.set(35)

    def clear_results(self):
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "No results yet. Load image and select ROI.\n")

    # ===================== Color Segmentation Utilities =====================
    def _on_mode_switch(self):
        self._apply_mode_visibility()
        # In color mode, force rectangle ROI and disable YOLO auto
        if self.app_mode.get() == "color":
            self.roi_method.set("rectangle")
            self.detected_objects = []
            self.object_highlighted = False
            self.display_image = cv2.cvtColor(self.original_image.copy(), cv2.COLOR_BGR2RGB) if self.original_image is not None else None
            self.display_image_on_canvas()
        else:
            # template mode
            if self.original_image is not None and self.roi_method.get() == "auto":
                self.auto_detect_objects()

    def _apply_mode_visibility(self):
        is_color = self.app_mode.get() == "color"
        # Color section visibility
        try:
            self.color_frame.pack_forget()
        except Exception:
            pass
        if is_color:
            self.color_frame.pack(fill=tk.BOTH, pady=10, expand=True)
        # Template-only sections visibility (ROI is always visible)
        for frame in [self.flexibility_frame]:
            try:
                frame.pack_forget()
            except Exception:
                pass
        if not is_color:
            try:
                self.flexibility_frame.pack(fill=tk.X, pady=10, after=self.roi_frame)
            except Exception:
                self.flexibility_frame.pack(fill=tk.X, pady=10)

    def _on_color_space_change(self):
        self.use_hsv = (self.color_space_var.get() == 1)
        if self.use_hsv:
            self.hue_label.config(text="Hue tol")
            self.sat_label.config(text="Sat tol")
            self.val_label.config(text="Val tol")
            self.hue_slider.config(from_=0, to=90)
            self.sat_slider.config(from_=0, to=127)
            self.val_slider.config(from_=0, to=127)
            self.hue_slider.set(self.tol_hue)
            self.sat_slider.set(self.tol_sat)
            self.val_slider.set(self.tol_val)
        else:
            self.hue_label.config(text="Blue tol")
            self.sat_label.config(text="Green tol")
            self.val_label.config(text="Red tol")
            self.hue_slider.config(from_=0, to=255)
            self.sat_slider.config(from_=0, to=255)
            self.val_slider.config(from_=0, to=255)
            self.hue_slider.set(self.tol_b)
            self.sat_slider.set(self.tol_g)
            self.val_slider.set(self.tol_r)
        self._update_color_preview()

    def _on_color_slider_change(self):
        # sync tolerances
        if self.use_hsv:
            self.tol_hue = int(self.hue_slider.get())
            self.tol_sat = int(self.sat_slider.get())
            self.tol_val = int(self.val_slider.get())
        else:
            self.tol_b = int(self.hue_slider.get())
            self.tol_g = int(self.sat_slider.get())
            self.tol_r = int(self.val_slider.get())
        self.kernel = int(self.kernel_slider.get())
        self.min_area = int(self.min_slider.get())
        self._update_color_preview()

    def _on_ws_toggle(self):
        self.ws_enabled = (self.ws_var.get() == 1)
        self._update_color_preview()

    def _on_ws_slider_change(self):
        self.ws_sensitivity = int(self.ws_slider.get())
        self._update_color_preview()

    def _update_roi_preview(self):
        if self.selected_roi is None or self.original_image is None:
            return
        x1, y1, x2, y2 = self.selected_roi
        if x2 <= x1 or y2 <= y1:
            return
        roi_bgr = self.original_image[y1:y2, x1:x2]
        if roi_bgr.size == 0:
            return
        roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
        h, w = roi_rgb.shape[:2]
        size = self.roi_preview_size
        scale = min(size / max(1, w), size / max(1, h))
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        roi_resized = cv2.resize(roi_rgb, (new_w, new_h))
        canvas = np.full((size, size, 3), 240, dtype=np.uint8)
        y_off = (size - new_h) // 2
        x_off = (size - new_w) // 2
        canvas[y_off:y_off+new_h, x_off:x_off+new_w] = roi_resized
        pil = Image.fromarray(canvas)
        self.roi_photo = ImageTk.PhotoImage(pil)
        self.roi_preview_label.config(image=self.roi_photo, text="")

    # ---- color math helpers ----
    def _clamp_bgr_range(self, center: Tuple[int, int, int]) -> Tuple[np.ndarray, np.ndarray]:
        b, g, r = center
        lower = np.array([max(0, b - self.tol_b), max(0, g - self.tol_g), max(0, r - self.tol_r)], dtype=np.uint8)
        upper = np.array([min(255, b + self.tol_b), min(255, g + self.tol_g), min(255, r + self.tol_r)], dtype=np.uint8)
        return lower, upper

    def _most_frequent_color(self, roi: np.ndarray, color_space: str = 'bgr') -> Tuple[int, int, int]:
        if roi is None or roi.size == 0:
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
            filtered = pixels[mask]
        else:
            brightness = np.mean(pixels, axis=1)
            mask = brightness < 200
            filtered = pixels[mask]
        if len(filtered) < len(pixels) * 0.1:
            filtered = pixels
        unique_colors, counts = np.unique(filtered, axis=0, return_counts=True)
        idx = int(np.argmax(counts)) if len(counts) > 0 else 0
        color = unique_colors[idx] if len(unique_colors) > 0 else [0, 0, 0]
        return int(color[0]), int(color[1]), int(color[2])

    def _compute_mask(self, img_bgr: np.ndarray, roi_mean_bgr: Tuple[int, int, int], roi_mean_hsv: Tuple[int, int, int]) -> np.ndarray:
        if self.use_hsv:
            hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
            h, s, v = roi_mean_hsv
            h_low = h - self.tol_hue
            h_high = h + self.tol_hue
            s_low = max(0, s - self.tol_sat)
            s_high = min(255, s + self.tol_sat)
            v_low = max(0, v - self.tol_val)
            v_high = min(255, v + self.tol_val)
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
            low, high = self._clamp_bgr_range(roi_mean_bgr)
            mask = cv2.inRange(img_bgr, low, high)

        if self.kernel > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.kernel, self.kernel))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        return mask

    def _apply_watershed(self, mask: np.ndarray, img_bgr: np.ndarray) -> np.ndarray:
        if not self.ws_enabled or mask is None or img_bgr is None:
            return mask
        bin_mask = (mask > 0).astype(np.uint8) * 255
        ksize = max(3, self.kernel | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        sure_bg = cv2.dilate(bin_mask, kernel, iterations=1)
        dist = cv2.distanceTransform(bin_mask, cv2.DIST_L2, 5)
        dist_norm = dist / (dist.max() + 1e-6) if dist.max() > 0 else dist
        thr = 0.2 + (min(max(self.ws_sensitivity, 1), 80) / 80.0) * (0.7 - 0.2)
        sure_fg = (dist_norm > thr).astype(np.uint8) * 255
        unknown = cv2.subtract(sure_bg, sure_fg)
        num_markers, markers = cv2.connectedComponents((sure_fg > 0).astype(np.uint8))
        markers = markers + 1
        markers[unknown > 0] = 0
        ws_markers = cv2.watershed(img_bgr.copy(), markers)
        refined = np.zeros_like(bin_mask)
        refined[ws_markers > 1] = 255
        return refined

    def _update_color_preview(self):
        if self.original_image is None or self.selected_roi is None:
            return
        x1, y1, x2, y2 = self.selected_roi
        roi = self.original_image[y1:y2, x1:x2]
        roi_mean_bgr = self._most_frequent_color(roi, 'bgr')
        roi_mean_hsv = self._most_frequent_color(roi, 'hsv')
        mask = self._compute_mask(self.original_image, roi_mean_bgr, roi_mean_hsv)
        if self.ws_enabled:
            mask = self._apply_watershed(mask, self.original_image)
        vis, _ = self._count_and_draw(mask, cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB), max(1, self.min_area))
        self.display_image = vis
        self.display_image_on_canvas()
        self._display_color_results(mask, (x2 - x1) * (y2 - y1))

    def _count_and_draw(self, mask: np.ndarray, img_rgb: np.ndarray, min_area: int):
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

    def _display_color_results(self, mask, ref_area):
        num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        objects = []
        for i in range(1, num):
            x, y, w, h, area = stats[i]
            if area >= self.min_area:
                objects.append((x, y, w, h, area))
        ws_info = f"  WS:{'on' if self.ws_enabled else 'off'}"
        if self.ws_enabled:
            ws_info += f"({int(self.ws_slider.get())})"
        if self.use_hsv:
            mode_info = f"H:{self.tol_hue}  S:{self.tol_sat}  V:{self.tol_val}"
        else:
            mode_info = f"B:{self.tol_b}  G:{self.tol_g}  R:{self.tol_r}"
        text = (f"Reference Area: {ref_area:.1f} px²\n"
                f"Objects Detected : {len(objects)}\n\n"
                f"Mode: {'HSV' if self.use_hsv else 'BGR'}{ws_info}\n"
                f"{mode_info}\n"
                f"Kernel:{self.kernel}  MinArea:{self.min_area}")
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, text)

    def _count_objects_color(self):
        if self.selected_roi is None:
            messagebox.showwarning("Warning", "Please select a reference ROI (rectangle)")
            return
        x1, y1, x2, y2 = self.selected_roi
        if (x2 - x1) < 5 or (y2 - y1) < 5:
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, "No objects found in the image.\n")
            # Update display to show original image without highlighting
            self.display_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            self.display_image_on_canvas()
            return
        self.use_hsv = (self.color_space_var.get() == 1)
        # sync tolerances
        self._on_color_slider_change()
        roi = self.original_image[y1:y2, x1:x2]
        roi_mean_bgr = self._most_frequent_color(roi, 'bgr')
        roi_mean_hsv = self._most_frequent_color(roi, 'hsv')
        mask = self._compute_mask(self.original_image, roi_mean_bgr, roi_mean_hsv)
        if self.ws_enabled:
            mask = self._apply_watershed(mask, self.original_image)
        vis, _ = self._count_and_draw(mask, cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB), self.min_area)
        # update results text
        self._display_color_results(mask, (x2 - x1) * (y2 - y1))
        # show annotated
        self.display_image = vis
        self.display_image_on_canvas()


def main():
    root = tk.Tk()
    app = SmartObjectCounter(root)

    def on_resize(event):
        if app.image is not None:
            app.display_image_on_canvas()

    root.bind("<Configure>", on_resize)
    root.mainloop()

if __name__ == "__main__":
    main()
