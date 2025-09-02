import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
from sklearn.cluster import DBSCAN

class SmartObjectCounter:
    def __init__(self, root):
        self.root = root
        self.root.title("Smart Object Counter")
        self.root.geometry("1200x800")

        # Images/state
        self.image = None                 # RGB for display
        self.original_image = None        # BGR for processing
        self.display_image = None         # RGB for display
        self.photo = None

        # Canvas scale/offset
        self.scale_factor = 1.0
        self.canvas_offset_x = 0
        self.canvas_offset_y = 0

        # Selection & detection state
        self.rect_start = None
        self.rect_end = None
        self.drawing = False
        self.selected_roi = None
        self.results = {}
        self.detected_objects = []        # list of dicts or tuples
        self.object_highlighted = False

        # YOLO (lazy load)
        self.yolo_model = None
        self.yolo_names = None
        self.use_yolo = True

        # Rotation/Scaling options
        self.enable_scaling = tk.BooleanVar(value=False)
        self.enable_rotation = tk.BooleanVar(value=False)

        # Sliders (shown only when Rotation is enabled)
        self.template_threshold = 0.70   # for template matching
        self.orb_ratio = 0.75            # Lowe's ratio for ORB

        # Post-processing mode (shown with Rotation)
        self.postproc_mode = tk.StringVar(value="none")  # "none" | "nms" | "watershed"

        self.setup_gui()

    # ---------------------------------------------------------
    # GUI
    # ---------------------------------------------------------
    def setup_gui(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        left_panel = ttk.Frame(main_frame, width=330)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_panel.pack_propagate(False)

        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Controls
        ttk.Button(left_panel, text="Load Image", command=self.load_image).pack(fill=tk.X, pady=5)
        ttk.Button(left_panel, text="Count Objects", command=self.count_objects).pack(fill=tk.X, pady=5)
        ttk.Button(left_panel, text="Reset", command=self.reset).pack(fill=tk.X, pady=5)
        ttk.Button(left_panel, text="Save Result", command=self.save_result).pack(fill=tk.X, pady=5)

        # Method Selection 
        methods_frame = ttk.LabelFrame(left_panel, text="Methods")
        methods_frame.pack(fill=tk.X, pady=10)
        ttk.Checkbutton(methods_frame, text="Scaling", variable=self.enable_scaling).pack(anchor=tk.W, padx=8, pady=2)
        ttk.Checkbutton(methods_frame, text="Rotation", variable=self.enable_rotation,
                        command=self._on_rotation_toggle).pack(anchor=tk.W, padx=8, pady=2)

        # Rotation-only controls (hidden when Rotation unchecked)
        self.rotation_opts_frame = ttk.LabelFrame(left_panel, text="Rotation Options")
        # Initially hidden; will be shown by _on_rotation_toggle()

        # Template Matching Threshold slider
        thr_frame = ttk.Frame(self.rotation_opts_frame)
        ttk.Label(thr_frame, text="Template Threshold").pack(anchor=tk.W)
        self.threshold_slider = tk.Scale(
            thr_frame, from_=0.50, to=0.95, resolution=0.01, orient=tk.HORIZONTAL,
            command=lambda v: self._update_threshold())
        self.threshold_slider.set(self.template_threshold)
        self.threshold_slider.pack(fill=tk.X)
        thr_frame.pack(fill=tk.X, padx=8, pady=6)

        # ORB Ratio slider
        ratio_frame = ttk.Frame(self.rotation_opts_frame)
        ttk.Label(ratio_frame, text="ORB Ratio (Lowe)").pack(anchor=tk.W)
        self.ratio_slider = tk.Scale(
            ratio_frame, from_=0.60, to=0.90, resolution=0.01, orient=tk.HORIZONTAL,
            command=lambda v: self._update_orb_ratio())
        self.ratio_slider.set(self.orb_ratio)
        self.ratio_slider.pack(fill=tk.X)
        ratio_frame.pack(fill=tk.X, padx=8, pady=6)

        # Post-processing choices
        post_frame = ttk.LabelFrame(self.rotation_opts_frame, text="Post-processing")
        ttk.Radiobutton(post_frame, text="None", variable=self.postproc_mode, value="none").pack(anchor=tk.W, padx=6)
        ttk.Radiobutton(post_frame, text="NMS", variable=self.postproc_mode, value="nms").pack(anchor=tk.W, padx=6)
        ttk.Radiobutton(post_frame, text="Watershed (ORB only)", variable=self.postproc_mode, value="watershed").pack(anchor=tk.W, padx=6)
        post_frame.pack(fill=tk.X, padx=8, pady=6)

        # Instructions
        instruction_frame = ttk.LabelFrame(left_panel, text="Instructions")
        instruction_frame.pack(fill=tk.X, pady=10)
        instructions = (
            "1. Load an image (YOLO auto-detects objects)\n"
            "2. Click a highlighted object to select reference, OR drag an ROI\n"
            "3. Choose method(s) and adjust sliders if needed\n"
            "4. Click 'Count Objects' to analyze"
        )
        ttk.Label(instruction_frame, text=instructions, justify=tk.LEFT).pack(padx=10, pady=8)

        # Results
        results_frame = ttk.LabelFrame(left_panel, text="Results")
        results_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        self.results_text = tk.Text(results_frame, height=12, width=36)
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Canvas
        self.canvas = tk.Canvas(right_panel, bg="white")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Bind mouse events
        self.canvas.bind("<Button-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)

        self.clear_results()
        self._on_rotation_toggle()  # ensure proper initial visibility

    def _on_rotation_toggle(self):
        # Show/hide the rotation options frame (sliders + postproc) based on Rotation checkbox
        if self.enable_rotation.get():
            self.rotation_opts_frame.pack(fill=tk.X, pady=10)
        else:
            self.rotation_opts_frame.pack_forget()

    def _update_threshold(self):
        self.template_threshold = float(self.threshold_slider.get())

    def _update_orb_ratio(self):
        self.orb_ratio = float(self.ratio_slider.get())

    # ---------------------------------------------------------
    # YOLO helpers
    # ---------------------------------------------------------
    def _ensure_yolo(self):
        """Lazy-load YOLOv8 (yolov8n.pt)."""
        if not self.use_yolo:
            return False
        if self.yolo_model is not None:
            return True
        try:
            from ultralytics import YOLO
            self.yolo_model = YOLO("yolov8n.pt")
            # names mapping
            self.yolo_names = self.yolo_model.model.names if hasattr(self.yolo_model, "model") else None
            return True
        except Exception as e:
            print(f"[YOLO] Failed to load: {e}")
            self.use_yolo = False
            self.yolo_model = None
            self.yolo_names = None
            return False

    def _norm_box(self, obj):
        """Return (x,y,w,h,label,conf) from either dict or tuple."""
        if isinstance(obj, dict):
            return (
                obj.get('x', 0), obj.get('y', 0), obj.get('w', 0), obj.get('h', 0),
                obj.get('label', ''), obj.get('conf', 0.0)
            )
        if isinstance(obj, (tuple, list)):
            if len(obj) >= 4:
                return obj[0], obj[1], obj[2], obj[3], (obj[4] if len(obj) > 4 else ''), (obj[5] if len(obj) > 5 else 0.0)
        return 0, 0, 0, 0, '', 0.0

    # ---------------------------------------------------------
    # Image I/O
    # ---------------------------------------------------------
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

        self.original_image = img_bgr.copy()                # BGR for processing
        self.image = cv2.cvtColor(img_bgr.copy(), cv2.COLOR_BGR2RGB)  # RGB for display
        self.display_image = self.image.copy()
        self.selected_roi = None
        self.detected_objects = []
        self.object_highlighted = False

        self.display_image_on_canvas()
        self.clear_results()

        # Auto-detect with YOLO and highlight
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

    # ---------------------------------------------------------
    # Canvas Interactions
    # ---------------------------------------------------------
    def on_mouse_down(self, event):
        if self.image is None:
            return

        # If YOLO boxes are highlighted, allow click-to-select ROI
        if self.object_highlighted and self.detected_objects:
            canvas_x = (event.x - self.canvas_offset_x) / self.scale_factor
            canvas_y = (event.y - self.canvas_offset_y) / self.scale_factor
            for i, obj in enumerate(self.detected_objects):
                x, y, w, h, _, _ = self._norm_box(obj)
                if x <= canvas_x <= x + w and y <= canvas_y <= y + h:
                    self.selected_roi = (x, y, x + w, y + h)
                    self.highlight_selected_object(i)
                    return

        # Otherwise, allow manual ROI drawing
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

    # ---------------------------------------------------------
    # YOLO auto-detect & highlight
    # ---------------------------------------------------------
    def auto_detect_objects(self):
        if self.original_image is None:
            return
        try:
            if self._ensure_yolo():
                results = self.yolo_model(self.original_image, verbose=False)[0]
                yolo_boxes = []
                for b in results.boxes:
                    x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                    cls_id = int(b.cls[0]) if hasattr(b, 'cls') else -1
                    label = results.names[cls_id] if hasattr(results, 'names') and cls_id in results.names else ''
                    conf = float(b.conf[0]) if hasattr(b, 'conf') else 0.0
                    x, y, w, h = x1, y1, x2 - x1, y2 - y1
                    yolo_boxes.append({'x': x, 'y': y, 'w': w, 'h': h, 'label': label, 'conf': conf})
                # light confidence filter
                self.detected_objects = [d for d in yolo_boxes if d['conf'] >= 0.10]
                if len(self.detected_objects) > 0:
                    self.highlight_detected_objects()
                    self.object_highlighted = True
                    messagebox.showinfo("Info", f"Found {len(self.detected_objects)} objects (YOLO). Click one to set ROI.")
                else:
                    self.object_highlighted = False
        except Exception as e:
            print(f"[YOLO] detection error: {e}")

    def highlight_detected_objects(self):
        if not self.detected_objects or self.original_image is None:
            return
        highlight_image = self.original_image.copy()
        for i, obj in enumerate(self.detected_objects):
            x, y, w, h, label, conf = self._norm_box(obj)
            color = (255, 0, 255)
            cv2.rectangle(highlight_image, (x, y), (x + w, y + h), color, 2)
            name = label if label else f"Object {i+1}"
            cv2.putText(highlight_image, name, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        self.display_image = cv2.cvtColor(highlight_image, cv2.COLOR_BGR2RGB)
        self.display_image_on_canvas()

    def highlight_selected_object(self, idx):
        if not self.detected_objects or self.original_image is None:
            return
        img = self.original_image.copy()
        for i, obj in enumerate(self.detected_objects):
            x, y, w, h, label, conf = self._norm_box(obj)
            if i == idx:
                color = (0, 255, 0); thickness = 3
                ref_text = f"Reference: {label}" if label else f"Reference (Object {i+1})"
                cv2.putText(img, ref_text, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                self.selected_roi = (x, y, x + w, y + h)
            else:
                color = (255, 0, 255); thickness = 2
                name = label if label else f"Object {i+1}"
                cv2.putText(img, name, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)
        self.display_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.display_image_on_canvas()

    # ---------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------
    def rotate_image(self, img, angle):
        # rotate BGR image about center, keep size
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            img, M, (w, h), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255)
        )
        return rotated

    def non_max_suppression(self, rectangles, overlapThresh=0.3):
        if len(rectangles) == 0:
            return []
        boxes = np.array(rectangles)
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 0] + boxes[:, 2]
        y2 = boxes[:, 1] + boxes[:, 3]
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)
        pick = []
        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            overlap = (w * h) / area[idxs[:last]]
            idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))
        return [rectangles[i] for i in pick]

    def apply_watershed(self, image, detections):
        if len(detections) == 0:
            return detections
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        for det in detections:
            x, y, w, h = det["bbox"]
            cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        image_copy = image.copy()
        markers = cv2.watershed(image_copy, markers)
        new_detections = []
        for label in np.unique(markers):
            if label <= 1:
                continue
            mask_region = np.uint8(markers == label)
            cnts, _ = cv2.findContours(mask_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                x, y, w, h = cv2.boundingRect(c)
                new_detections.append({"bbox": (x, y, w, h)})
        return new_detections

    # ---------------------------------------------------------
    # Counting
    # ---------------------------------------------------------
    def count_objects(self):
        if self.original_image is None:
            messagebox.showwarning("Warning", "Please load an image first")
            return
        if self.selected_roi is None:
            messagebox.showwarning("Warning", "Please select a reference ROI (click a YOLO box or draw one)")
            return

        x1, y1, x2, y2 = self.selected_roi
        roi_bgr = self.original_image[y1:y2, x1:x2]
        if roi_bgr.size == 0:
            messagebox.showwarning("Warning", "Invalid ROI")
            return

        # Build angle/scale lists based on method checkboxes
        angles = list(range(0, 360, 30)) if self.enable_rotation.get() else [0]
        scales = [0.8, 1.0, 1.2] if self.enable_scaling.get() else [1.0]

        # Try ORB first; if not enough features, fall back to template matching
        orb = cv2.ORB_create(nfeatures=400)
        roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        img_gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        kp_roi, des_roi = orb.detectAndCompute(roi_gray, None)
        kp_img, des_img = orb.detectAndCompute(img_gray, None)

        use_orb = (des_roi is not None and des_img is not None and kp_roi is not None and len(kp_roi) >= 20)
        method_used = "ORB (FLANN)" if use_orb else "Template Matching"

        vis_bgr = self.original_image.copy()
        objects_found = 0
        rectangles = []

        if use_orb:
            # FLANN LSH for ORB
            index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=2)
            search_params = dict(checks=50)
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            try:
                matches_knn = flann.knnMatch(des_roi, des_img, k=2)
            except Exception:
                matches_knn = []

            good = []
            r = float(self.orb_ratio)
            for m_n in matches_knn:
                if len(m_n) != 2:
                    continue
                m, n = m_n
                if m.distance < r * n.distance:
                    good.append(m)

            orb_rects = []
            if len(good) >= 8:
                dst_pts = np.float32([kp_img[m.trainIdx].pt for m in good])
                roi_diag = float(np.hypot(roi_bgr.shape[1], roi_bgr.shape[0]))
                eps = max(10.0, roi_diag * 0.25)
                clustering = DBSCAN(eps=eps, min_samples=6).fit(dst_pts)
                labels = clustering.labels_
                unique_labels = set(labels)
                if -1 in unique_labels:
                    unique_labels.remove(-1)

                for lbl in unique_labels:
                    inds = [i for i, lab in enumerate(labels) if lab == lbl]
                    if len(inds) < 6:
                        continue
                    cluster_matches = [good[i] for i in inds]
                    src_pts = np.float32([kp_roi[m.queryIdx].pt for m in cluster_matches]).reshape(-1, 1, 2)
                    dst_pts_cluster = np.float32([kp_img[m.trainIdx].pt for m in cluster_matches]).reshape(-1, 1, 2)
                    M, mask = cv2.findHomography(src_pts, dst_pts_cluster, cv2.RANSAC, 5.0)
                    if M is not None and mask is not None:
                        inliers = mask.ravel().sum()
                        if inliers < 10:
                            continue
                        h_roi, w_roi = roi_bgr.shape[:2]
                        pts = np.float32([[0, 0], [w_roi, 0], [w_roi, h_roi], [0, h_roi]]).reshape(-1, 1, 2)
                        dst = cv2.perspectiveTransform(pts, M)
                        x, y, w, h = cv2.boundingRect(np.int32(dst))
                        orb_rects.append((x, y, w, h))

            # Post-processing
            mode = self.postproc_mode.get()
            if mode == "nms":
                orb_rects = self.non_max_suppression(orb_rects, overlapThresh=0.3)
            elif mode == "watershed":
                dets = [{"bbox": b} for b in orb_rects]
                dets = self.apply_watershed(vis_bgr, dets)
                orb_rects = [d["bbox"] for d in dets]

            rectangles = orb_rects

        else:
            # Template Matching with angles/scales based on checkboxes
            rectangles_tm = []
            for angle in angles:
                rotated_roi = self.rotate_image(roi_bgr, angle)
                rotated_gray = cv2.cvtColor(rotated_roi, cv2.COLOR_BGR2GRAY)
                for scale in scales:
                    sw = int(rotated_gray.shape[1] * scale)
                    sh = int(rotated_gray.shape[0] * scale)
                    if sw < 10 or sh < 10:
                        continue
                    tpl = cv2.resize(rotated_gray, (sw, sh), interpolation=cv2.INTER_AREA)
                    res = cv2.matchTemplate(img_gray, tpl, cv2.TM_CCOEFF_NORMED)
                    loc = np.where(res >= float(self.template_threshold))
                    for pt in zip(*loc[::-1]):
                        rectangles_tm.append([pt[0], pt[1], tpl.shape[1], tpl.shape[0]])

            # Group & optional NMS
            if rectangles_tm:
                rectangles_tm, _ = cv2.groupRectangles(rectangles_tm, groupThreshold=1, eps=0.5)
                if self.postproc_mode.get() == "nms":
                    rectangles_tm = self.non_max_suppression(rectangles_tm, overlapThresh=0.3)
            rectangles = rectangles_tm

        # Draw detections
        for (x, y, w, h) in rectangles:
            cv2.rectangle(vis_bgr, (x, y), (x + w, y + h), (0, 255, 0) if method_used.startswith("ORB") else (255, 255, 150), 2)
        objects_found = len(rectangles)

        # Reference area (bounding box)
        ref_area = max(1, (x2 - x1) * (y2 - y1))
        # Classify sizes 
        small, same, large = [], [], []
        small_threshold = 0.7
        large_threshold = 1.3
        for (x, y, w, h) in rectangles:
            area = w * h
            ratio = area / ref_area
            if ratio < small_threshold:
                small.append((x, y, w, h, area))
            elif ratio > large_threshold:
                large.append((x, y, w, h, area))
            else:
                same.append((x, y, w, h, area))

        # Annotate S/M/L
        for (x, y, w, h, _) in small:
            cv2.rectangle(vis_bgr, (x, y), (x + w, y + h), (0, 0, 255), 1)
            cv2.putText(vis_bgr, "S", (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        for (x, y, w, h, _) in same:
            cv2.rectangle(vis_bgr, (x, y), (x + w, y + h), (255, 0, 0), 1)
            cv2.putText(vis_bgr, "M", (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        for (x, y, w, h, _) in large:
            cv2.rectangle(vis_bgr, (x, y), (x + w, y + h), (255, 255, 0), 1)
            cv2.putText(vis_bgr, "L", (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

        # Draw reference ROI
        cv2.rectangle(vis_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis_bgr, "Reference", (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

        self.display_image = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)
        self.display_image_on_canvas()

        # Results text
        self.results_text.delete(1.0, tk.END)
        info = []
        info.append(f"Method used: {method_used}")
        info.append(f"Objects Found: {objects_found}")
        info.append(f"Reference Area: {ref_area}")
        info.append(f"Scaling enabled: {'Yes' if self.enable_scaling.get() else 'No'}")
        info.append(f"Rotation enabled: {'Yes' if self.enable_rotation.get() else 'No'}")
        if not method_used.startswith("ORB"):
            info.append(f"Template Threshold: {self.template_threshold:.2f}")
        else:
            info.append(f"ORB Ratio: {self.orb_ratio:.2f}")
        info.append(f"Post-processing: {self.postproc_mode.get().capitalize()}")
        info.append("")
        info.append(f"Small: {len(small)} | Same: {len(same)} | Large: {len(large)}")
        self.results_text.insert(tk.END, "\n".join(info))

    # ---------------------------------------------------------
    # Save / Reset
    # ---------------------------------------------------------
    def save_result(self):
        if self.display_image is None:
            messagebox.showwarning("Warning", "No result image to save")
            return
        save_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg;*.jpeg")]
        )
        if not save_path:
            return
        bgr = cv2.cvtColor(self.display_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, bgr)
        messagebox.showinfo("Saved", f"Result image saved to:\n{save_path}")

    def reset(self):
        self.image = None
        self.original_image = None
        self.display_image = None
        self.photo = None
        self.detected_objects = []
        self.object_highlighted = False
        self.selected_roi = None
        self.canvas.delete("all")
        self.clear_results()

    def clear_results(self):
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "No results yet. Load image and select ROI.\n")


# ---------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------
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
