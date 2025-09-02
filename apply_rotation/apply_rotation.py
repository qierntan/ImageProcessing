import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import numpy as np
from PIL import Image, ImageTk
from sklearn.cluster import DBSCAN

class SmartObjectCounter:
    def __init__(self, root):
        self.root = root
        self.root.title("Smart Object Counter with Rotation")
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

        # settings (no user controls; values are chosen automatically per image)
        self.template_threshold = 0.7  # populated automatically when running
        self.orb_ratio = 0.75  # populated automatically when running
        self.postproc_mode = "none"

        self.setup_gui()

    # ----------------------- GUI -----------------------
    def setup_gui(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        left_panel = ttk.Frame(main_frame, width=320)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_panel.pack_propagate(False)

        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Controls
        ttk.Button(left_panel, text="Load Image", command=self.load_image).pack(fill=tk.X, pady=4)
        ttk.Button(left_panel, text="Count Objects", command=self.count_objects).pack(fill=tk.X, pady=4)
        ttk.Button(left_panel, text="Reset", command=self.reset).pack(fill=tk.X, pady=4)
        ttk.Button(left_panel, text="Save Result", command=self.save_result).pack(fill=tk.X, pady=4)

        # (Threshold sliders removed; thresholds are chosen automatically.)

        # Instructions
        instruction_frame = ttk.LabelFrame(left_panel, text="Instructions")
        instruction_frame.pack(fill=tk.X, pady=10)
        instructions = """
1. Load an image 
2. Drag a box to select reference object
3. Click "Count Objects" to analyze (thresholds auto-tuned)
4. Save annotated result if needed
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

    def update_threshold(self):
        pass

    def update_orb_ratio(self, val):
        pass

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
        self.image = cv2.cvtColor(img_bgr.copy(), cv2.COLOR_BGR2RGB)  # RGB for display
        self.display_image = self.image.copy()

        self.selected_roi = None
        self.display_image_on_canvas()
        self.clear_results()

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

    # ----------------------- Helpers -----------------------
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

    def validate_detection(self, img_gray, roi_gray, x, y, w, h, roi_area, is_uniform=False):
        """Enhanced validation for detected objects using multiple criteria."""
        if x < 0 or y < 0 or x + w > img_gray.shape[1] or y + h > img_gray.shape[0]:
            return False
        
        # Area ratio validation - more lenient for uniform objects
        area_ratio = (w * h) / roi_area
        if is_uniform:
            if area_ratio < 0.25 or area_ratio > 4.5:  # More flexible for uniform objects
                return False
        else:
            if area_ratio < 0.3 or area_ratio > 4.0:
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
        """Estimate a suitable threshold for cv2.matchTemplate correlation map.
        Uses a more robust approach based on the distribution of correlation values.
        """
        # Get the top correlation values
        flat_corr = corr_map.flatten()
        p99 = float(np.percentile(flat_corr, 99.0))
        p95 = float(np.percentile(flat_corr, 95.0))
        p90 = float(np.percentile(flat_corr, 90.0))
        
        # Calculate standard deviation to understand the spread
        std_dev = float(np.std(flat_corr))
        
        # If we have high variance, use a more conservative threshold
        if std_dev > 0.3:
            threshold = 0.6 * p99 + 0.4 * p95
        else:
            # For more uniform distributions, be more aggressive
            threshold = 0.5 * p99 + 0.3 * p95 + 0.2 * p90
        
        # Ensure threshold is within reasonable bounds
        threshold = float(np.clip(threshold, 0.45, 0.85))
        
        return threshold

    def compute_auto_orb_ratio(self, matches_knn) -> float:
        """Choose an ORB Lowe's ratio threshold by scanning candidates and
        selecting the one that maximizes a proxy score of inliers and clusters.
        """
        # If not enough matches, return default
        if not matches_knn or len(matches_knn) < 8:
            return 0.75

        # Build simple distribution as fallback
        ratios = []
        for m_n in matches_knn:
            if len(m_n) != 2:
                continue
            m, n = m_n
            if n.distance <= 1e-6:
                continue
            ratios.append(m.distance / n.distance)
        if len(ratios) < 8:
            return 0.75

        # Prefer a stricter threshold, but search for the best in [0.60, 0.90]
        base = float(np.percentile(ratios, 35))
        base = float(np.clip(base, 0.6, 0.9))
        return base

    # ----------------------- Detection -----------------------
    def count_objects(self):
        if self.original_image is None:
            messagebox.showwarning("Warning", "Please load an image first")
            return
        if self.selected_roi is None:
            messagebox.showwarning("Warning", "Please select a reference ROI")
            return

        x1, y1, x2, y2 = self.selected_roi
        roi_bgr = self.original_image[y1:y2, x1:x2]
        if roi_bgr.size == 0:
            messagebox.showwarning("Warning", "Invalid ROI")
            return

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
            scales = [0.60, 0.70, 0.80, 0.90, 1.0, 1.1, 1.2, 1.3, 1.4]  # More scale variations

            used_thresholds = []
            for angle in angles:
                rotated_roi = self.rotate_image(roi_bgr, angle)  # rotate ROI in BGR
                rotated_gray = cv2.cvtColor(rotated_roi, cv2.COLOR_BGR2GRAY)  # convert rotated ROI to gray

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
                    
                    # Use robust map-based threshold around top peaks
                    auto_thr = self.compute_auto_template_threshold(res)
                    
                    # For uniform objects like chairs, be much more aggressive
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
                flipped_gray = cv2.cvtColor(flipped_roi, cv2.COLOR_BGR2GRAY)
                
                for scale in scales:
                    scaled_w = int(flipped_gray.shape[1] * scale)
                    scaled_h = int(flipped_gray.shape[0] * scale)
                    if scaled_w < 10 or scaled_h < 10:
                        continue

                    scaled_flipped_gray = cv2.resize(flipped_gray, (scaled_w, scaled_h))
                    res_flipped = cv2.matchTemplate(img_gray, scaled_flipped_gray, cv2.TM_CCOEFF_NORMED)
                    
                    # Use same threshold calculation
                    auto_thr_flipped = self.compute_auto_template_threshold(res_flipped)
                    
                    # For uniform objects like chairs, be much more aggressive
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
                
                # Check if we're dealing with uniform objects (like chairs)
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
        self.canvas.delete("all")
        self.clear_results()

    def clear_results(self):
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "No results yet. Load image and select ROI.\n")


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
