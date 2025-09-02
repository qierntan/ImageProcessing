import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import numpy as np
from PIL import Image, ImageTk
from dataclasses import dataclass
from typing import List, Tuple

# Optional imports (YOLO, DBSCAN). Guarded in code
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

try:
    from sklearn.cluster import DBSCAN
except Exception:
    DBSCAN = None


@dataclass
class Tolerances:
    hue: int = 15
    sat: int = 60
    val: int = 60
    r: int = 30
    g: int = 30
    b: int = 30


def get_most_frequent_color(roi: np.ndarray, color_space: str = 'bgr') -> Tuple[int, int, int]:
    if roi is None or roi.size == 0:
        return (0, 0, 0)
    if color_space == 'hsv':
        roi_converted = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    else:
        roi_converted = roi.copy()
    pixels = roi_converted.reshape(-1, 3)
    if len(pixels) > 10000:
        step = max(1, len(pixels) // 5000)
        pixels = pixels[::step]
    if color_space == 'hsv':
        mask = (pixels[:, 1] > 30) & (pixels[:, 2] < 240)
        filtered = pixels[mask]
    else:
        brightness = np.mean(pixels, axis=1)
        filtered = pixels[brightness < 200]
    if len(filtered) < len(pixels) * 0.1:
        filtered = pixels
    unique_colors, counts = np.unique(filtered, axis=0, return_counts=True)
    idx = int(np.argmax(counts))
    return tuple(map(int, unique_colors[idx]))


def compute_mask(
    img_bgr: np.ndarray,
    roi_mean_bgr: Tuple[int, int, int],
    roi_mean_hsv: Tuple[int, int, int],
    tol: Tolerances,
    use_hsv: bool,
    morph_kernel: int,
) -> np.ndarray:
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
        b, g, r = roi_mean_bgr
        low = np.array([max(0, b - tol.b), max(0, g - tol.g), max(0, r - tol.r)], dtype=np.uint8)
        high = np.array([min(255, b + tol.b), min(255, g + tol.g), min(255, r + tol.r)], dtype=np.uint8)
        mask = cv2.inRange(img_bgr, low, high)
    if morph_kernel > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return mask


def apply_watershed_split(mask: np.ndarray, img_bgr: np.ndarray, sensitivity: int, morph_kernel: int) -> np.ndarray:
    if mask is None or img_bgr is None:
        return mask
    bin_mask = (mask > 0).astype(np.uint8) * 255
    ksize = max(3, morph_kernel | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    sure_bg = cv2.dilate(bin_mask, kernel, iterations=1)
    dist = cv2.distanceTransform(bin_mask, cv2.DIST_L2, 5)
    if dist.max() > 0:
        dist_norm = dist / (dist.max() + 1e-6)
    else:
        dist_norm = dist
    thr = 0.2 + (min(max(sensitivity, 1), 80) / 80.0) * (0.7 - 0.2)
    sure_fg = (dist_norm > thr).astype(np.uint8) * 255
    unknown = cv2.subtract(sure_bg, sure_fg)
    num_markers, markers = cv2.connectedComponents((sure_fg > 0).astype(np.uint8))
    markers = markers + 1
    markers[unknown > 0] = 0
    ws_markers = cv2.watershed(img_bgr.copy(), markers)
    refined = np.zeros_like(bin_mask)
    refined[ws_markers > 1] = 255
    return refined


def nms_boxes(boxes: List[Tuple[int, int, int, int]], iou_thresh: float = 0.3) -> List[Tuple[int, int, int, int]]:
    if not boxes:
        return []
    arr = np.array(boxes, dtype=float)
    x1 = arr[:, 0]
    y1 = arr[:, 1]
    x2 = arr[:, 0] + arr[:, 2]
    y2 = arr[:, 1] + arr[:, 3]
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
        union = areas[i] + areas[idxs[1:]] - inter + 1e-6
        iou = inter / union
        idxs = idxs[np.where(iou <= iou_thresh)[0] + 1]
    return arr[picked].astype(int).tolist()


# Classical mask->bboxes (from applyscaling2)

def _merge_adjacent_boxes(bboxes, image_shape):
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
        return inter / (union + 1e-6)

    def gap(a, b):
        ax1, ay1, aw, ah = a
        bx1, by1, bw, bh = b
        ax2, ay2 = ax1 + aw, ay1 + ah
        bx2, by2 = bx1 + bw, by1 + bh
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
                if (area_a < small_area_thresh and area_b < small_area_thresh):
                    should_merge = iou(a, b) > 0.6
                else:
                    should_merge = iou(a, b) > 0.4 or close_aligned
                if should_merge:
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


def extract_bboxes_from_mask(binary_mask: np.ndarray) -> List[Tuple[int, int, int, int]]:
    mask = (binary_mask > 0).astype(np.uint8) * 255
    try:
        mask_eroded = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
        dist = cv2.distanceTransform(mask_eroded, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist, 0.60 * dist.max(), 255, 0)
        sure_fg = sure_fg.astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        sure_bg = cv2.dilate(mask_eroded, kernel, iterations=2)
        unknown = cv2.subtract(sure_bg, sure_fg)
        num_labels, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        color = cv2.cvtColor(mask_eroded, cv2.COLOR_GRAY2BGR)
        cv2.watershed(color, markers)
        h_img, w_img = mask.shape[:2]
        bboxes = []
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
            aspect_ratio = w / h if h > 0 else 0
            bbox_area = w * h
            extent = area / (bbox_area + 1e-6)
            if aspect_ratio > 6.0 and h < int(0.18 * h_img):
                continue
            if h < max(10, int(0.02 * h_img)) or w < max(10, int(0.02 * w_img)):
                continue
            if extent < 0.22:
                continue
            bboxes.append((x, y, w, h))
        return _merge_adjacent_boxes(bboxes, (h_img, w_img))
    except Exception:
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
            extent = area / (bbox_area + 1e-6)
            if aspect_ratio > 6.0 and h < int(0.18 * h_img):
                continue
            if h < max(10, int(0.02 * h_img)) or w < max(10, int(0.02 * w_img)):
                continue
            if extent < 0.22:
                continue
            bboxes.append((x, y, w, h))
        return _merge_adjacent_boxes(bboxes, (h_img, w_img))


class UnifiedObjectCounter:
    def __init__(self, root):
        self.root = root
        self.root.title("Unified Smart Object Counter")
        self.root.geometry("1280x860")

        # images/state
        self.original_bgr = None
        self.display_rgb = None
        self.photo = None
        self.scale_factor = 1.0
        self.canvas_offset_x = 0
        self.canvas_offset_y = 0

        # selection
        self.detected_objects = []  # (x,y,w,h,label,conf)
        self.selected_roi = None
        self.object_highlighted = False

        # YOLO
        self.yolo_model = None
        self.use_yolo = True

        # method options
        self.template_threshold = 0.7
        self.orb_ratio = 0.75
        self.color_tol = Tolerances()
        self.color_use_hsv = True
        self.color_kernel = 3
        self.color_min_area = 100
        self.color_ws = False
        self.color_ws_sens = 35

        # UI
        self._setup_gui()

    # --------------- GUI ---------------
    def _setup_gui(self):
        main = ttk.Frame(self.root)
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left scrollable panel container
        left_container = ttk.Frame(main, width=350)
        left_container.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_container.pack_propagate(False)

        # Create canvas + scrollbar + inner frame
        left_canvas = tk.Canvas(left_container, width=330, highlightthickness=0)
        left_scrollbar = ttk.Scrollbar(left_container, orient="vertical", command=left_canvas.yview)
        scrollable_frame = ttk.Frame(left_canvas, width=330)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: left_canvas.configure(scrollregion=left_canvas.bbox("all"))
        )
        left_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw", width=330)
        left_canvas.configure(yscrollcommand=left_scrollbar.set)

        left_canvas.pack(side="left", fill="both", expand=True)
        left_scrollbar.pack(side="right", fill="y")

        # Mouse wheel scroll binding
        def _on_mousewheel(event):
            delta = -1 * int(event.delta / 120)
            left_canvas.yview_scroll(delta, "units")
        left_canvas.bind_all("<MouseWheel>", _on_mousewheel)

        right = ttk.Frame(main)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Controls
        ttk.Button(scrollable_frame, text="Load Image", command=self.load_image).pack(fill=tk.X, pady=4)
        ttk.Button(scrollable_frame, text="Run Count", command=self.run_count).pack(fill=tk.X, pady=4)
        ttk.Button(scrollable_frame, text="Reset", command=self.reset).pack(fill=tk.X, pady=4)
        ttk.Button(scrollable_frame, text="Save Result", command=self.save_result).pack(fill=tk.X, pady=4)

        # Instructions
        inst = ttk.LabelFrame(scrollable_frame, text="Flow")
        inst.pack(fill=tk.X, pady=8)
        ttk.Label(inst, text=(
            "1) Load image (YOLO auto-detect)\n"
            "2) Click a detected box or draw ROI\n"
            "3) Choose one/more methods\n"
            "4) Run Count to get total"
        ), justify=tk.LEFT).pack(padx=8, pady=6)

        # Method toggles
        methods = ttk.LabelFrame(scrollable_frame, text="Methods")
        methods.pack(fill=tk.X, pady=8)
        self.var_scaling = tk.IntVar(value=1)
        self.var_rotation = tk.IntVar(value=1)
        self.var_color = tk.IntVar(value=1)
        self.var_gray = tk.IntVar(value=1)
        ttk.Checkbutton(methods, text="Scaling / Watershed", variable=self.var_scaling,
                        command=self._update_settings_visibility).pack(anchor=tk.W)
        ttk.Checkbutton(methods, text="Rotation (ORB/Template)", variable=self.var_rotation,
                        command=self._update_settings_visibility).pack(anchor=tk.W)
        ttk.Checkbutton(methods, text="Color Matching (HSV/BGR)", variable=self.var_color,
                        command=self._update_settings_visibility).pack(anchor=tk.W)
        ttk.Checkbutton(methods, text="Grayscale / Contours", variable=self.var_gray,
                        command=self._update_settings_visibility).pack(anchor=tk.W)

        # Template/ORB settings (Rotation)
        self.frame_rotation_settings = ttk.LabelFrame(scrollable_frame, text="Rotation/Template Settings")
        self.frame_rotation_settings.pack(fill=tk.X, pady=8)
        ttk.Label(self.frame_rotation_settings, text="Template threshold").pack(anchor=tk.W, padx=6)
        self.slider_thresh = tk.Scale(self.frame_rotation_settings, from_=0.5, to=0.95, resolution=0.01, orient=tk.HORIZONTAL,
                                      command=lambda v: self._on_thresh_change())
        self.slider_thresh.set(self.template_threshold)
        self.slider_thresh.pack(fill=tk.X, padx=6)
        ttk.Label(self.frame_rotation_settings, text="ORB ratio").pack(anchor=tk.W, padx=6)
        self.slider_orb = tk.Scale(self.frame_rotation_settings, from_=0.6, to=0.9, resolution=0.01, orient=tk.HORIZONTAL,
                                   command=lambda v: self._on_orb_change())
        self.slider_orb.set(self.orb_ratio)
        self.slider_orb.pack(fill=tk.X, padx=6)

        # Color settings (BGR/HSV, kernel, min area, segmentation refinement + sensitivity)
        self.frame_color_settings = ttk.LabelFrame(scrollable_frame, text="Color Matching Settings")
        self.frame_color_settings.pack(fill=tk.X, pady=8)
        self.mode_var = tk.IntVar(value=1)  # 1 HSV, 0 BGR
        ttk.Radiobutton(self.frame_color_settings, text="HSV", variable=self.mode_var, value=1, command=self._on_color_mode).pack(anchor=tk.W)
        ttk.Radiobutton(self.frame_color_settings, text="BGR", variable=self.mode_var, value=0, command=self._on_color_mode).pack(anchor=tk.W)

        # Tolerance sliders (labels and controls switch depending on mode)
        self.label_tol1 = ttk.Label(self.frame_color_settings, text="Hue tol")
        self.label_tol1.pack(anchor=tk.W, padx=6)
        self.slider_tol1 = tk.Scale(self.frame_color_settings, from_=0, to=90, orient=tk.HORIZONTAL,
                                    command=lambda v: self._on_color_sliders_change())
        self.slider_tol1.set(self.color_tol.hue)
        self.slider_tol1.pack(fill=tk.X, padx=6)

        self.label_tol2 = ttk.Label(self.frame_color_settings, text="Sat tol")
        self.label_tol2.pack(anchor=tk.W, padx=6)
        self.slider_tol2 = tk.Scale(self.frame_color_settings, from_=0, to=127, orient=tk.HORIZONTAL,
                                    command=lambda v: self._on_color_sliders_change())
        self.slider_tol2.set(self.color_tol.sat)
        self.slider_tol2.pack(fill=tk.X, padx=6)

        self.label_tol3 = ttk.Label(self.frame_color_settings, text="Val tol")
        self.label_tol3.pack(anchor=tk.W, padx=6)
        self.slider_tol3 = tk.Scale(self.frame_color_settings, from_=0, to=127, orient=tk.HORIZONTAL,
                                    command=lambda v: self._on_color_sliders_change())
        self.slider_tol3.set(self.color_tol.val)
        self.slider_tol3.pack(fill=tk.X, padx=6)

        ttk.Label(self.frame_color_settings, text="Morph kernel").pack(anchor=tk.W, padx=6)
        self.slider_kernel = tk.Scale(self.frame_color_settings, from_=0, to=25, orient=tk.HORIZONTAL,
                                      command=lambda v: self._on_color_settings_change())
        self.slider_kernel.set(self.color_kernel)
        self.slider_kernel.pack(fill=tk.X, padx=6)
        ttk.Label(self.frame_color_settings, text="Min area").pack(anchor=tk.W, padx=6)
        self.slider_minarea = tk.Scale(self.frame_color_settings, from_=1, to=20000, orient=tk.HORIZONTAL,
                                       command=lambda v: self._on_color_settings_change())
        self.slider_minarea.set(self.color_min_area)
        self.slider_minarea.pack(fill=tk.X, padx=6)
        self.ws_var = tk.IntVar(value=0)
        ttk.Checkbutton(self.frame_color_settings, text="Segmentation refinement (watershed)", variable=self.ws_var,
                        command=lambda: self._on_color_settings_change()).pack(anchor=tk.W)
        ttk.Label(self.frame_color_settings, text="Sensitivity").pack(anchor=tk.W, padx=6)
        self.slider_ws = tk.Scale(self.frame_color_settings, from_=1, to=80, orient=tk.HORIZONTAL,
                                  command=lambda v: self._on_color_settings_change())
        self.slider_ws.set(self.color_ws_sens)
        self.slider_ws.pack(fill=tk.X, padx=6)

        # Results box
        res = ttk.LabelFrame(scrollable_frame, text="Results")
        res.pack(fill=tk.BOTH, pady=8, expand=True)
        self.results_text = tk.Text(res, height=14, width=40)
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # Canvas
        self.canvas = tk.Canvas(right, bg="white")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)

        # Initialize visibility based on defaults
        self._update_settings_visibility()

    def _update_settings_visibility(self):
        # Rotation settings
        if self.var_rotation.get() == 1:
            if not self.frame_rotation_settings.winfo_ismapped():
                self.frame_rotation_settings.pack(fill=tk.X, pady=8)
        else:
            if self.frame_rotation_settings.winfo_ismapped():
                self.frame_rotation_settings.pack_forget()
        # Color settings
        if self.var_color.get() == 1:
            if not self.frame_color_settings.winfo_ismapped():
                self.frame_color_settings.pack(fill=tk.X, pady=8)
        else:
            if self.frame_color_settings.winfo_ismapped():
                self.frame_color_settings.pack_forget()

    def _on_thresh_change(self):
        self.template_threshold = float(self.slider_thresh.get())

    def _on_orb_change(self):
        self.orb_ratio = float(self.slider_orb.get())

    def _on_color_mode(self):
        self.color_use_hsv = (self.mode_var.get() == 1)
        if self.color_use_hsv:
            # Switch labels and ranges for HSV
            self.label_tol1.config(text="Hue tol")
            self.label_tol2.config(text="Sat tol")
            self.label_tol3.config(text="Val tol")
            self.slider_tol1.config(from_=0, to=90)
            self.slider_tol2.config(from_=0, to=127)
            self.slider_tol3.config(from_=0, to=127)
            self.slider_tol1.set(self.color_tol.hue)
            self.slider_tol2.set(self.color_tol.sat)
            self.slider_tol3.set(self.color_tol.val)
        else:
            # Switch labels and ranges for BGR
            self.label_tol1.config(text="Blue tol")
            self.label_tol2.config(text="Green tol")
            self.label_tol3.config(text="Red tol")
            self.slider_tol1.config(from_=0, to=255)
            self.slider_tol2.config(from_=0, to=255)
            self.slider_tol3.config(from_=0, to=255)
            self.slider_tol1.set(self.color_tol.b)
            self.slider_tol2.set(self.color_tol.g)
            self.slider_tol3.set(self.color_tol.r)
        # Live update
        self._update_color_preview()

    def _on_color_sliders_change(self):
        # Update tolerance values in realtime based on mode
        if self.color_use_hsv:
            self.color_tol.hue = int(self.slider_tol1.get())
            self.color_tol.sat = int(self.slider_tol2.get())
            self.color_tol.val = int(self.slider_tol3.get())
        else:
            self.color_tol.b = int(self.slider_tol1.get())
            self.color_tol.g = int(self.slider_tol2.get())
            self.color_tol.r = int(self.slider_tol3.get())
        # Live update
        self._update_color_preview()

    def _on_color_settings_change(self):
        # Called by kernel/min-area/ws controls
        self.color_kernel = int(self.slider_kernel.get())
        self.color_min_area = int(self.slider_minarea.get())
        self.color_ws_sens = int(self.slider_ws.get())
        self.color_use_hsv = (self.mode_var.get() == 1)
        # Live update
        self._update_color_preview()

    def _update_color_preview(self):
        # Show instant color-matching result on the canvas when settings change
        if self.original_bgr is None:
            return
        if self.selected_roi is None:
            return
        if self.var_color.get() != 1:
            return
        x1, y1, x2, y2 = self.selected_roi
        if x2 <= x1 or y2 <= y1:
            return
        roi = self.original_bgr[y1:y2, x1:x2]
        use_hsv = (self.mode_var.get() == 1)
        # Ensure tolerance values reflect sliders
        if use_hsv:
            tol = Tolerances(hue=int(self.slider_tol1.get()), sat=int(self.slider_tol2.get()), val=int(self.slider_tol3.get()),
                             r=self.color_tol.r, g=self.color_tol.g, b=self.color_tol.b)
        else:
            tol = Tolerances(hue=self.color_tol.hue, sat=self.color_tol.sat, val=self.color_tol.val,
                             r=int(self.slider_tol3.get()), g=int(self.slider_tol2.get()), b=int(self.slider_tol1.get()))
        kernel = int(self.slider_kernel.get())
        min_area = int(self.slider_minarea.get())
        roi_mean_bgr = get_most_frequent_color(roi, 'bgr')
        roi_mean_hsv = get_most_frequent_color(roi, 'hsv')
        mask = compute_mask(self.original_bgr, roi_mean_bgr, roi_mean_hsv, tol, use_hsv, kernel)
        if int(self.ws_var.get()) == 1:
            sens = int(self.slider_ws.get())
            mask = apply_watershed_split(mask, self.original_bgr, sens, kernel)
        num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        boxes = []
        for i in range(1, num):
            x, y, w, h, area = stats[i]
            if area >= min_area:
                boxes.append((int(x), int(y), int(w), int(h)))
        # Draw preview
        vis = cv2.cvtColor(self.original_bgr.copy(), cv2.COLOR_BGR2RGB)
        for (bx, by, bw, bh) in boxes:
            cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), (50, 180, 255), 2)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis, "Color preview", (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 180, 255), 1)
        self.display_rgb = vis
        self._display_image()

    # --------------- IO ---------------
    def load_image(self):
        path = filedialog.askopenfilename(title="Select Image",
                                          filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp *.avif")])
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            try:
                pil = Image.open(path)
                img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
            except Exception:
                messagebox.showerror("Error", "Failed to load image")
                return
        self.original_bgr = img
        self.display_rgb = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
        self.selected_roi = None
        self.detected_objects = []
        self.object_highlighted = False
        self._display_image()
        self._clear_results()
        # YOLO auto-detect
        self._auto_detect_yolo()

    def save_result(self):
        if self.display_rgb is None:
            messagebox.showwarning("Warning", "No result image to save")
            return
        save_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                 filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg;*.jpeg")])
        if not save_path:
            return
        bgr = cv2.cvtColor(self.display_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, bgr)
        messagebox.showinfo("Saved", f"Saved to: {save_path}")

    # --------------- Canvas ---------------
    def _display_image(self):
        if self.display_rgb is None:
            return
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw <= 1 or ch <= 1:
            self.root.after(100, self._display_image)
            return
        h, w = self.display_rgb.shape[:2]
        scale = min(cw / w, ch / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(self.display_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        pil = Image.fromarray(resized)
        self.photo = ImageTk.PhotoImage(pil)
        self.canvas.delete("all")
        self.canvas.create_image(cw // 2, ch // 2, image=self.photo, anchor=tk.CENTER)
        self.scale_factor = scale
        self.canvas_offset_x = (cw - new_w) // 2
        self.canvas_offset_y = (ch - new_h) // 2

        # If detections exist, overlay them
        if self.detected_objects:
            overlay = self._draw_on_rgb(self.display_rgb.copy(), boxes=self.detected_objects, selected=None)
            self.display_rgb = overlay

    def on_mouse_down(self, event):
        if self.original_bgr is None:
            return
        # Try selecting detected box
        if self.detected_objects:
            cx = (event.x - self.canvas_offset_x) / (self.scale_factor + 1e-6)
            cy = (event.y - self.canvas_offset_y) / (self.scale_factor + 1e-6)
            for i, (x, y, w, h, label, conf) in enumerate(self.detected_objects):
                if x <= cx <= x + w and y <= cy <= y + h:
                    self.selected_roi = (x, y, x + w, y + h)
                    self._highlight_selection(i)
                    return
        # Fallback to drawing ROI
        self.rect_start = (event.x, event.y)
        self.drawing = True
        self.canvas.delete("selection_rect")

    def on_mouse_drag(self, event):
        if not getattr(self, 'drawing', False) or not hasattr(self, 'rect_start') or self.rect_start is None:
            return
        self.rect_end = (event.x, event.y)
        self.canvas.delete("selection_rect")
        x1, y1 = self.rect_start
        x2, y2 = self.rect_end
        self.canvas.create_rectangle(x1, y1, x2, y2, outline="red", width=2, tags="selection_rect")

    def on_mouse_up(self, event):
        if not getattr(self, 'drawing', False):
            return
        self.drawing = False
        self.rect_end = (event.x, event.y)
        if self.rect_start and self.rect_end:
            x1 = (min(self.rect_start[0], self.rect_end[0]) - self.canvas_offset_x) / (self.scale_factor + 1e-6)
            y1 = (min(self.rect_start[1], self.rect_end[1]) - self.canvas_offset_y) / (self.scale_factor + 1e-6)
            x2 = (max(self.rect_start[0], self.rect_end[0]) - self.canvas_offset_x) / (self.scale_factor + 1e-6)
            y2 = (max(self.rect_start[1], self.rect_end[1]) - self.canvas_offset_y) / (self.scale_factor + 1e-6)
            x1 = max(0, int(round(x1)))
            y1 = max(0, int(round(y1)))
            x2 = min(self.display_rgb.shape[1], int(round(x2)))
            y2 = min(self.display_rgb.shape[0], int(round(y2)))
            if x2 > x1 and y2 > y1:
                self.selected_roi = (x1, y1, x2, y2)
                messagebox.showinfo("Info", f"ROI selected: {self.selected_roi}")

    def _highlight_selection(self, idx):
        img = cv2.cvtColor(self.original_bgr.copy(), cv2.COLOR_BGR2RGB)
        for i, (x, y, w, h, label, conf) in enumerate(self.detected_objects):
            if i == idx:
                color = (0, 255, 0)
                thickness = 3
                text = f"Reference: {label}" if label else "Reference"
            else:
                color = (255, 0, 255)
                thickness = 2
                text = label if label else "Object"
            cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)
            cv2.putText(img, text, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        self.display_rgb = img
        self._display_image()

    def _draw_on_rgb(self, rgb_img, boxes, selected=None):
        for i, (x, y, w, h, label, conf) in enumerate(boxes):
            color = (255, 0, 255)
            thickness = 2
            name = label if label else f"Object {i+1}"
            if selected is not None and i == selected:
                color = (0, 255, 0)
                thickness = 3
                name = f"Reference: {name}"
            cv2.rectangle(rgb_img, (x, y), (x + w, y + h), color, thickness)
            cv2.putText(rgb_img, name, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return rgb_img

    # --------------- YOLO ---------------
    def _ensure_yolo(self) -> bool:
        if not self.use_yolo:
            return False
        if self.yolo_model is not None:
            return True
        if YOLO is None:
            self.use_yolo = False
            return False
        try:
            self.yolo_model = YOLO("yolov8n.pt")
            return True
        except Exception:
            self.use_yolo = False
            self.yolo_model = None
            return False

    def _auto_detect_yolo(self):
        if not self._ensure_yolo() or self.original_bgr is None:
            return
        try:
            results = self.yolo_model(self.original_bgr, verbose=False)[0]
            boxes = []
            for b in results.boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                cls_id = int(b.cls[0]) if hasattr(b, 'cls') else -1
                label = results.names[cls_id] if hasattr(results, 'names') and cls_id in results.names else ''
                conf = float(b.conf[0]) if hasattr(b, 'conf') else 0.0
                boxes.append((x1, y1, x2 - x1, y2 - y1, label, conf))
            self.detected_objects = [d for d in boxes if d[5] >= 0.1]
            if self.detected_objects:
                self.display_rgb = self._draw_on_rgb(cv2.cvtColor(self.original_bgr.copy(), cv2.COLOR_BGR2RGB), self.detected_objects)
                self._display_image()
                messagebox.showinfo("Info", f"YOLO found {len(self.detected_objects)} objects. Click one to choose reference, or draw ROI.")
        except Exception as e:
            print(f"YOLO detection failed: {e}")

    # --------------- Methods ---------------
    def method_scaling(self, ref_area: int) -> List[Tuple[int, int, int, int]]:
        gray = cv2.cvtColor(self.original_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        white_ratio = float(np.mean(otsu == 255))
        if white_ratio > 0.5:
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
        cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)), iterations=1)
        bboxes = extract_bboxes_from_mask(cleaned)
        return bboxes

    def method_rotation(self, roi_bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
        boxes = []
        if roi_bgr is None or roi_bgr.size == 0:
            return boxes
        img_gray = cv2.cvtColor(self.original_bgr, cv2.COLOR_BGR2GRAY)
        roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

        # Try ORB
        try:
            orb = cv2.ORB_create(nfeatures=400)
            kp_roi, des_roi = orb.detectAndCompute(roi_gray, None)
            kp_img, des_img = orb.detectAndCompute(img_gray, None)
            if des_roi is not None and des_img is not None and len(kp_roi) >= 20 and DBSCAN is not None:
                index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=2)
                search_params = dict(checks=50)
                flann = cv2.FlannBasedMatcher(index_params, search_params)
                matches_knn = flann.knnMatch(des_roi, des_img, k=2)
                good = []
                for m_n in matches_knn:
                    if len(m_n) != 2:
                        continue
                    m, n = m_n
                    if m.distance < self.orb_ratio * n.distance:
                        good.append(m)
                if len(good) >= 8:
                    dst_pts = np.float32([kp_img[m.trainIdx].pt for m in good])
                    roi_diag = np.hypot(roi_bgr.shape[1], roi_bgr.shape[0])
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
                        if M is None or mask is None or mask.ravel().sum() < 10:
                            continue
                        h_roi, w_roi = roi_bgr.shape[:2]
                        pts = np.float32([[0, 0], [w_roi, 0], [w_roi, h_roi], [0, h_roi]]).reshape(-1, 1, 2)
                        dst = cv2.perspectiveTransform(pts, M)
                        x, y, w, h = cv2.boundingRect(np.int32(dst))
                        boxes.append((x, y, w, h))
        except Exception:
            pass

        # Fallback: template matching with rotations/scales
        rectangles = []
        angles = range(0, 360, 30)
        scales = [0.8, 1.0, 1.2]
        for angle in angles:
            h, w = roi_gray.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated_roi = cv2.warpAffine(roi_bgr, M, (w, h), flags=cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
            rotated_gray = cv2.cvtColor(rotated_roi, cv2.COLOR_BGR2GRAY)
            for scale in scales:
                sw = int(rotated_gray.shape[1] * scale)
                sh = int(rotated_gray.shape[0] * scale)
                if sw < 10 or sh < 10:
                    continue
                scaled_roi = cv2.resize(rotated_gray, (sw, sh))
                res = cv2.matchTemplate(img_gray, scaled_roi, cv2.TM_CCOEFF_NORMED)
                loc = np.where(res >= self.template_threshold)
                for pt in zip(*loc[::-1]):
                    rectangles.append([pt[0], pt[1], sw, sh])
        if rectangles:
            rectangles, _ = cv2.groupRectangles(rectangles, groupThreshold=1, eps=0.5)
            rectangles = nms_boxes(rectangles, iou_thresh=0.3)
            boxes.extend(rectangles)
        return boxes

    def method_color(self, roi_rect: Tuple[int, int, int, int]) -> List[Tuple[int, int, int, int]]:
        x1, y1, x2, y2 = roi_rect
        roi = self.original_bgr[y1:y2, x1:x2]
        tol = Tolerances(self.color_tol.hue, self.color_tol.sat, self.color_tol.val,
                         self.color_tol.r, self.color_tol.g, self.color_tol.b)
        kernel = int(self.slider_kernel.get())
        min_area = int(self.slider_minarea.get())
        use_hsv = (self.mode_var.get() == 1)
        roi_mean_bgr = get_most_frequent_color(roi, 'bgr')
        roi_mean_hsv = get_most_frequent_color(roi, 'hsv')
        mask = compute_mask(self.original_bgr, roi_mean_bgr, roi_mean_hsv, tol, use_hsv, kernel)
        if int(self.ws_var.get()) == 1:
            sens = int(self.slider_ws.get())
            mask = apply_watershed_split(mask, self.original_bgr, sens, kernel)
        num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        boxes = []
        for i in range(1, num):
            x, y, w, h, area = stats[i]
            if area >= min_area:
                boxes.append((int(x), int(y), int(w), int(h)))
        return boxes

    def method_gray(self, roi_rect: Tuple[int, int, int, int]) -> List[Tuple[int, int, int, int]]:
        x1, y1, x2, y2 = roi_rect
        ref_w = max(1, x2 - x1)
        ref_h = max(1, y2 - y1)
        ref_area = ref_w * ref_h
        gray = cv2.cvtColor(self.original_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        binary1 = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        _, binary2 = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        edges = cv2.Canny(blurred, 50, 150)
        combined = cv2.bitwise_or(binary1, binary2)
        combined = cv2.bitwise_or(combined, edges)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        ref_aspect = ref_h / ref_w if ref_w > 0 else 1.0
        for c in contours:
            area = cv2.contourArea(c)
            if 0.2 * ref_area < area < 5.0 * ref_area:
                x, y, w, h = cv2.boundingRect(c)
                aspect = h / (w + 1e-6)
                if 0.5 * ref_aspect < aspect < 2.0 * ref_aspect:
                    boxes.append((x, y, w, h))
        boxes = nms_boxes(boxes, iou_thresh=0.2)
        return boxes

    # --------------- Run ---------------
    def run_count(self):
        if self.original_bgr is None:
            messagebox.showwarning("Warning", "Please load an image first")
            return
        if self.selected_roi is None:
            messagebox.showwarning("Warning", "Please select a reference object (click detected box or draw ROI)")
            return
        x1, y1, x2, y2 = self.selected_roi
        if (x2 - x1) < 5 or (y2 - y1) < 5:
            messagebox.showwarning("Warning", "Selected ROI is too small")
            return
        roi_bgr = self.original_bgr[y1:y2, x1:x2]
        ref_area = (x2 - x1) * (y2 - y1)

        selected_methods = []
        if self.var_scaling.get() == 1:
            selected_methods.append("scaling")
        if self.var_rotation.get() == 1:
            selected_methods.append("rotation")
        if self.var_color.get() == 1:
            selected_methods.append("color")
        if self.var_gray.get() == 1:
            selected_methods.append("gray")
        if not selected_methods:
            messagebox.showwarning("Warning", "Please enable at least one method")
            return

        all_boxes = []
        method_counts = {}
        # Run methods
        if "scaling" in selected_methods:
            b = self.method_scaling(ref_area)
            method_counts['scaling'] = len(b)
            all_boxes.extend(b)
        if "rotation" in selected_methods:
            b = self.method_rotation(roi_bgr)
            method_counts['rotation'] = len(b)
            all_boxes.extend(b)
        if "color" in selected_methods:
            b = self.method_color((x1, y1, x2, y2))
            method_counts['color'] = len(b)
            all_boxes.extend(b)
        if "gray" in selected_methods:
            b = self.method_gray((x1, y1, x2, y2))
            method_counts['gray'] = len(b)
            all_boxes.extend(b)

        # Merge results
        merged = nms_boxes(all_boxes, iou_thresh=0.3)
        total = len(merged)

        # Draw
        vis = cv2.cvtColor(self.original_bgr.copy(), cv2.COLOR_BGR2RGB)
        for (x, y, w, h) in merged:
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 200, 255), 2)
        # draw ref
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis, "Reference", (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        self.display_rgb = vis
        self._display_image()

        # Results text
        self.results_text.delete(1.0, tk.END)
        lines = [f"Method counts: {method_counts}", f"Final merged total: {total}"]
        self.results_text.insert(tk.END, "\n".join(lines))

    # --------------- Utils ---------------
    def _clear_results(self):
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "No results yet. Load image and select reference.\n")

    def reset(self):
        self.original_bgr = None
        self.display_rgb = None
        self.photo = None
        self.detected_objects = []
        self.selected_roi = None
        self.object_highlighted = False
        self.canvas.delete("all")
        self._clear_results()


def main():
    root = tk.Tk()
    app = UnifiedObjectCounter(root)

    def on_resize(event):
        if app.display_rgb is not None:
            app._display_image()
    root.bind("<Configure>", on_resize)
    root.mainloop()


if __name__ == "__main__":
    main()
