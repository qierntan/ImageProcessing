# object_counter.py
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np
from PIL import Image, ImageTk


# ----------------------- Utilities -----------------------
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
    # FIXED: previously upper used an undefined v / tol.val
    b, g, r = center
    lower = np.array([max(0, b - tol.b), max(0, g - tol.g), max(0, r - tol.r)], dtype=np.uint8)
    upper = np.array([min(255, b + tol.b), min(255, g + tol.g), min(255, r + tol.r)], dtype=np.uint8)
    return lower, upper


def get_most_frequent_color(roi: np.ndarray, color_space: str = 'bgr') -> Tuple[int, int, int]:
    """Find the most frequent color in the ROI, excluding white/background colors.
    This helps avoid background color interference when ROI is too large.
    
    Args:
        roi: Region of interest image array
        color_space: 'bgr' or 'hsv'
    
    Returns:
        Most frequent non-background color as (channel1, channel2, channel3)
    """
    if roi.size == 0:
        return (0, 0, 0)
    
    # Convert to the desired color space if needed
    if color_space == 'hsv':
        roi_converted = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    else:
        roi_converted = roi.copy()
    
    # Reshape to get all pixels
    pixels = roi_converted.reshape(-1, 3)
    
    # For better performance with large ROIs, we can sample pixels
    if len(pixels) > 10000:  # If ROI is very large, sample pixels
        step = len(pixels) // 5000
        pixels = pixels[::step]
    
    # Filter out background colors (white, very light colors)
    if color_space == 'hsv':
        # In HSV: filter out pixels with very low saturation (grayish/white)
        # and very high value (bright/white)
        mask = (pixels[:, 1] > 30) & (pixels[:, 2] < 240)  # S > 30, V < 240
        filtered_pixels = pixels[mask]
    else:
        # In BGR: filter out pixels that are too bright (white-ish)
        brightness = np.mean(pixels, axis=1)
        mask = brightness < 200  # Exclude very bright pixels
        filtered_pixels = pixels[mask]
    
    # If we filtered out too many pixels, fall back to original
    if len(filtered_pixels) < len(pixels) * 0.1:  # Less than 10% remaining
        filtered_pixels = pixels
    
    # Find unique colors and their counts
    unique_colors, counts = np.unique(filtered_pixels, axis=0, return_counts=True)
    
    # Get the most frequent color
    most_frequent_idx = np.argmax(counts)
    most_frequent_color = unique_colors[most_frequent_idx]
    
    return tuple(map(int, most_frequent_color))


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
        # Handle hue wrap-around for colors like red near 0/179
        h_low = h - tol.hue
        h_high = h + tol.hue
        s_low = max(0, s - tol.sat)
        s_high = min(255, s + tol.sat)
        v_low = max(0, v - tol.val)
        v_high = min(255, v + tol.val)
        if h_low < 0 or h_high > 179:
            # Split into two ranges around the circular hue
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


# ----------------------- Segmentation helpers -----------------------
def apply_watershed_split(mask: np.ndarray, img_bgr: np.ndarray, sensitivity: int, morph_kernel: int) -> np.ndarray:
    """Split touching objects in a binary mask using watershed.

    sensitivity: 1..80 roughly controls foreground threshold; higher splits more.
    morph_kernel: reuse UI kernel size to build structuring element.
    """
    if mask is None or img_bgr is None:
        return mask
    # Ensure binary mask 0/255
    bin_mask = (mask > 0).astype(np.uint8) * 255

    ksize = max(3, morph_kernel | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    # Sure background
    sure_bg = cv2.dilate(bin_mask, kernel, iterations=1)
    # Distance transform for sure foreground
    dist = cv2.distanceTransform(bin_mask, cv2.DIST_L2, 5)
    if dist.max() > 0:
        dist_norm = dist / (dist.max() + 1e-6)
    else:
        dist_norm = dist
    # Map sensitivity (1..80) to threshold 0.2..0.7
    thr = 0.2 + (min(max(sensitivity, 1), 80) / 80.0) * (0.7 - 0.2)
    sure_fg = (dist_norm > thr).astype(np.uint8) * 255
    # Unknown region
    unknown = cv2.subtract(sure_bg, sure_fg)
    # Markers
    num_markers, markers = cv2.connectedComponents((sure_fg > 0).astype(np.uint8))
    markers = markers + 1
    markers[unknown > 0] = 0
    # Watershed expects 3-channel BGR image
    ws_markers = cv2.watershed(img_bgr.copy(), markers)
    # Build refined mask: labels > 1 are objects; boundary is -1
    refined = np.zeros_like(bin_mask)
    refined[ws_markers > 1] = 255
    return refined


# ----------------------- GUI -----------------------
class ObjectCounterGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Smart Object Counter")
        self.root.geometry("1200x800")

        # images/state
        self.image = None                 # RGB for display
        self.original_image = None        # BGR as loaded by cv2
        self.display_image = None
        self.photo = None
        self.scale_factor = 1.0
        self.canvas_offset_x = 0
        self.canvas_offset_y = 0

        # ROI/detection
        self.selected_roi = None          # (x1, y1, x2, y2)
        self.detected_objects = []        # list of (x,y,w,h)
        self.object_highlighted = False

        # processing params and UI defaults
        self.tol = Tolerances()
        self.kernel = 3
        self.min_area = 100
        self.use_hsv = True               # default mode

        # precomputed helpers
        self.gray_image = None
        self.binary_image = None

        # watershed options
        self.use_watershed = False
        self.ws_sensitivity = 35

        self.setup_gui()

    def setup_gui(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create a frame for the left panel with fixed width
        left_panel_container = ttk.Frame(main_frame, width=320)
        left_panel_container.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_panel_container.pack_propagate(False)  # Prevent the frame from shrinking
        left_panel_container.configure(width=320)  # Ensure fixed width

        # Create a canvas and scrollbar for the left panel
        canvas = tk.Canvas(left_panel_container, width=300)
        scrollbar = ttk.Scrollbar(left_panel_container, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas, width=300)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw", width=300)
        canvas.configure(yscrollcommand=scrollbar.set)

        # Pack the canvas and scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Bind mouse wheel to scroll
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Controls
        btn_frame = ttk.Frame(scrollable_frame)
        btn_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(btn_frame, text="Load Image", command=self.load_image).pack(fill=tk.X, pady=4)
        ttk.Button(btn_frame, text="Count Objects", command=self.count_objects).pack(fill=tk.X, pady=4)
        ttk.Button(btn_frame, text="Save Result", command=self.save_result).pack(fill=tk.X, pady=4)
        ttk.Button(btn_frame, text="Reset", command=self.reset).pack(fill=tk.X, pady=4)

        instruction_frame = ttk.LabelFrame(scrollable_frame, text="Instructions")
        instruction_frame.pack(fill=tk.X, pady=10)
        instructions = (
            "1) Load an image\n"
            "2) Drag a box to select ROI (reference object)\n"
            "3) Adjust sliders (HSV/BGR)\n"
            "4) Click 'Count Objects'"
        )
        ttk.Label(instruction_frame, text=instructions, justify=tk.LEFT).pack(padx=10, pady=8)

        # ROI preview panel
        roi_frame = ttk.LabelFrame(scrollable_frame, text="Selected ROI")
        roi_frame.pack(fill=tk.X, pady=10)
        self.roi_preview_size = 150
        self.roi_preview_label = tk.Label(roi_frame, width=self.roi_preview_size, height=self.roi_preview_size, bg="#f0f0f0")
        self.roi_preview_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.roi_photo = None

        results_frame = ttk.LabelFrame(scrollable_frame, text="Results")
        results_frame.pack(fill=tk.X, pady=10)
        self.results_text = tk.Text(results_frame, height=8, width=36)
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Image canvas
        self.canvas = tk.Canvas(right_panel, bg="white")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)

        # Sliders + mode
        sliders = ttk.LabelFrame(scrollable_frame, text="Detection Settings")
        sliders.pack(fill=tk.X, pady=10)

        # Create labels for sliders (will be updated based on mode)
        self.hue_label = ttk.Label(sliders, text="Hue tol")
        self.hue_label.pack(anchor=tk.W, padx=6)
        self.hue_slider = tk.Scale(sliders, from_=0, to=90, orient=tk.HORIZONTAL,
                                   command=lambda v: self.on_slider_change())
        self.hue_slider.set(self.tol.hue)
        self.hue_slider.pack(fill=tk.X)

        self.sat_label = ttk.Label(sliders, text="Sat tol")
        self.sat_label.pack(anchor=tk.W, padx=6)
        self.sat_slider = tk.Scale(sliders, from_=0, to=127, orient=tk.HORIZONTAL,
                                   command=lambda v: self.on_slider_change())
        self.sat_slider.set(self.tol.sat)
        self.sat_slider.pack(fill=tk.X)

        self.val_label = ttk.Label(sliders, text="Val tol")
        self.val_label.pack(anchor=tk.W, padx=6)
        self.val_slider = tk.Scale(sliders, from_=0, to=127, orient=tk.HORIZONTAL,
                                   command=lambda v: self.on_slider_change())
        self.val_slider.set(self.tol.val)
        self.val_slider.pack(fill=tk.X)

        ttk.Label(sliders, text="Morph kernel").pack(anchor=tk.W, padx=6)
        self.kernel_slider = tk.Scale(sliders, from_=0, to=25, orient=tk.HORIZONTAL,
                                      command=lambda v: self.on_slider_change())
        self.kernel_slider.set(self.kernel)
        self.kernel_slider.pack(fill=tk.X)

        ttk.Label(sliders, text="Min area").pack(anchor=tk.W, padx=6)
        self.min_slider = tk.Scale(sliders, from_=1, to=20000, orient=tk.HORIZONTAL,
                                   command=lambda v: self.on_slider_change())
        self.min_slider.set(self.min_area)
        self.min_slider.pack(fill=tk.X)

        mode_frame = ttk.LabelFrame(scrollable_frame, text="Color Space")
        mode_frame.pack(fill=tk.X, pady=10)
        self.mode_var = tk.IntVar(value=1)  # 1 = HSV, 0 = BGR
        ttk.Radiobutton(mode_frame, text="HSV", variable=self.mode_var, value=1).pack(anchor=tk.W)
        ttk.Radiobutton(mode_frame, text="BGR", variable=self.mode_var, value=0).pack(anchor=tk.W)
        try:
            self.mode_var.trace_add("write", lambda *a: self.on_mode_change())
        except Exception:
            self.mode_var.trace("w", lambda *a: self.on_mode_change())

        # Watershed refinement controls
        ws_frame = ttk.LabelFrame(scrollable_frame, text="Segmentation Refinement")
        ws_frame.pack(fill=tk.X, pady=10)
        self.ws_var = tk.IntVar(value=0)
        def _on_ws_toggle():
            self.use_watershed = (self.ws_var.get() == 1)
            self.update_preview()
            if hasattr(self, 'results') and self.results:
                self.display_results()
        ttk.Checkbutton(ws_frame, text="Split touching objects (watershed)", variable=self.ws_var,
                        command=_on_ws_toggle).pack(anchor=tk.W)
        ttk.Label(ws_frame, text="Sensitivity").pack(anchor=tk.W, padx=6)
        self.ws_slider = tk.Scale(ws_frame, from_=1, to=80, orient=tk.HORIZONTAL,
                                  command=lambda v: self.on_ws_slider_change())
        self.ws_slider.set(self.ws_sensitivity)
        self.ws_slider.pack(fill=tk.X)

        # CLAHE functionality removed as requested

        self.clear_results()

    # ----------------------- Image / ROI handlers -----------------------
    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp *.avif")]
        )
        if not file_path:
            return
        
        # Try OpenCV first
        img = cv2.imread(file_path)
        
        # If OpenCV fails and it's an AVIF file, try Pillow
        if img is None and file_path.lower().endswith('.avif'):
            try:
                from PIL import Image
                pil_img = Image.open(file_path)
                img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load AVIF image: {str(e)}")
                return
        
        if img is None:
            messagebox.showerror("Error", "Failed to load image")
            return

        self.original_image = img
        self.image = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
        self.display_image = self.image.copy()

        self.gray_image = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        _, self.binary_image = cv2.threshold(self.gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        self.selected_roi = None
        self.detected_objects = []
        self.object_highlighted = False

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
        self.rect_start = (event.x, event.y)
        self.drawing = True
        self.canvas.delete("selection_rect")

    def on_mouse_drag(self, event):
        if not getattr(self, "drawing", False) or not hasattr(self, "rect_start") or self.rect_start is None:
            return
        self.rect_end = (event.x, event.y)
        self.canvas.delete("selection_rect")
        x1, y1 = self.rect_start
        x2, y2 = self.rect_end
        self.canvas.create_rectangle(x1, y1, x2, y2, outline="green", width=2, tags="selection_rect")

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
            x1 = max(0, int(x1)); y1 = max(0, int(y1))
            x2 = min(self.image.shape[1], int(x2)); y2 = min(self.image.shape[0], int(y2))
            if x2 > x1 and y2 > y1:
                self.selected_roi = (x1, y1, x2, y2)
                roi = self.original_image[y1:y2, x1:x2]
                # Use most frequent color instead of mean to avoid background interference
                self.roi_mean_bgr = get_most_frequent_color(roi, 'bgr')
                self.roi_mean_hsv = get_most_frequent_color(roi, 'hsv')
                messagebox.showinfo("Info", f"ROI selected: {self.selected_roi}")
                self.update_roi_preview()

    # ----------------------- Manual ROI selection only -----------------------

    # ----------------------- Results & counting -----------------------
    def clear_results(self):
        self.results = {}
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "No results yet. Load an image and select a reference object.\n")
        # Clear ROI preview
        if hasattr(self, 'roi_preview_label'):
            self.roi_preview_label.config(image="", text="(no ROI)")
            self.roi_photo = None

    def count_objects(self):
        if self.image is None:
            messagebox.showwarning("Warning", "Please load an image first")
            return
        if self.selected_roi is None:
            self.results = {'objects': [], 'reference_area': 0, 'no_reference_selected': True}
            self.display_results()
            return
        try:
            x1, y1, x2, y2 = self.selected_roi
            if (x2 - x1) < 5 or (y2 - y1) < 5:
                self.results = {'objects': [], 'reference_area': 0, 'no_objects_found': True}
                self.display_results()
                self.draw_results_on_image()
                return

            # read sliders based on current mode
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
            self.use_hsv = (self.mode_var.get() == 1)

            # compute mask using original BGR image
            roi = self.original_image[y1:y2, x1:x2]
            # Use most frequent color instead of mean to avoid background interference
            roi_mean_bgr = get_most_frequent_color(roi, 'bgr')
            roi_mean_hsv = get_most_frequent_color(roi, 'hsv')
            mask = compute_mask(self.original_image, roi_mean_bgr, roi_mean_hsv,
                                self.tol, self.use_hsv, self.kernel)

            # Optional watershed split
            if self.use_watershed:
                self.ws_sensitivity = int(self.ws_slider.get())
                mask = apply_watershed_split(mask, self.original_image, self.ws_sensitivity, self.kernel)

            vis, _ = count_and_draw(mask, self.image, self.min_area)

            # extract object boxes
            num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
            objects = []
            for i in range(1, num):
                x, y, w, h, area = stats[i]
                if area >= self.min_area:
                    objects.append((x, y, w, h, area))
            self.results = {'objects': objects, 'reference_area': (x2 - x1) * (y2 - y1), 'no_objects_found': False}
            self.display_results()

            # show annotated
            self.display_image = vis
            self.display_image_on_canvas()
        except Exception as e:
            messagebox.showerror("Error", f"Error counting objects: {str(e)}")

    def display_results(self):
        self.results_text.delete(1.0, tk.END)
        if not self.results:
            self.results_text.insert(tk.END, "No results available.\n")
            return
        if self.results.get('no_reference_selected', False):
            results_text = ("No reference object selected.\n\n"
                            "Objects Detected : 0\n\n"
                            "Tip: Drag to select an ROI (reference object).")
        elif self.results.get('no_objects_found', False):
            results_text = ("No objects found in the image.\n\n"
                            "Objects Detected : 0\n\n"
                            "Tip: Ensure you selected a valid object area.")
        else:
            # Show appropriate values based on current mode
            if self.use_hsv:
                mode_info = f"H:{self.tol.hue}  S:{self.tol.sat}  V:{self.tol.val}"
            else:
                mode_info = f"B:{self.tol.b}  G:{self.tol.g}  R:{self.tol.r}"
            
            ws_info = f"  WS:{'on' if self.use_watershed else 'off'}"
            if self.use_watershed:
                ws_info += f"({int(self.ws_slider.get())})"
            results_text = (f"Reference Object Area: {self.results['reference_area']:.1f} px²\n"
                            f"Objects Detected : {len(self.results['objects'])}\n\n"
                            f"Mode: {'HSV' if self.use_hsv else 'BGR'}{ws_info}\n"
                            f"{mode_info}\n"
                            f"Kernel:{self.kernel}  MinArea:{self.min_area}")
        self.results_text.insert(tk.END, results_text)

    def draw_results_on_image(self):
        if not self.results:
            return
        result_image = self.image.copy()
        if self.selected_roi and not self.results.get('no_objects_found', False):
            x1, y1, x2, y2 = self.selected_roi
            cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(result_image, "Reference", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        for x, y, w, h, area in self.results.get('objects', []):
            cv2.rectangle(result_image, (x, y), (x + w, y + h), (255, 0, 0), 1)
            cv2.putText(result_image, "O", (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
        self.display_image = result_image
        self.display_image_on_canvas()

    # ----------------------- Preview helpers -----------------------

    def on_slider_change(self):
        """Called when any slider changes - updates preview and results instantly"""
        # Update tolerance values based on current mode
        if self.use_hsv:
            self.tol.hue = int(self.hue_slider.get())
            self.tol.sat = int(self.sat_slider.get())
            self.tol.val = int(self.val_slider.get())
        else:
            self.tol.b = int(self.hue_slider.get())
            self.tol.g = int(self.sat_slider.get())
            self.tol.r = int(self.val_slider.get())
        
        self.update_preview()
        # Update results display immediately when sliders change
        if hasattr(self, 'results') and self.results:
            self.display_results()

    def on_mode_change(self):
        self.use_hsv = (self.mode_var.get() == 1)
        
        # Update slider labels and ranges based on mode
        if self.use_hsv:
            # HSV mode
            self.hue_label.config(text="Hue tol")
            self.sat_label.config(text="Sat tol")
            self.val_label.config(text="Val tol")
            
            # Update slider ranges for HSV
            self.hue_slider.config(from_=0, to=90)
            self.sat_slider.config(from_=0, to=127)
            self.val_slider.config(from_=0, to=127)
            
            # Set current values (convert from BGR if needed)
            self.hue_slider.set(self.tol.hue)
            self.sat_slider.set(self.tol.sat)
            self.val_slider.set(self.tol.val)
        else:
            # BGR mode
            self.hue_label.config(text="Blue tol")
            self.sat_label.config(text="Green tol")
            self.val_label.config(text="Red tol")
            
            # Update slider ranges for BGR
            self.hue_slider.config(from_=0, to=255)
            self.sat_slider.config(from_=0, to=255)
            self.val_slider.config(from_=0, to=255)
            
            # Set current values (convert from HSV if needed)
            self.hue_slider.set(self.tol.b)
            self.sat_slider.set(self.tol.g)
            self.val_slider.set(self.tol.r)
        
        self.update_preview()
        # Update results display immediately when mode changes
        if hasattr(self, 'results') and self.results:
            self.display_results()

    def update_preview(self):
        if self.original_image is None or self.selected_roi is None:
            return
        # read sliders based on current mode
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
        self.use_hsv = (self.mode_var.get() == 1)

        x1, y1, x2, y2 = self.selected_roi
        roi = self.original_image[y1:y2, x1:x2]
        # Use most frequent color instead of mean to avoid background interference
        roi_mean_bgr = get_most_frequent_color(roi, 'bgr')
        roi_mean_hsv = get_most_frequent_color(roi, 'hsv')
        mask = compute_mask(self.original_image, roi_mean_bgr, roi_mean_hsv,
                            self.tol, self.use_hsv, self.kernel)
        # Optional watershed split for preview
        if self.use_watershed:
            self.ws_sensitivity = int(self.ws_slider.get())
            mask = apply_watershed_split(mask, self.original_image, self.ws_sensitivity, self.kernel)
        vis, count = count_and_draw(mask, self.image, max(1, self.min_area))
        
        # Update results with the current detection
        num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        objects = []
        for i in range(1, num):
            x, y, w, h, area = stats[i]
            if area >= self.min_area:
                objects.append((x, y, w, h, area))
        self.results = {'objects': objects, 'reference_area': (x2 - x1) * (y2 - y1), 'no_objects_found': False}
        
        self.display_image = vis
        self.display_image_on_canvas()

    def on_ws_slider_change(self):
        self.ws_sensitivity = int(self.ws_slider.get())
        self.update_preview()
        if hasattr(self, 'results') and self.results:
            self.display_results()



    # ----------------------- Save & Reset -----------------------
    def save_result(self):
        if self.display_image is None:
            messagebox.showwarning("Warning", "Nothing to save yet.")
            return
        path = filedialog.asksaveasfilename(defaultextension='.png',
                                            filetypes=[('PNG', '*.png'), ('JPEG', '*.jpg;*.jpeg')])
        if not path:
            return
        bgr = cv2.cvtColor(self.display_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, bgr)
        messagebox.showinfo("Saved", f"Saved to: {path}")

    def reset(self):
        # Clear state
        self.image = None
        self.original_image = None
        self.display_image = None
        self.photo = None
        self.selected_roi = None
        self.detected_objects = []
        self.object_highlighted = False
        self.gray_image = None
        self.binary_image = None

        # Reset UI elements back to defaults (first-launch look)
        self.mode_var.set(1)     # HSV
        self.hue_slider.set(15)
        self.sat_slider.set(60)
        self.val_slider.set(60)
        self.kernel_slider.set(3)
        self.min_slider.set(100)

        # Watershed
        if hasattr(self, 'ws_var'):
            self.ws_var.set(0)
        if hasattr(self, 'ws_slider'):
            self.ws_slider.set(35)

        self.canvas.delete("all")
        self.clear_results()

    def update_roi_preview(self):
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
        # Paste centered on a gray canvas
        canvas = np.full((size, size, 3), 240, dtype=np.uint8)
        y_off = (size - new_h) // 2
        x_off = (size - new_w) // 2
        canvas[y_off:y_off+new_h, x_off:x_off+new_w] = roi_resized
        pil = Image.fromarray(canvas)
        self.roi_photo = ImageTk.PhotoImage(pil)
        self.roi_preview_label.config(image=self.roi_photo, text="")

    # ----------------------- End class -----------------------


def main():
    root = tk.Tk()
    app = ObjectCounterGUI(root)

    def on_resize(event):
        if app.image is not None:
            app.display_image_on_canvas()

    root.bind("<Configure>", on_resize)
    root.mainloop()


if __name__ == "__main__":
    main()