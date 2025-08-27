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
        low, high = clamp_hsv_range(roi_mean_hsv, tol)
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
    cv2.putText(vis, f"Count: {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (36, 255, 12), 2, cv2.LINE_AA)
    return vis, count


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
        self._preview_after_id = None

        # precomputed helpers
        self.gray_image = None
        self.binary_image = None

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
            "2) App highlights objects; click one to set as reference, or drag a box to pick ROI\n"
            "3) Adjust sliders (HSV/BGR)\n"
            "4) Click 'Count Objects'"
        )
        ttk.Label(instruction_frame, text=instructions, justify=tk.LEFT).pack(padx=10, pady=8)

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

        self.clear_results()

    # ----------------------- Image / ROI handlers -----------------------
    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp")]
        )
        if not file_path:
            return
        img = cv2.imread(file_path)
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
        if self.object_highlighted and self.detected_objects:
            canvas_x = (event.x - self.canvas_offset_x) / self.scale_factor
            canvas_y = (event.y - self.canvas_offset_y) / self.scale_factor
            for i, (x, y, w, h) in enumerate(self.detected_objects):
                if x <= canvas_x <= x + w and y <= canvas_y <= y + h:
                    self.selected_roi = (x, y, x + w, y + h)
                    self.highlight_selected_object(i)
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
            x1 = max(0, int(x1)); y1 = max(0, int(y1))
            x2 = min(self.image.shape[1], int(x2)); y2 = min(self.image.shape[0], int(y2))
            if x2 > x1 and y2 > y1:
                self.selected_roi = (x1, y1, x2, y2)
                roi = self.original_image[y1:y2, x1:x2]
                self.roi_mean_bgr = tuple(map(int, np.mean(roi.reshape(-1, 3), axis=0)))
                hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                self.roi_mean_hsv = tuple(map(int, np.mean(hsv.reshape(-1, 3), axis=0)))
                messagebox.showinfo("Info", f"ROI selected: {self.selected_roi}")

    # ----------------------- Auto-detect & highlight -----------------------
    def auto_detect_objects(self):
        if self.gray_image is None:
            return
        blurred = cv2.GaussianBlur(self.gray_image, (5, 5), 0)
        binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_area = 500
        max_area = 30000
        self.detected_objects = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = h / w if w > 0 else 0
                if 0.3 < aspect_ratio < 3.0:
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                        if 0.2 < circularity < 0.9:
                            rect_area = w * h
                            if rect_area > 0 and area / rect_area > 0.3:
                                self.detected_objects.append((x, y, w, h))

        if not self.detected_objects:
            messagebox.showinfo("Info", "No objects detected in the image.")
            return
        self.highlight_detected_objects()
        self.object_highlighted = True
        messagebox.showinfo("Info", f"Found {len(self.detected_objects)} objects. Click one to use as reference.")

    def highlight_detected_objects(self):
        if not self.detected_objects:
            return
        highlight_image = self.image.copy()
        for i, (x, y, w, h) in enumerate(self.detected_objects):
            color = (255, 0, 255)
            cv2.rectangle(highlight_image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(highlight_image, f"Object {i+1}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        self.display_image = highlight_image
        self.display_image_on_canvas()

    def highlight_selected_object(self, selected_index: int):
        if not self.detected_objects:
            return
        highlight_image = self.image.copy()
        for i, (x, y, w, h) in enumerate(self.detected_objects):
            if i == selected_index:
                color = (0, 255, 0); thickness = 3
                cv2.putText(highlight_image, f"Reference (Object {i+1})", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            else:
                color = (255, 0, 255); thickness = 2
                cv2.putText(highlight_image, f"Object {i+1}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.rectangle(highlight_image, (x, y), (x + w, y + h), color, thickness)
        self.display_image = highlight_image
        self.display_image_on_canvas()

    # ----------------------- Results & counting -----------------------
    def clear_results(self):
        self.results = {}
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "No results yet. Load an image and select a reference object.\n")

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
            roi_mean_bgr = tuple(map(int, np.mean(roi.reshape(-1, 3), axis=0)))
            roi_mean_hsv = tuple(map(int, np.mean(cv2.cvtColor(roi, cv2.COLOR_BGR2HSV).reshape(-1, 3), axis=0)))
            mask = compute_mask(self.original_image, roi_mean_bgr, roi_mean_hsv,
                                self.tol, self.use_hsv, self.kernel)

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
                            "Tip: Click a highlighted object or drag to select an ROI.")
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
            
            results_text = (f"Reference Object Area: {self.results['reference_area']:.1f} px²\n"
                            f"Objects Detected : {len(self.results['objects'])}\n\n"
                            f"Mode: {'HSV' if self.use_hsv else 'BGR'}\n"
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
    def schedule_preview(self, delay: int = 120):
        try:
            if self._preview_after_id is not None:
                self.root.after_cancel(self._preview_after_id)
        except Exception:
            pass
        self._preview_after_id = self.root.after(delay, self.update_preview)

    def on_slider_change(self):
        """Called when any slider changes - updates preview and results"""
        # Update tolerance values based on current mode
        if self.use_hsv:
            self.tol.hue = int(self.hue_slider.get())
            self.tol.sat = int(self.sat_slider.get())
            self.tol.val = int(self.val_slider.get())
        else:
            self.tol.b = int(self.hue_slider.get())
            self.tol.g = int(self.sat_slider.get())
            self.tol.r = int(self.val_slider.get())
        
        self.schedule_preview(10)
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
        
        self.schedule_preview(10)
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
        roi_mean_bgr = tuple(map(int, np.mean(roi.reshape(-1, 3), axis=0)))
        roi_mean_hsv = tuple(map(int, np.mean(cv2.cvtColor(roi, cv2.COLOR_BGR2HSV).reshape(-1, 3), axis=0)))
        mask = compute_mask(self.original_image, roi_mean_bgr, roi_mean_hsv,
                            self.tol, self.use_hsv, self.kernel)
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

        self.canvas.delete("all")
        self.clear_results()

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