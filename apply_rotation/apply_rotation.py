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

        # settings
        self.template_threshold = 0.7  # default for template matching
        self.orb_ratio = 0.75  # default Lowe’s ratio threshold
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

        # Template Matching Threshold Slider
        threshold_frame = ttk.LabelFrame(left_panel, text="Template Matching Threshold")
        threshold_frame.pack(fill=tk.X, pady=10)
        self.threshold_slider = tk.Scale(threshold_frame, from_=0.5, to=0.95, resolution=0.01,
                                         orient=tk.HORIZONTAL, command=lambda v: self.update_threshold())
        self.threshold_slider.set(self.template_threshold)
        self.threshold_slider.pack(fill=tk.X, padx=6, pady=6)

        # ORB ratio slider
        ratio_frame = ttk.LabelFrame(left_panel, text="ORB Ratio Threshold")
        ratio_frame.pack(fill=tk.X, pady=10)
        self.ratio_slider = tk.Scale(ratio_frame, from_=0.6, to=0.9, resolution=0.01, 
                                     orient=tk.HORIZONTAL, command=lambda val: self.update_orb_ratio(val))
        self.ratio_slider.set(self.orb_ratio)
        self.ratio_slider.pack(fill=tk.X, padx=6, pady=6)

        # Instructions
        instruction_frame = ttk.LabelFrame(left_panel, text="Instructions")
        instruction_frame.pack(fill=tk.X, pady=10)
        instructions = """
1. Load an image 
2. Drag a box to select reference object
3. Adjust threshold if needed
4. Click "Count Objects" to analyze
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

    def update_threshold(self):
        self.template_threshold = float(self.threshold_slider.get())

    def update_orb_ratio(self, val):
        self.orb_ratio = float(self.ratio_slider.get())

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
            # Convert original image and ROI to grayscale
            img_gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

            rectangles = []
            angles = range(0, 360, 30)
            scales = [0.8, 1.0, 1.2]

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
                    loc = np.where(res >= self.template_threshold)

                    for pt in zip(*loc[::-1]):
                        rect = [pt[0], pt[1], scaled_roi_gray.shape[1], scaled_roi_gray.shape[0]]
                        rectangles.append(rect)

            # Group & suppress overlapping rectangles
            if rectangles:
                rectangles, _ = cv2.groupRectangles(rectangles, groupThreshold=1, eps=0.5)
                rectangles = self.non_max_suppression(rectangles, overlapThresh=0.3)
            else:
                rectangles = []

            # Draw results
            vis_bgr = self.original_image.copy()
            for (x, y, w, h) in rectangles:
                cv2.rectangle(vis_bgr, (x, y), (x + w, y + h), (255, 255, 150), 2)

            self.display_image = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)
            self.display_image_on_canvas()
            results_info = (f"Method used: {method_used}\n"
                            f"Objects Found: {len(rectangles)}\n"
                            f"Threshold: {self.template_threshold:.2f}\n")

        else:
            # ORB-based multi-instance detection (FLANN + ratio test)
            orb = cv2.ORB_create(nfeatures=400)
            roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
            img_gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
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

            # Lowe’s ratio test
            good = []
            for m_n in matches_knn:
                if len(m_n) != 2:
                    continue
                m, n = m_n
                if m.distance < self.orb_ratio * n.distance:
                    good.append(m)

            objects_found = 0
            if len(good) >= 8:
                dst_pts = np.float32([kp_img[m.trainIdx].pt for m in good])

                # cluster matches relative to ROI size (not whole image)
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
                    if M is not None and mask is not None:
                        inliers = mask.ravel().sum()
                        if inliers < 10:  # reject weak homographies
                            continue

                        h_roi, w_roi = roi_bgr.shape[:2]
                        pts = np.float32([[0,0],[w_roi,0],[w_roi,h_roi],[0,h_roi]]).reshape(-1,1,2)
                        dst = cv2.perspectiveTransform(pts, M)

                        # draw bounding rectangle instead of polygon
                        x, y, w, h = cv2.boundingRect(np.int32(dst))
                        cv2.rectangle(vis_bgr, (x, y), (x+w, y+h), (0,255,0), 2)
                        objects_found += 1

            self.display_image = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)
            self.display_image_on_canvas()
            results_info = (f"Method used: ORB (FLANN)\n"
                            f"Keypoints in ROI: {len(kp_roi) if kp_roi is not None else 0}\n"
                            f"Good Matches: {len(good)}\n"
                            f"Objects Found: {objects_found}\n"
                            f"Ratio Threshold: {self.orb_ratio:.2f}\n")

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
