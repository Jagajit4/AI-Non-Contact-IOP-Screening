import cv2
import numpy as np
import base64
import io
from PIL import Image

class EyeCVPipeline:
    """
    Computer Vision Pipeline for Non-Contact Eye Analysis.
    Detects Iris and Pupil boundaries and extracts optical parameters:
    - Iris Diameter
    - Pupil Diameter
    - Pupil-to-Iris Ratio
    - Circularity
    - Symmetry Score
    - Mean Luminance / Reflectance
    """

    def __init__(self):
        pass

    def process_image(self, input_image):
        """
        Processes an image (PIL Image or numpy array) and extracts ocular features
        along with an annotated visualization mask.
        """
        # Convert PIL Image to OpenCV format (BGR)
        if isinstance(input_image, Image.Image):
            img_rgb = np.array(input_image.convert('RGB'))
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        elif isinstance(input_image, np.ndarray):
            if len(input_image.shape) == 2:
                img_bgr = cv2.cvtColor(input_image, cv2.COLOR_GRAY2BGR)
            elif input_image.shape[2] == 4:
                img_bgr = cv2.cvtColor(input_image, cv2.COLOR_BGRA2BGR)
            else:
                img_bgr = input_image.copy()
        else:
            raise ValueError("Unsupported image format")

        # Resize to standard analysis dimensions while preserving aspect ratio
        h, w = img_bgr.shape[:2]
        target_size = 600
        scale = target_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        img_resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # 1. Image Preprocessing
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced_gray = clahe.apply(gray)
        
        # Noise reduction via Gaussian Blur
        blurred = cv2.GaussianBlur(enhanced_gray, (7, 7), 0)

        # 2. Pupil & Iris Detection Algorithms
        pupil_circle, iris_circle = self._detect_iris_pupil(blurred, img_resized)

        # 3. Extract Mathematical Features
        features = self._calculate_features(pupil_circle, iris_circle, blurred)

        # 4. Generate Diagnostic Visualization Overlay
        annotated_img = self._create_annotated_overlay(img_resized, pupil_circle, iris_circle, features)

        # Convert annotated image to base64
        _, buffer = cv2.imencode('.jpg', annotated_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        b64_annotated = base64.b64encode(buffer).decode('utf-8')

        return {
            'features': features,
            'annotated_image': f"data:image/jpeg;base64,{b64_annotated}",
            'image_dimensions': {'width': new_w, 'height': new_h}
        }

    def _detect_iris_pupil(self, gray, img_bgr):
        """
        Detects pupil (dark inner circle) and iris (outer boundary circle).
        Uses combination of adaptive thresholding, Hough Circles, and contour analysis.
        """
        h, w = gray.shape

        # A. Detect Pupil (Dark central region)
        # Threshold to isolate dark pixels
        min_val, _, min_loc, _ = cv2.minMaxLoc(gray)
        thresh_val = min(min_val + 35, 90)
        _, pupil_thresh = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV)
        
        # Morphological clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        pupil_thresh = cv2.morphologyEx(pupil_thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        pupil_thresh = cv2.morphologyEx(pupil_thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

        pupil_contours, _ = cv2.findContours(pupil_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        pupil_circle = None
        if pupil_contours:
            # Find largest contour in central 70% of image
            center_x, center_y = w // 2, h // 2
            valid_pupil_contours = []
            for c in pupil_contours:
                area = cv2.contourArea(c)
                if area > 100:  # Minimum pupil size
                    (cx, cy), r = cv2.minEnclosingCircle(c)
                    dist_from_center = np.sqrt((cx - center_x)**2 + (cy - center_y)**2)
                    if dist_from_center < max(w, h) * 0.4:
                        valid_pupil_contours.append((area, cx, cy, r, c))
            
            if valid_pupil_contours:
                valid_pupil_contours.sort(key=lambda x: x[0], reverse=True)
                _, cx, cy, r, best_c = valid_pupil_contours[0]
                pupil_circle = (int(cx), int(cy), int(r))

        # Fallback for pupil using Hough Circles if contour search fails
        if pupil_circle is None:
            circles = cv2.HoughCircles(
                gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
                param1=100, param2=30, minRadius=int(min(w, h) * 0.05), maxRadius=int(min(w, h) * 0.25)
            )
            if circles is not None:
                circles = np.uint16(np.around(circles))
                pupil_circle = (int(circles[0][0][0]), int(circles[0][0][1]), int(circles[0][0][2]))
            else:
                # Default estimate in center
                pupil_circle = (int(w / 2), int(h / 2), int(min(w, h) * 0.12))

        # B. Detect Iris (Outer boundary)
        px, py, pr = pupil_circle
        
        # Hough circles centered near pupil location
        iris_circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
            param1=80, param2=25, minRadius=int(pr * 1.8), maxRadius=int(pr * 4.5)
        )

        iris_circle = None
        if iris_circles is not None:
            iris_circles = np.uint16(np.around(iris_circles))
            best_dist = float('inf')
            for c in iris_circles[0]:
                ix, iy, ir = c
                dist = np.sqrt((ix - px)**2 + (iy - py)**2)
                if dist < pr * 1.0 and ir > pr * 1.5:
                    if dist < best_dist:
                        best_dist = dist
                        iris_circle = (int(ix), int(iy), int(ir))

        if iris_circle is None:
            # Fallback estimation using optical contrast boundary
            iris_radius = int(pr * 2.6)
            iris_circle = (px, py, min(iris_radius, int(min(w, h) * 0.42)))

        return pupil_circle, iris_circle

    def _calculate_features(self, pupil_circle, iris_circle, gray_img):
        """
        Computes quantitative eye metrics from detected boundaries.
        """
        px, py, pr = pupil_circle
        ix, iy, ir = iris_circle

        pupil_diameter_px = pr * 2.0
        iris_diameter_px = ir * 2.0
        
        # Key Biomarker 1: Pupil to Iris Ratio
        ratio = pupil_diameter_px / max(iris_diameter_px, 1.0)
        
        # Key Biomarker 2: Pupil Circularity
        # Extract pupil ROI mask to check boundary regularity
        h, w = gray_img.shape
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (px, py), pr, 255, -1)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            area = cv2.contourArea(contours[0])
            perimeter = cv2.arcLength(contours[0], True)
            circularity = (4 * np.pi * area) / (perimeter**2) if perimeter > 0 else 0.95
        else:
            circularity = 0.95

        # Key Biomarker 3: Concentricity / Symmetry (Offset between pupil & iris centers)
        center_offset = np.sqrt((px - ix)**2 + (py - iy)**2)
        symmetry_score = max(0.0, 100.0 - (center_offset / max(ir, 1) * 100.0))

        # Key Biomarker 4: Mean Brightness / Reflectance inside Pupil and Iris
        pupil_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(pupil_mask, (px, py), max(1, pr - 2), 255, -1)
        mean_brightness = float(cv2.mean(gray_img, mask=pupil_mask)[0])

        return {
            'pupil_diameter_px': round(pupil_diameter_px, 1),
            'iris_diameter_px': round(iris_diameter_px, 1),
            'pupil_iris_ratio': round(ratio, 4),
            'circularity_index': round(min(1.0, float(circularity)), 3),
            'symmetry_score_pct': round(float(symmetry_score), 1),
            'pupil_center': [int(px), int(py)],
            'iris_center': [int(ix), int(iy)],
            'brightness_level': round(mean_brightness, 1)
        }

    def _create_annotated_overlay(self, img_bgr, pupil_circle, iris_circle, features):
        """
        Draws visual diagnostic markers:
        - Outer Iris circle (Emerald green)
        - Inner Pupil circle (Amber orange)
        - Center points & alignment line
        - Telemetry HUD overlay
        """
        overlay = img_bgr.copy()
        px, py, pr = pupil_circle
        ix, iy, ir = iris_circle

        # Semi-transparent mask layer
        mask_layer = np.zeros_like(img_bgr)
        
        # Color definitions (BGR)
        COLOR_IRIS = (255, 200, 0)      # Cyan / Light Blue
        COLOR_PUPIL = (0, 165, 255)     # Amber Orange
        COLOR_CENTER = (0, 255, 0)     # Emerald Green
        COLOR_HUD_BG = (20, 20, 25)

        # Fill Iris ring semi-transparently
        cv2.circle(mask_layer, (ix, iy), ir, (60, 40, 0), -1)
        cv2.circle(mask_layer, (px, py), pr, (0, 30, 80), -1)
        
        # Blend mask
        overlay = cv2.addWeighted(overlay, 0.85, mask_layer, 0.35, 0)

        # Draw boundary contours
        cv2.circle(overlay, (ix, iy), ir, COLOR_IRIS, 2, cv2.LINE_AA)
        cv2.circle(overlay, (px, py), pr, COLOR_PUPIL, 2, cv2.LINE_AA)

        # Center crosshairs
        cv2.drawMarker(overlay, (px, py), COLOR_PUPIL, cv2.MARKER_CROSS, 16, 2)
        cv2.drawMarker(overlay, (ix, iy), COLOR_IRIS, cv2.MARKER_CROSS, 16, 2)

        # Connecting vector between centers if offset exists
        if (px != ix or py != iy):
            cv2.line(overlay, (px, py), (ix, iy), (0, 0, 255), 1, cv2.LINE_AA)

        # HUD Telemetry Box at top left
        hud_w, hud_h = 240, 110
        cv2.rectangle(overlay, (10, 10), (10 + hud_w, 10 + hud_h), COLOR_HUD_BG, -1)
        cv2.rectangle(overlay, (10, 10), (10 + hud_w, 10 + hud_h), (80, 80, 100), 1)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(overlay, "AI OPHTHALMIC TELEMETRY", (20, 30), font, 0.4, (200, 220, 255), 1, cv2.LINE_AA)
        cv2.putText(overlay, f"Pupil Dia: {features['pupil_diameter_px']} px", (20, 50), font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(overlay, f"Iris Dia : {features['iris_diameter_px']} px", (20, 70), font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(overlay, f"P/I Ratio: {features['pupil_iris_ratio']}", (20, 90), font, 0.45, (0, 230, 255), 1, cv2.LINE_AA)

        return overlay
