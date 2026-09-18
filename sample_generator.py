import cv2
import numpy as np
import os

def create_sample_eye_images(output_dir):
    """
    Synthesizes realistic eye images with varied iris/pupil characteristics
    for demonstration and offline evaluation.
    """
    os.makedirs(output_dir, exist_ok=True)
    size = (600, 600)

    samples = [
        {
            'filename': 'normal_eye.jpg',
            'iris_r': 150,
            'pupil_r': 52,
            'pupil_offset': (0, 0),
            'iris_color': (140, 90, 40), # Brown/Hazel
            'title': 'Normal Eye Baseline'
        },
        {
            'filename': 'moderate_risk_eye.jpg',
            'iris_r': 150,
            'pupil_r': 70,
            'pupil_offset': (4, -3),
            'iris_color': (160, 110, 50), # Amber
            'title': 'Moderate IOP Risk Case'
        },
        {
            'filename': 'high_risk_eye.jpg',
            'iris_r': 150,
            'pupil_r': 88,
            'pupil_offset': (10, -8),
            'iris_color': (120, 80, 30), # Dark Brown
            'title': 'High IOP Risk / Glaucoma Suspect'
        }
    ]

    for sample in samples:
        img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        
        # 1. Skin & Sclera background (Light creamy tone with gradient)
        cx, cy = size[0] // 2, size[1] // 2
        
        # Draw background sclera / skin texture
        for y in range(size[1]):
            for x in range(size[0]):
                dist = np.sqrt((x - cx)**2 + (y - cy)**2)
                # Base sclera (whiteish/cream)
                if dist < 220:
                    # Sclera shade
                    factor = 1.0 - (dist / 240) * 0.15
                    img[y, x] = [int(240 * factor), int(245 * factor), int(250 * factor)]
                else:
                    # Surrounding eyelid skin
                    img[y, x] = [170, 190, 215]

        # Add subtle sclera blood vessel textures
        for _ in range(8):
            start_angle = np.random.uniform(0, 2 * np.pi)
            r1 = np.random.uniform(160, 200)
            x1 = int(cx + r1 * np.cos(start_angle))
            y1 = int(cy + r1 * np.sin(start_angle))
            x2 = int(cx + (r1 - 40) * np.cos(start_angle + 0.1))
            y2 = int(cy + (r1 - 40) * np.sin(start_angle + 0.1))
            cv2.line(img, (x1, y1), (x2, y2), (180, 180, 230), 1, cv2.LINE_AA)

        # 2. Draw Iris
        iris_r = sample['iris_r']
        cv2.circle(img, (cx, cy), iris_r, sample['iris_color'], -1, cv2.LINE_AA)
        
        # Add radial iris fibers / striations
        for angle_deg in range(0, 360, 3):
            rad = np.deg2rad(angle_deg)
            r_inner = sample['pupil_r'] + 5
            r_outer = iris_r - 4
            x_in = int(cx + r_inner * np.cos(rad))
            y_in = int(cy + r_inner * np.sin(rad))
            x_out = int(cx + r_outer * np.cos(rad))
            y_out = int(cy + r_outer * np.sin(rad))
            
            shade = int(np.random.uniform(-30, 40))
            fiber_color = (
                max(0, min(255, sample['iris_color'][0] + shade)),
                max(0, min(255, sample['iris_color'][1] + shade)),
                max(0, min(255, sample['iris_color'][2] + shade))
            )
            cv2.line(img, (x_in, y_in), (x_out, y_out), fiber_color, 1, cv2.LINE_AA)

        # Outer limbal ring border
        cv2.circle(img, (cx, cy), iris_r, (40, 30, 20), 3, cv2.LINE_AA)

        # 3. Draw Pupil (with offset for high risk asymmetry)
        off_x, off_y = sample['pupil_offset']
        px, py = cx + off_x, cy + off_y
        pupil_r = sample['pupil_r']
        
        cv2.circle(img, (px, py), pupil_r, (12, 10, 10), -1, cv2.LINE_AA)

        # 4. Add Cornea Light Reflection Specular Highlight
        cv2.circle(img, (px - int(pupil_r * 0.35), py - int(pupil_r * 0.35)), int(pupil_r * 0.18), (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(img, (px + int(pupil_r * 0.4), py + int(pupil_r * 0.3)), int(pupil_r * 0.08), (220, 230, 240), -1, cv2.LINE_AA)

        # Blur slightly to look natural
        img = cv2.GaussianBlur(img, (3, 3), 0)

        filepath = os.path.join(output_dir, sample['filename'])
        cv2.imwrite(filepath, img)
        print(f"Generated sample image: {filepath}")

if __name__ == '__main__':
    samples_dir = os.path.join(os.path.dirname(__file__), 'static', 'samples')
    create_sample_eye_images(samples_dir)
