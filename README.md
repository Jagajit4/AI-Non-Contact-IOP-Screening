#  AI-Based Non-Contact Intraocular Pressure (IOP) Screening System Using Iris & Pupil Analysis


> *A low-cost, non-contact, AI-assisted early glaucoma risk detection and non-contact IOP estimation system powered by OpenCV computer vision algorithms and Scikit-Learn machine learning classifiers.*

---

##  Medical Disclaimer
> [!IMPORTANT]
> This system is strictly designed as an **AI-assisted screening tool** for early risk detection and non-contact IOP estimation, not a medical diagnostic replacement for Goldmann Applanation Tonometry or professional ophthalmological examinations.

---

##  Problem Statement

Glaucoma is a leading cause of irreversible blindness worldwide. Existing Intraocular Pressure (IOP) measurement devices (like Goldmann Applanation Tonometers or Air-Puff Tonometers) present significant challenges:
- **High Cost:** Devices cost between $3,000 and $15,000+, rendering them unavailable in rural or under-resourced clinics.
- **Physical Contact & Inconvenience:** Requires physical eye contact, topical anesthesia drops, or uncomfortable high-pressure air bursts.
- **Lack of Accessibility:** Remote screening in rural health camps is virtually non-existent due to bulky hardware requirements.

There is an urgent need for a **low-cost, non-contact, camera-based screening solution** for early glaucoma risk detection.

---

##  Proposed Solution

Our solution captures a user's eye image or video using a standard consumer camera (smartphone or webcam). A custom computer vision pipeline isolates pupil and iris boundaries, extracting key biometrics:
1. **Pupil Diameter ($D_p$)**
2. **Iris Diameter ($D_i$)**
3. **Pupil-to-Iris Ratio ($D_p / D_i$)**
4. **Pupil Circularity Index ($4\pi \times \text{Area} / \text{Perimeter}^2$)**
5. **Concentricity & Symmetry Score (%)**
6. **Reflectance / Luminance Index**

These feature vectors are analyzed by a local **Random Forest Machine Learning model** trained on feature representations derived from benchmark clinical eye datasets (**RIGA, DRISHTI-GS, ORIGA, REFUGE**) to estimate glaucoma risk levels (**Normal, Moderate, High**) and predict an approximate IOP range (**mmHg**).

---

##  Key Features & Novelty

-  **No Eye Contact:** 100% optical measurement. Zero physical touch or corneal abrasion risk.
-  **No Anesthesia Drops:** Eliminates discomfort and medical risk.
-  **Smartphone & Web Compatible:** Works on standard consumer cameras without expensive hardware.
-  **100% Local Machine Learning:** Runs entirely offline without cloud API dependency, ensuring zero operational cost and complete patient data privacy.
-  **Real-time Diagnostic Telemetry:** Instant OpenCV boundary visualization overlays and clinical recommendations.

---

##  System Architecture & Workflow Flow Diagram

Below is the complete human-friendly end-to-end processing pipeline mapping the models, tools, and datasets utilized at each stage:

```text
+-----------------------------------------------------------------------------------+
| 1. EYE IMAGE CAPTURE                                                              |
| - Tool: HTML5 MediaDevices API / Web Camera / File Upload                         |
| - Output: High-Resolution Eye RGB Image Matrix                                    |
+-----------------------------------------------------------------------------------+
                                      │
                                      ▼
+-----------------------------------------------------------------------------------+
| 2. IMAGE PREPROCESSING & NOISE REDUCTION                                          |
| - Tool: OpenCV (cv2)                                                              |
| - Methods: Grayscale Conversion + CLAHE (Adaptive Histogram) + Gaussian Blur      |
| - Goal: Contrast enhancement between iris and dark pupil region                   |
+-----------------------------------------------------------------------------------+
                                      │
                                      ▼
+-----------------------------------------------------------------------------------+
| 3. IRIS & PUPIL BOUNDARY SEGMENTATION                                             |
| - Tool: OpenCV Hough Circle Transform & Canny Edge Detection                      |
| - Methods: Morphological Ellipse Fitting & Contour Radius Estimation              |
| - Output: Annotated Image Overlay (Outer Iris Cyan Circle, Inner Pupil Amber Circle)|
+-----------------------------------------------------------------------------------+
                                      │
                                      ▼
+-----------------------------------------------------------------------------------+
| 4. MATHEMATICAL FEATURE EXTRACTION                                                |
| - Tool: NumPy & SciPy                                                             |
| - Biometrics Extracted:                                                           |
|   • Pupil Diameter (px)    • Iris Diameter (px)      • Pupil/Iris Ratio (Dp/Di)  |
|   • Circularity Index      • Concentricity/Symmetry • Reflectance / Luminance   |
+-----------------------------------------------------------------------------------+
                                      │
                                      ▼
+-----------------------------------------------------------------------------------+
| 5. MACHINE LEARNING RISK ESTIMATION                                               |
| - Tool: Scikit-Learn                                                              |
| - Models: Random Forest Classifier (Risk Tiers) & Regressor (IOP mmHg Range)       |
| - Datasets Grounded On: RIGA, DRISHTI-GS, ORIGA, REFUGE                           |
| - Output: Risk Level (Normal/Moderate/High), Confidence %, IOP Range (mmHg)       |
+-----------------------------------------------------------------------------------+
                                      │
                                      ▼
+-----------------------------------------------------------------------------------+
| 6. INTERACTIVE UI DASHBOARD RESULT                                                |
| - Tech: Flask + Modern HTML5 / CSS3 (Glassmorphism) / Vanilla JavaScript          |
| - UI Components: Real-Time Risk Gauge, Feature Telemetry Cards, Action Plan       |
+-----------------------------------------------------------------------------------+
```

---

##  Scalability Roadmap

- **Phase 1 (Hackathon Prototype):** Web application with OpenCV feature extraction and local Random Forest model inference.
- **Phase 2:** Cross-platform Mobile Application (iOS & Android) with cloud database sync and patient history tracking.
- **Phase 3:** Standalone low-cost hardware screening kiosk for rural healthcare camps and community centers.
- **Phase 4:** Full integration with hospital Electronic Health Record (EHR) systems and slit-lamp camera attachments.

---


##  Tech Stack

- **Backend Framework:** Python 3, Flask 3.1
- **Computer Vision:** OpenCV (`opencv-python`)
- **Machine Learning:** Scikit-Learn (`scikit-learn`), NumPy, Pandas
- **Frontend UI:** HTML5, Modern Vanilla CSS (Glassmorphism), JavaScript (ES6)

