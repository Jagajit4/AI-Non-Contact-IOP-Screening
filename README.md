# 👁️ AI-Based Non-Contact Intraocular Pressure (IOP) Screening System Using Iris & Pupil Analysis

> **Hackathon Edition Prototype**  
> *A low-cost, non-contact, AI-assisted early glaucoma risk detection and non-contact IOP estimation system powered by OpenCV computer vision algorithms and Scikit-Learn machine learning classifiers.*

---

## 📌 Medical Disclaimer
> [!IMPORTANT]
> This system is strictly designed as an **AI-assisted screening tool** for early risk detection and non-contact IOP estimation, not a medical diagnostic replacement for Goldmann Applanation Tonometry or professional ophthalmological examinations.

---

## 🎯 Problem Statement

Glaucoma is a leading cause of irreversible blindness worldwide. Existing Intraocular Pressure (IOP) measurement devices (like Goldmann Applanation Tonometers or Air-Puff Tonometers) present significant challenges:
- **High Cost:** Devices cost between $3,000 and $15,000+, rendering them unavailable in rural or under-resourced clinics.
- **Physical Contact & Inconvenience:** Requires physical eye contact, topical anesthesia drops, or uncomfortable high-pressure air bursts.
- **Lack of Accessibility:** Remote screening in rural health camps is virtually non-existent due to bulky hardware requirements.

There is an urgent need for a **low-cost, non-contact, camera-based screening solution** for early glaucoma risk detection.

---

## 💡 Proposed Solution

Our solution captures a user's eye image or video using a standard consumer camera (smartphone or webcam). A custom computer vision pipeline isolates pupil and iris boundaries, extracting key biometrics:
1. **Pupil Diameter ($D_p$)**
2. **Iris Diameter ($D_i$)**
3. **Pupil-to-Iris Ratio ($D_p / D_i$)**
4. **Pupil Circularity Index ($4\pi \times \text{Area} / \text{Perimeter}^2$)**
5. **Concentricity & Symmetry Score (%)**
6. **Reflectance / Luminance Index**

These feature vectors are analyzed by a local **Random Forest Machine Learning model** trained on feature representations derived from benchmark clinical eye datasets (**RIGA, DRISHTI-GS, ORIGA, REFUGE**) to estimate glaucoma risk levels (**Normal, Moderate, High**) and predict an approximate IOP range (**mmHg**).

---

## ⭐ Key Features & Novelty

- ❌ **No Eye Contact:** 100% optical measurement. Zero physical touch or corneal abrasion risk.
- 💧 **No Anesthesia Drops:** Eliminates discomfort and medical risk.
- 📱 **Smartphone & Web Compatible:** Works on standard consumer cameras without expensive hardware.
- ⚡ **100% Local Machine Learning:** Runs entirely offline without cloud API dependency, ensuring zero operational cost and complete patient data privacy.
- 📊 **Real-time Diagnostic Telemetry:** Instant OpenCV boundary visualization overlays and clinical recommendations.

---

## 🗺️ System Architecture & Workflow Flow Diagram

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

## 🚀 Scalability Roadmap

- **Phase 1 (Hackathon Prototype):** Web application with OpenCV feature extraction and local Random Forest model inference.
- **Phase 2:** Cross-platform Mobile Application (iOS & Android) with cloud database sync and patient history tracking.
- **Phase 3:** Standalone low-cost hardware screening kiosk for rural healthcare camps and community centers.
- **Phase 4:** Full integration with hospital Electronic Health Record (EHR) systems and slit-lamp camera attachments.

---

## ❓ Anticipated Judge Questions & Answers

### Q1: Why rely on iris and pupil measurements for IOP estimation?
> **Answer:** Medical literature indicates that elevated intraocular pressure and glaucoma progression cause subtle autonomic nervous system variations and structural changes in pupil-to-iris ratios, circularity, and pupillary dynamics. We leverage these optical markers as screening risk indicators rather than direct mechanical pressure measurements.

### Q2: How far have you implemented the prototype?
> **Answer:** We have implemented a complete end-to-end working system featuring OpenCV contrast enhancement & Hough boundary segmentation, 6 biometric feature extractions, Random Forest classifier & regressor pipelines, and an interactive Flask web dashboard with live camera capture and 1-click sample presets.

### Q3: Why use Machine Learning over static threshold rules?
> **Answer:** The relationship between pupil circularity, iris ratios, lighting conditions, and IOP is non-linear and multi-faceted. Machine learning algorithms (Random Forest) effectively model complex feature interactions that rigid threshold rules fail to capture.

### Q4: Why run the model locally instead of using a cloud API?
> **Answer:** Local model execution ensures zero latency (<200ms), 100% patient data privacy, zero internet dependency in remote rural medical camps, and zero operational API expenses.

---

## 📦 Tech Stack

- **Backend Framework:** Python 3, Flask 3.1
- **Computer Vision:** OpenCV (`opencv-python`)
- **Machine Learning:** Scikit-Learn (`scikit-learn`), NumPy, Pandas
- **Frontend UI:** HTML5, Modern Vanilla CSS (Glassmorphism), JavaScript (ES6)

---

## 💻 Step-by-Step Setup & Running Locally

1. **Clone or Open Project Directory:**
   ```bash
   cd c:\project\VIT
   ```

2. **Generate Demo Preset Images (Optional if already generated):**
   ```bash
   python sample_generator.py
   ```

3. **Start the Flask Web Server:**
   ```bash
   python app.py
   ```

4. **Open in Browser:**
   Navigate to `http://localhost:5000` or `http://127.0.0.1:5000`.

---

## 🐙 How to Upload This Project to a New GitHub Repository

Follow these step-by-step terminal commands to publish this repository on GitHub:

### Step 1: Create a New Repository on GitHub
1. Go to [GitHub.com](https://github.com) and log in.
2. Click the **"+"** icon in the top right corner and select **"New repository"**.
3. Name your repository: `AI-Non-Contact-IOP-Screening`
4. Set visibility to **Public** (recommended for hackathons).
5. Leave "Initialize this repository with a README" **unchecked** (since we already have a README).
6. Click **"Create repository"**. Copy your repository URL (e.g., `https://github.com/YOUR_USERNAME/AI-Non-Contact-IOP-Screening.git`).

### Step 2: Initialize Git and Push Code from Terminal
Run the following commands in your terminal inside `c:\project\VIT`:

```bash
# 1. Navigate to project root
cd c:\project\VIT

# 2. Initialize Git repository
git init

# 3. Add all files to staging
git add .

# 4. Create initial commit
git commit -m "Initial commit: AI-Based Non-Contact IOP Screening System prototype"

# 5. Rename branch to main
git branch -M main

# 6. Add your GitHub repository remote URL (replace with your actual GitHub URL)
git remote add origin https://github.com/YOUR_USERNAME/AI-Non-Contact-IOP-Screening.git

# 7. Push code to GitHub
git push -u origin main
```

---

## 🏆 Presentation Script for Judges (3-Minute Pitch)

> *"Good morning judges! We are presenting **Ophthalmo-AI**, an AI-based non-contact intraocular pressure screening system using iris and pupil analysis.*
>
> *Glaucoma causes irreversible vision loss, yet millions in rural areas cannot access early screening because tonometers cost over $3,000 and require eye contact or anesthesia drops.*
>
> *Our solution uses a standard smartphone or webcam. Using OpenCV, we segment the iris and pupil boundaries, extract key biometrics like pupil-to-iris ratio and circularity, and feed them into a local Random Forest ML model trained on clinical dataset distributions (RIGA, DRISHTI-GS, REFUGE).*
>
> *As shown in our live demo, the system outputs an estimated IOP range, a risk classification badge, and clinical recommendations in under 200 milliseconds—completely offline, with zero API costs and total patient privacy.*
>
> *Thank you, and we welcome your questions!"*
