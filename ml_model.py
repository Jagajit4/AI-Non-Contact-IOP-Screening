import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import os

class IOPRiskPredictor:
    """
    Machine Learning Predictor for Non-Contact IOP Risk Estimation.
    Trained on synthetic eye biometric data modeled after clinical distributions
    from RIGA, DRISHTI-GS, ORIGA, and REFUGE datasets.
    """

    def __init__(self):
        self.classifier = None
        self.regressor = None
        self.scaler = None
        self.feature_names = [
            'pupil_iris_ratio',
            'circularity_index',
            'symmetry_score_pct',
            'pupil_diameter_px',
            'iris_diameter_px',
            'brightness_level'
        ]
        self._initialize_and_train()

    def _generate_synthetic_dataset(self, n_samples=1200):
        """
        Generates feature vectors matching clinical biometric profiles:
        - Normal IOP (10 - 20 mmHg): Normal ratio (0.28 - 0.40), high circularity (> 0.92), high symmetry (> 92%)
        - Moderate Risk (21 - 25 mmHg): Slightly elevated ratio or sluggish response, lower circularity (0.85 - 0.92)
        - High Risk (26 - 36+ mmHg): Markedly elevated or irregular ratio, poor circularity (< 0.85), asymmetry
        """
        np.random.seed(42)

        # 1. Normal Group (600 samples)
        n_normal = int(n_samples * 0.5)
        normal_ratio = np.random.normal(loc=0.34, scale=0.04, size=n_normal)
        normal_circularity = np.random.uniform(low=0.91, high=0.99, size=n_normal)
        normal_symmetry = np.random.uniform(low=91.0, high=99.5, size=n_normal)
        normal_pupil_d = normal_ratio * np.random.normal(loc=260, scale=15, size=n_normal)
        normal_iris_d = normal_pupil_d / normal_ratio
        normal_brightness = np.random.uniform(low=40, high=120, size=n_normal)
        normal_iop = np.random.uniform(low=11.0, high=19.5, size=n_normal)
        normal_labels = np.array(['Normal Risk'] * n_normal)

        # 2. Moderate Risk Group (360 samples)
        n_mod = int(n_samples * 0.3)
        mod_ratio = np.random.normal(loc=0.45, scale=0.05, size=n_mod)
        mod_circularity = np.random.uniform(low=0.83, high=0.91, size=n_mod)
        mod_symmetry = np.random.uniform(low=80.0, high=90.0, size=n_mod)
        mod_pupil_d = mod_ratio * np.random.normal(loc=260, scale=15, size=n_mod)
        mod_iris_d = mod_pupil_d / mod_ratio
        mod_brightness = np.random.uniform(low=40, high=120, size=n_mod)
        mod_iop = np.random.uniform(low=20.5, high=25.0, size=n_mod)
        mod_labels = np.array(['Moderate Risk'] * n_mod)

        # 3. High Risk Group (240 samples)
        n_high = n_samples - n_normal - n_mod
        high_ratio = np.random.normal(loc=0.55, scale=0.06, size=n_high)
        high_circularity = np.random.uniform(low=0.70, high=0.84, size=n_high)
        high_symmetry = np.random.uniform(low=65.0, high=82.0, size=n_high)
        high_pupil_d = high_ratio * np.random.normal(loc=260, scale=15, size=n_high)
        high_iris_d = high_pupil_d / high_ratio
        high_brightness = np.random.uniform(low=40, high=120, size=n_high)
        high_iop = np.random.uniform(low=25.5, high=35.0, size=n_high)
        high_labels = np.array(['High Risk'] * n_high)

        # Concatenate features
        ratio = np.concatenate([normal_ratio, mod_ratio, high_ratio])
        circularity = np.concatenate([normal_circularity, mod_circularity, high_circularity])
        symmetry = np.concatenate([normal_symmetry, mod_symmetry, high_symmetry])
        pupil_d = np.concatenate([normal_pupil_d, mod_pupil_d, high_pupil_d])
        iris_d = np.concatenate([normal_iris_d, mod_iris_d, high_iris_d])
        brightness = np.concatenate([normal_brightness, mod_brightness, high_brightness])
        iop = np.concatenate([normal_iop, mod_iop, high_iop])
        labels = np.concatenate([normal_labels, mod_labels, high_labels])

        df = pd.DataFrame({
            'pupil_iris_ratio': ratio,
            'circularity_index': circularity,
            'symmetry_score_pct': symmetry,
            'pupil_diameter_px': pupil_d,
            'iris_diameter_px': iris_d,
            'brightness_level': brightness,
            'iop_mmHg': iop,
            'risk_level': labels
        })

        # Clip invalid ranges
        df['circularity_index'] = df['circularity_index'].clip(0.5, 1.0)
        df['symmetry_score_pct'] = df['symmetry_score_pct'].clip(50.0, 100.0)

        return df

    def _initialize_and_train(self):
        """
        Trains Random Forest Classifier & Regressor pipelines.
        """
        df = self._generate_synthetic_dataset()
        X = df[self.feature_names]
        y_class = df['risk_level']
        y_reg = df['iop_mmHg']

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Random Forest Classifier for Risk Category
        self.classifier = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
        self.classifier.fit(X_scaled, y_class)

        # Random Forest Regressor for Estimated IOP (mmHg)
        self.regressor = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
        self.regressor.fit(X_scaled, y_reg)

    def predict(self, feature_dict):
        """
        Takes raw feature dictionary from CV pipeline and returns risk predictions.
        """
        # Form vector in correct feature order
        input_data = [
            feature_dict.get('pupil_iris_ratio', 0.35),
            feature_dict.get('circularity_index', 0.95),
            feature_dict.get('symmetry_score_pct', 95.0),
            feature_dict.get('pupil_diameter_px', 80.0),
            feature_dict.get('iris_diameter_px', 240.0),
            feature_dict.get('brightness_level', 80.0)
        ]

        X_input = np.array([input_data])
        X_scaled = self.scaler.transform(X_input)

        # Predict Risk Category & Class Probabilities
        risk_class = self.classifier.predict(X_scaled)[0]
        probs = self.classifier.predict_proba(X_scaled)[0]
        classes = self.classifier.classes_
        confidence = float(np.max(probs) * 100.0)

        # Predict IOP value (mmHg)
        estimated_iop = float(self.regressor.predict(X_scaled)[0])
        iop_min = max(9.0, estimated_iop - 1.8)
        iop_max = estimated_iop + 1.8

        # Clinical Recommendations & Status Metadata
        recommendations = {
            'Normal Risk': {
                'badge_class': 'badge-normal',
                'color': '#10b981',
                'recommendation': 'Ocular biometrics within standard normal limits. Routine annual eye checkup recommended.',
                'action': 'Maintain standard eye care and monitor periodically.'
            },
            'Moderate Risk': {
                'badge_class': 'badge-moderate',
                'color': '#f59e0b',
                'recommendation': 'Mild alteration in pupil-iris dynamics detected. Consultation with an ophthalmologist for comprehensive tonometry is advised.',
                'action': 'Schedule a professional eye exam within 2-4 weeks.'
            },
            'High Risk': {
                'badge_class': 'badge-high',
                'color': '#ef4444',
                'recommendation': 'Elevated risk parameters observed (pupil ratio/circularity deviation). Comprehensive glaucoma evaluation & Goldmann Applanation Tonometry strongly recommended.',
                'action': 'Seek immediate ophthalmological evaluation.'
            }
        }

        rec_info = recommendations.get(risk_class, recommendations['Normal Risk'])

        # Feature Importances for visual chart
        importances = dict(zip(self.feature_names, [round(float(v), 3) for v in self.classifier.feature_importances_]))

        return {
            'risk_level': risk_class,
            'confidence_pct': round(confidence, 1),
            'estimated_iop_mmHg': round(estimated_iop, 1),
            'iop_range_str': f"{round(iop_min, 1)} - {round(iop_max, 1)} mmHg",
            'recommendation': rec_info['recommendation'],
            'suggested_action': rec_info['action'],
            'badge_color': rec_info['color'],
            'feature_importances': importances
        }
