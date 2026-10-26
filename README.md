# ECG Record Digitization and Heart Disease Diagnosis using CNN

An end-to-end framework that converts legacy screen-recorded or printed ECG videos/images into calibrated digital time-series signals and performs automated cardiac arrhythmia diagnosis using a Convolutional Neural Network (CNN).

---

## Key Features
1. **Video & Image Ingestion:** Extracts temporal frames from ECG monitors, applies Gaussian smoothing, and performs Hough Transform grid line suppression.
2. **Spatial-Temporal Calibration:** Converts pixel trajectories into standardized time-series voltage signals ($360\text{ Hz}$).
3. **1D CNN Classification:** Accurately classifies cardiac conditions into **6 standard classes**: Normal Sinus Rhythm (NSR), Atrial Fibrillation (AFib), Premature Ventricular Contractions (PVC), Bradycardia, Ventricular Tachycardia (VT), and Others.
4. **Statistical Validation:** Implements McNemar's statistical testing, ablation analyses, and artifact robustness benchmarks matching experimental results.

---

## Installation

```bash
git clone [https://github.com/your-username/ecg-digitization-diagnosis.git](https://github.com/your-username/ecg-digitization-diagnosis.git)
cd ecg-digitization-diagnosis
pip install -r requirements.txt