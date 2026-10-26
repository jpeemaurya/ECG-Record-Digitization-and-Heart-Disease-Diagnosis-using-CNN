import numpy as np
import os
from src.digitization import ECGDigitizer
from src.signal_processing import segment_signal
from src.models import build_1d_cnn
from src.evaluation import evaluate_performance

def main():
    print("==================================================")
    print("ECG Digitization and Heart Disease Diagnosis Pipeline")
    print("==================================================")
    
    # 1. Initialize Digitizer
    digitizer = ECGDigitizer()
    print("[1] ECGDigitizer initialized successfully.")
    
    # 2. Simulate or Load Data Pipeline
    print("[2] Generating synthetic/sample signal for pipeline demonstration...")
    dummy_signal = np.sin(np.linspace(0, 50, 5000)) + np.random.normal(0, 0.1, 5000)
    segments = segment_signal(dummy_signal)
    print(f"    - Extracted {len(segments)} segments of shape {segments.shape[1]} from signal.")
    
    # 3. Build Model
    print("[3] Building 1D CNN Diagnostic Model...")
    model = build_1d_cnn(input_shape=(segments.shape[1], 1), num_classes=6)
    model.summary()
    
    # 4. Run Mock Inference Evaluation
    print("[4] Running evaluation pipeline...")
    X_dummy = np.expand_dims(segments, axis=-1)
    preds = model.predict(X_dummy)
    pred_classes = np.argmax(preds, axis=1)
    true_classes = np.random.randint(0, 6, size=len(pred_classes))
    
    metrics = evaluate_performance(true_classes, pred_classes)
    print("\n--- Evaluation Results ---")
    print(f"Accuracy : {metrics['Accuracy']:.4f}")
    print(f"Precision: {metrics['Precision']:.4f}")
    print(f"Recall   : {metrics['Recall']:.4f}")
    print(f"F1-Score : {metrics['F1-Score']:.4f}")
    print("Pipeline execution completed successfully!")

if __name__ == "__main__":
    main()