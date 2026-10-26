import numpy as np

def z_score_normalize(signal):
    """Applies Z-score standardization to the ECG signal."""
    mean = np.mean(signal)
    std = np.std(signal)
    if std == 0:
        return signal - mean
    return (signal - mean) / std

def segment_signal(signal, sampling_rate=360, window_duration=1.5, overlap=0.5):
    """Segments continuous ECG waveforms into overlapping windows for CNN input."""
    window_size = int(window_duration * sampling_rate)
    step_size = int(window_size * (1 - overlap))
    
    segments = []
    for start in range(0, len(signal) - window_size + 1, step_size):
        segment = signal[start:start + window_size]
        segments.append(z_score_normalize(segment))
        
    return np.array(segments)