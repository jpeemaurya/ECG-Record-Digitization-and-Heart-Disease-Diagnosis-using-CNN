import cv2
import numpy as np

class ECGDigitizer:
    """
    Handles ECG screen recording digitization: frame extraction, 
    grid removal, noise filtering, and spatial-temporal calibration.
    """
    def __init__(self, t_ref=1.0, v_ref=1.0, px_t=100, px_v=100, fs=360):
        self.t_ref = t_ref  # Reference time interval (seconds)
        self.v_ref = v_ref  # Reference voltage interval (mV)
        self.px_t = px_t    # Pixel distance corresponding to t_ref
        self.px_v = px_v    # Pixel distance corresponding to v_ref
        self.fs = fs        # Sampling frequency

    def extract_frames(self, video_path):
        """Extracts frames from an ECG monitor screen video."""
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        return frames

    def preprocess_frame(self, frame):
        """Grayscale conversion, Gaussian filtering, and thresholding."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY_INV)
        return thresh

    def remove_grid_and_extract_waveform(self, thresh_frame):
        """Suppresses background grid lines using Hough Transform & extracts contours."""
        edges = cv2.Canny(thresh_frame, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)
        
        cleaned = thresh_frame.copy()
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Suppress grid lines (horizontal/vertical lines)
                if abs(x1 - x2) < 5 or abs(y1 - y2) < 5:
                    cv2.line(cleaned, (x1, y1), (x2, y2), (0, 0, 0), 2)
                    
        # Extract pixel trajectory of the waveform
        y_coords, x_coords = np.where(cleaned > 0)
        return x_coords, y_coords

    def calibrate_signal(self, x_coords, y_coords, image_height):
        """
        Calibrates pixel coordinates into time-series voltage signals 
        using linear scaling equations: t = alpha_x * (xp - x0), V = alpha_y * (y0 - yp)
        """
        if len(x_coords) == 0 or len(y_coords) == 0:
            return np.zeros(self.fs * 5)

        alpha_x = self.t_ref / self.px_t
        alpha_y = self.v_ref / self.px_v

        x0 = np.min(x_coords)
        y0 = np.mean(y_coords)

        # Sort along x-axis to maintain temporal sequence
        sorted_indices = np.argsort(x_coords)
        x_sorted = x_coords[sorted_indices]
        y_sorted = y_coords[sorted_indices]

        # Map to physical units
        t_signal = alpha_x * (x_sorted - x0)
        v_signal = alpha_y * (y0 - y_sorted)

        # Resample to uniform length
        unique_t, indices = np.unique(t_signal, return_index=True)
        unique_v = v_signal[indices]
        
        if len(unique_t) > 1:
            uniform_t = np.linspace(unique_t[0], unique_t[-1], self.fs * 5)
            reconstructed_signal = np.interp(uniform_t, unique_t, unique_v)
        else:
            reconstructed_signal = np.zeros(self.fs * 5)

        return reconstructed_signal