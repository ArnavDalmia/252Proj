import numpy as np
from scipy.signal import butter, lfilter, filtfilt

def lowpass_butter(cutoff: float, fs: int, order: int = 4):
    b, a = butter(order, cutoff / (fs / 2), btype="low")
    return b, a

def envelope_rectify_lpf(x, fs: int, cutoff: float = 400.0, order: int = 4, zero_phase: bool = True):
    rect = np.abs(x)
    b, a = lowpass_butter(cutoff, fs, order)
    env = filtfilt(b, a, rect) if zero_phase else lfilter(b, a, rect)
    return np.maximum(env, 0.0).astype(np.float32)

def envelopes_for_bands(bands, fs: int, cutoff: float = 400.0, order: int = 4, zero_phase: bool = True):
    return np.array([envelope_rectify_lpf(b, fs, cutoff, order, zero_phase) for b in bands])