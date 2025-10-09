import numpy as np
from scipy.signal import firwin, iirfilter, lfilter, filtfilt




def make_bands(N: int, fmin: float = 100.0, fmax: float = 8000.0, spacing: str = "log"):
    if spacing == "log":
        edges = np.geomspace(fmin, fmax, num=N + 1)
    else:
        edges = np.linspace(fmin, fmax, num=N + 1)
    centers = np.sqrt(edges[:-1] * edges[1:])  # geometric mean
    return edges, centers




def design_bandpass(fs: int, f1: float, f2: float, kind: str = "fir", order: int = 512):
    nyq = fs / 2
    low, high = f1 / nyq, f2 / nyq
    if not (0 < low < high < 1):
        raise ValueError(f"Bad band edges: {f1}-{f2} vs Nyquist {nyq}")
    if kind == "fir":
        taps = firwin(order + 1, [low, high], pass_zero=False)
        return ("fir", taps)
    # IIR Butterworth
    b, a = iirfilter(order, [low, high], btype="band", ftype="butter")
    return ("iir", (b, a))




def apply_bandpass(x, fs: int, f1: float, f2: float, kind: str = "fir", order: int = 512, zero_phase: bool = True):
    mode, coeffs = design_bandpass(fs, f1, f2, kind=kind, order=order)
    if mode == "fir":
        b = coeffs
        y = filtfilt(b, [1.0], x) if zero_phase else lfilter(b, [1.0], x)
    else:
        b, a = coeffs
        y = filtfilt(b, a, x) if zero_phase else lfilter(b, a, x)
    return y.astype(np.float32)




def filterbank_signals(x, fs: int, N: int = 8, spacing: str = "log", kind: str = "fir", order: int = 512):
    edges, centers = make_bands(N, spacing=spacing)
    bands = []
    for i in range(N):
        y = apply_bandpass(x, fs, float(edges[i]), float(edges[i + 1]), kind=kind, order=order)
        bands.append(y)
    return np.array(bands), np.array(centers), np.array(edges)