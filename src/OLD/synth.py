import numpy as np




def synthesize_from_envelopes(envelopes, centers, fs: int, phases=None):
    n_bands, n_samples = envelopes.shape
    if phases is None:
        phases = np.zeros(n_bands)
    t = np.arange(n_samples) / fs
    y = np.zeros(n_samples, dtype=np.float32)
    for i in range(n_bands):
        carrier = np.cos(2 * np.pi * float(centers[i]) * t + float(phases[i])).astype(np.float32)
        y += envelopes[i] * carrier
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 0:
        y = 0.99 * y / peak
    return y