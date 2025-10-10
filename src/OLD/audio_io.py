from pathlib import Path
import numpy as np
import soundfile as sf
from math import gcd
from scipy.signal import resample_poly


def load_mono(path, target_fs: int = 16000):
    """Read WAV, downmix to mono, resample to target_fs, normalize to [-1, 1]."""
    x, fs = sf.read(str(path), always_2d=False)
    x = np.asarray(x)
    if x.ndim == 2:
        x = x.mean(axis=1)
    if fs != target_fs:
        g = gcd(fs, target_fs)
        up, down = target_fs // g, fs // g
        x = resample_poly(x, up, down)
        fs = target_fs
    peak = np.max(np.abs(x)) if x.size else 0
    if peak > 0:
        x = x / peak
    return x.astype(np.float32), fs




def save_wav(path, x, fs):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), x, fs)




def cosine_tone(length_samples: int, fs: int, freq: float = 1000.0, phase: float = 0.0):
    t = np.arange(length_samples) / fs
    return np.cos(2 * np.pi * freq * t + phase).astype(np.float32)