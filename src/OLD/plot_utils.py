from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt




def plot_waveform(x, fs: int, title: str, outpath):
    t = np.arange(len(x)) / fs
    plt.figure()
    plt.plot(t, x)
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.title(title)
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()




def plot_two(x1, fs: int, x2, title1: str, title2: str, outpath):
    t = np.arange(len(x1)) / fs
    plt.figure(figsize=(8, 5))
    plt.subplot(2, 1, 1)
    plt.plot(t, x1)
    plt.title(title1)
    plt.subplot(2, 1, 2)
    plt.plot(t, x2)
    plt.title(title2)
    plt.xlabel("Time [s]")
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()