# pip install -r requirements.txt

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import resample_poly, butter, filtfilt, fftconvolve, lfilter
from math import gcd
import sounddevice as sd
import os


# ---------------------------------------------------------
# Phase 1: loading, resampling, saving, basic plots
# ---------------------------------------------------------

def process_audio_file_phase1(filename, target_fs=16000, play_audio=True):
    """
    Phase 1:
    - Read WAV
    - Convert to mono
    - Resample to target_fs
    - Normalize
    - Save processed WAV
    - Plot waveform

    Returns:
        audio (float32), fs (int), file_output_dir (str), base_filename (str)
    """
    # Task 3.1: Read audio file and get sampling rate
    fs_original, audio_data = wavfile.read(filename)
    print(f'Original sampling rate: {fs_original} Hz')
    print(f'Original audio shape: {audio_data.shape}')

    # Convert to float for processing
    if audio_data.dtype == np.int16:
        audio_data = audio_data.astype(np.float32) / 32768.0
    elif audio_data.dtype == np.int32:
        audio_data = audio_data.astype(np.float32) / 2147483648.0
    else:
        audio_data = audio_data.astype(np.float32)

    # Task 3.2: Convert stereo to mono if necessary - using avg and not sum
    if audio_data.ndim == 2:
        print('Multichannel audio detected. Converting to mono (average)...')
        audio_data = audio_data.mean(axis=1)
    else:
        print('Mono audio detected.')

    # Task 3.6: Check sampling rate and resample to 16 kHz
    if fs_original != target_fs:
        print(f'Resampling from {fs_original} Hz to {target_fs} Hz with polyphase...')
        g = gcd(fs_original, target_fs)
        up, down = target_fs // g, fs_original // g
        audio_data = resample_poly(audio_data, up, down)
        fs = target_fs
    else:
        print('Sampling rate already at target_fs.')
        fs = fs_original

    # Normalize audio
    audio_data = audio_data / np.max(np.abs(audio_data))

    if play_audio:
        print('Playing audio...')
        try:
            sd.play(audio_data, fs)
            sd.wait()
        except Exception as e:
            print(f'Could not play audio: {e}')

    # Create a subfolder named after the input file
    base_filename = os.path.splitext(os.path.basename(filename))[0]
    file_output_dir = os.path.join('data/output', base_filename)
    os.makedirs(file_output_dir, exist_ok=True)

    # Task 3.4: Write processed sound file
    output_filename = os.path.join(file_output_dir,
                                   'processed_' + os.path.basename(filename))
    audio_int16 = (audio_data * 32767).astype(np.int16)
    wavfile.write(output_filename, fs, audio_int16)
    print(f'Processed audio saved to: {output_filename}')

    # Task 3.5: Plot waveform as function of sample number
    plt.figure(figsize=(12, 4))
    plt.plot(np.arange(len(audio_data)), audio_data)
    plt.xlabel('Sample Number')
    plt.ylabel('Amplitude')
    plt.title(f'Audio Waveform: {filename}')
    plt.grid(True)
    plt.tight_layout()
    waveform_path = os.path.join(file_output_dir,
                                 f'{base_filename}_waveform.png')
    plt.savefig(waveform_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Waveform plot saved to: {waveform_path}')

    return audio_data, fs, file_output_dir, base_filename

def cosine_demo(audio_data, fs, file_output_dir, base_filename):
    """
    Task 3.7 cosine demo (Phase 1).
    """
    freq = 1000  # 1 kHz
    t = np.arange(len(audio_data)) / fs
    cosine_signal = np.cos(2 * np.pi * freq * t)

    print('Playing 1 kHz cosine signal...')
    try:
        sd.play(cosine_signal, fs)
        sd.wait()
    except Exception as e:
        print(f'Could not play cosine: {e}')

    samples_per_cycle = fs / freq
    two_cycles = int(2 * samples_per_cycle)

    plt.figure(figsize=(10, 4))
    plt.plot(t[:two_cycles], cosine_signal[:two_cycles], 'b-', linewidth=1.5)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.title('1 kHz Cosine Signal - Two Cycles')
    plt.grid(True)
    plt.tight_layout()
    cosine_path = os.path.join(file_output_dir,
                               f'{base_filename}_cosine.png')
    plt.savefig(cosine_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Cosine plot saved to: {cosine_path}')


# ---------------------------------------------------------
# Phase 2: gammatone filterbank + envelopes
# ---------------------------------------------------------

BAND_EDGES = np.array(
    [100, 144, 207, 299, 430, 619,
     892, 1284, 1849, 2663, 3835, 5522, 8000],
    dtype=float
)

# for example one band could be 100-144 hz, or even 5522-8000hz


def erb(fc_hz): #conversion - mathematical equality
    return 24.7 * (4.37 * fc_hz / 1000.0 + 1.0) # Glasberg & Moore ERB in Hz


def design_gammatone_ir(fc, fs, n_order=4, t_len=0.030, b_factor=1.019):
    """
    Design a single gammatone FIR impulse response for center frequency fc.

    Returns:
        h (np.ndarray): impulse response (1D float32)
    """
    L = int(np.round(t_len * fs))
    t = np.arange(L) / fs

    ERB = erb(fc)
    b = b_factor * ERB  # bandwidth parameter

    env = (t ** (n_order - 1)) * np.exp(-2 * np.pi * b * t)
    carrier = np.cos(2 * np.pi * fc * t)
    h = env * carrier

    # Hann window for smooth truncation
    h *= np.hanning(L)
    h /= np.max(np.abs(h) + 1e-12) # basic normalization

    return h.astype(np.float32) 


def design_gammatone_bank(fs, band_edges=BAND_EDGES, n_order=4):
    """
    Design a bank of gammatone FIR filters using custom band edges.

    - band_edges: array of length N+1 (low0, high0, high1, ..., highN-1)
    - center freqs = geometric mean of each band

    Returns:
        h_bank: 2D array [L, N] of filter taps
        center_freqs: 1D array [N] of center frequencies
    """
    lows = band_edges[:-1]
    highs = band_edges[1:]
    center_freqs = np.sqrt(lows * highs)  # geometric mean per band, still in list format

    filters = [design_gammatone_ir(fc, fs, n_order=n_order)
               for fc in center_freqs]

    max_len = max(len(h) for h in filters)
    N = len(filters)
    h_bank = np.zeros((max_len, N), dtype=np.float32)

    for k, h in enumerate(filters):
        h_bank[:len(h), k] = h

    print('Designed gammatone bank with center freqs (Hz):')
    print(np.round(center_freqs, 1))

    return h_bank, center_freqs


def apply_gammatone_bank(x, h_bank):
    """
    Convolve input x with each column of h_bank.

    Returns:
        bands: 2D array [len(x), N]
    """
    n_samples = len(x)
    N = h_bank.shape[1]
    bands = np.zeros((n_samples, N), dtype=np.float32)

    for k in range(N):
        h = h_bank[:, k]
        y = fftconvolve(x, h, mode="same")
        bands[:, k] = y.astype(np.float32)

    return bands


def extract_envelopes(bands, fs, fc_lp=400.0, order=4):
    """
    Envelope extraction:
    - full-wave rectify each channel
    - lowpass at fc_lp (Butterworth, order 'order')

    Returns:
        envelopes: 2D array [len(x), N]
    """
    rectified = np.abs(bands) #rectify task 7

    # Butterworth lowpass - task 8
    nyq = fs / 2.0
    Wn = fc_lp / nyq
    b_lp, a_lp = butter(order, Wn, btype='low')

    # filtfilt for zero-phase envelopes
    #envelopes = filtfilt(b_lp, a_lp, rectified, axis=0)
    envelopes = lfilter(b_lp, a_lp, rectified, axis=0)
    # passing the rectified signal through a 400 Hz Butterworth lowpass filter - manually doing the envelope extraction as “rectify + lowpass”

    return envelopes

def plot_phase2_results(audio_data, fs, bands, envelopes, center_freqs, file_output_dir, base_filename):
    t = np.arange(len(audio_data)) / fs

    # ---------- Lowest and highest band outputs (Task 6) ----------
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(t, bands[:, 0])
    plt.title(f'Lowest Band Output ({center_freqs[0]:.1f} Hz)')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(t, bands[:, -1])
    plt.title(f'Highest Band Output ({center_freqs[-1]:.1f} Hz)')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)

    plt.tight_layout()
    band_plot_path = os.path.join(
        file_output_dir, f'{base_filename}_bands_low_high.png'
    )
    plt.savefig(band_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Band output plots saved to: {band_plot_path}')

    # ---------- Lowest and highest envelopes (Task 9) ----------
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(t, envelopes[:, 0])
    plt.title(f'Envelope - Lowest Band ({center_freqs[0]:.1f} Hz)')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(t, envelopes[:, -1])
    plt.title(f'Envelope - Highest Band ({center_freqs[-1]:.1f} Hz)')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)

    plt.tight_layout()
    env_plot_path = os.path.join(
        file_output_dir, f'{base_filename}_envelopes_low_high.png'
    )
    plt.savefig(env_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Envelope plots saved to: {env_plot_path}')

    # ---------- All band outputs----------
    num_ch = bands.shape[1]
    cols = 4
    rows = int(np.ceil(num_ch / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(16, 3 * rows), sharex=True)
    axes = axes.flatten()

    for k in range(num_ch):
        ax = axes[k]
        ax.plot(t, bands[:, k])
        ax.set_title(f'Band {k+1} ({center_freqs[k]:.1f} Hz)', fontsize=9)
        ax.grid(True)

    # Turn off any unused subplots
    for ax in axes[num_ch:]:
        ax.axis('off')

    fig.suptitle('All Band Outputs', fontsize=14)
    fig.supxlabel('Time (s)')
    fig.supylabel('Amplitude')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    bands_all_path = os.path.join(
        file_output_dir, f'{base_filename}_bands_all.png'
    )
    fig.savefig(bands_all_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'All-band output plots saved to: {bands_all_path}')

    # ---------- All band envelopes----------
    fig, axes = plt.subplots(rows, cols, figsize=(16, 3 * rows), sharex=True)
    axes = axes.flatten()

    for k in range(num_ch):
        ax = axes[k]
        ax.plot(t, envelopes[:, k])
        ax.set_title(f'Env Band {k+1} ({center_freqs[k]:.1f} Hz)', fontsize=9)
        ax.grid(True)

    for ax in axes[num_ch:]:
        ax.axis('off')

    fig.suptitle('Envelopes of All Bands', fontsize=14)
    fig.supxlabel('Time (s)')
    fig.supylabel('Amplitude')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    env_all_path = os.path.join(
        file_output_dir, f'{base_filename}_envelopes_all.png'
    )
    fig.savefig(env_all_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'All-band envelope plots saved to: {env_all_path}')

def run_phase2_pipeline(audio_data, fs, file_output_dir, base_filename): #command func
    print('\n--- Phase 2: Designing gammatone filterbank ---')
    h_bank, center_freqs = design_gammatone_bank(fs)

    print('Filtering audio through gammatone bank (Task 5)...')
    bands = apply_gammatone_bank(audio_data, h_bank)

    print('Extracting envelopes (Tasks 7–8)...')
    envelopes = extract_envelopes(bands, fs, fc_lp=400.0, order=4)

    print('Plotting Phase 2 results (Tasks 6 & 9)...')
    plot_phase2_results(audio_data, fs, bands, envelopes,
                        center_freqs, file_output_dir, base_filename)

    return bands, envelopes, center_freqs

def process_audio_file(filename, run_phase2=True, play_audio=True):
    """
    High-level function that:
    - Runs Phase 1 on the file
    - Runs Phase 2 on the processed audio
    """
    audio_data, fs, file_output_dir, base_filename = \
        process_audio_file_phase1(filename, play_audio=play_audio)

    # Optional Phase 1 cosine (Task 3.7) – comment out for Phase 2 if needed
    # cosine_demo(audio_data, fs, file_output_dir, base_filename)
    if run_phase2:
        run_phase2_pipeline(audio_data, fs, file_output_dir, base_filename)

    return audio_data, fs


def process_multiple_files(file_list, run_phase2=True, play_audio=False):
    for i, filename in enumerate(file_list):
        print(f'\n{"="*60}')
        print(f'Processing file {i+1}/{len(file_list)}: {filename}')
        print(f'{"="*60}')
        try:
            process_audio_file(filename,
                               run_phase2=run_phase2,
                               play_audio=play_audio)
        except Exception as e:
            print(f'Error processing {filename}: {e}')


if __name__ == "__main__":
    audio, fs = process_audio_file(
        'data/input/child_quiet_single_fast/child_quiet_single_fast.wav',
        run_phase2=True, play_audio=False
    )

    files = [
        'data/input/child_quiet_single_fast/child_quiet_single_fast.wav',
        'data/input/female_noisy_single_neutral/female_noisy_single_neutral.wav',
        'data/input/female_quiet_single_neutral/female_quiet_single_neutral.wav',
        'data/input/male_quiet_single_slow/male_quiet_single_slow.wav',
        'data/input/females_noisy_convo_neutral/females_noisy_convo_neutral.wav',
        'data/input/male_noisy_single_neutral/male_noisy_single_neutral.wav',
        'data/input/multiple_noisy_convo_neutral/multiple_noisy_convo_neutral.wav',
        'data/input/multiple_noisy_overlapped_neutral/multiple_noisy_overlapped_neutral.wav'
    ]

    # Run Phase 1+2 on all files (no playback to save your ears)
    # process_multiple_files(files, run_phase2=True, play_audio=False)
