# pip install -r requirements.txt

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import resample_poly, butter, filtfilt, fftconvolve, lfilter
from math import gcd
import sounddevice as sd
import os


# ---------------------------------------------------------
# Phase 1: loading, resampling, saving, basic plots - from prev code, simple refactoring
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

# 12 bands (original)
BAND_EDGES_12 = np.array(
    [100, 144, 207, 299, 430, 619,
     892, 1284, 1849, 2663, 3835, 5522, 8000],
    dtype=float
)

# 8 bands (reduced for iteration 3)
BAND_EDGES_8 = np.array(
    [100, 172, 299, 515, 892, 1540, 2663, 4602, 8000],
    dtype=float
)

BAND_EDGES = BAND_EDGES_12  # default

# for example one band could be 100-144 hz, or even 5522-8000hz


def erb(fc_hz): #conversion - mathematical equality
    return 24.7 * (4.37 * fc_hz / 1000.0 + 1.0) # Glasberg & Moore ERB in Hz


def design_gammatone_ir(fc, fs, n_order=4, t_len=0.030, b_factor=1.019, bandwidth_extension=1.0):
    """
    Design a single gammatone FIR impulse response for center frequency fc.

    Parameters:
        bandwidth_extension: Factor to widen bandwidth (default 1.0 = no change)
                           Use 1.1-1.2 for 10-20% overlap (Iteration 1)
    Returns:
        h (np.ndarray): impulse response (1D float32)
    """
    L = int(np.round(t_len * fs))
    t = np.arange(L) / fs

    ERB = erb(fc)
    b = b_factor * ERB * bandwidth_extension  # bandwidth parameter with extension

    env = (t ** (n_order - 1)) * np.exp(-2 * np.pi * b * t)
    carrier = np.cos(2 * np.pi * fc * t)
    h = env * carrier

    # Hann window for smooth truncation
    h *= np.hanning(L)
    h /= np.max(np.abs(h) + 1e-12) # basic normalization

    return h.astype(np.float32) 


def design_butterworth_bandpass(fc, fs, bandwidth_hz, order=4):
    """
    Design a Butterworth bandpass filter centered at fc.
    Used for high-frequency bands in Iteration 1.
    
    Returns:
        b, a: Filter coefficients for scipy.signal.lfilter
    """
    nyq = fs / 2.0
    low = (fc - bandwidth_hz/2) / nyq
    high = (fc + bandwidth_hz/2) / nyq
    # Clamp to valid range
    low = max(0.01, min(0.99, low))
    high = max(0.01, min(0.99, high))
    b, a = butter(order, [low, high], btype='band')
    return b, a


def design_gammatone_bank(fs, band_edges=BAND_EDGES_12, n_order=4, bandwidth_extension=1.0, 
                         use_butterworth_high=False, n_butterworth_bands=0):
    """
    Design: Bank of gammatone FIR filters using custom band edges desc above

    - band_edges: array of length N+1 (low0, high0, high1, ..., highN-1) created***
    - center freqs = geometric mean of each band = (sqrt(low * high))
    - bandwidth_extension: Factor to widen gammatone bandwidth (Iteration 1: 1.15 for 15% overlap)
    - use_butterworth_high: If True, use Butterworth for top bands (Iteration 1)
    - n_butterworth_bands: Number of top bands to use Butterworth (Iteration 1: 3-4)

    Returns:
        h_bank: 2D array [L, N] of filter taps - to be used later on
        center_freqs: 1D array [N] of center frequencies
        butterworth_filters: list of (b, a) tuples for Butterworth bands (or None)
    """
    lows = band_edges[:-1]
    highs = band_edges[1:]
    center_freqs = np.sqrt(lows * highs)  # geometric mean per band, still in list format

    # Design gammatone filters with potential bandwidth extension
    filters = [design_gammatone_ir(fc, fs, n_order=n_order, bandwidth_extension=bandwidth_extension)
               for fc in center_freqs]

    max_len = max(len(h) for h in filters)
    N = len(filters)
    h_bank = np.zeros((max_len, N), dtype=np.float32)

    for k, h in enumerate(filters):
        h_bank[:len(h), k] = h

    # Prepare Butterworth filters for high bands if requested
    butterworth_filters = None
    if use_butterworth_high and n_butterworth_bands > 0:
        butterworth_filters = []
        for k in range(N):
            if k >= (N - n_butterworth_bands):
                # Use Butterworth for this high band
                bandwidth_hz = erb(center_freqs[k]) * bandwidth_extension * 1.019
                b, a = design_butterworth_bandpass(center_freqs[k], fs, bandwidth_hz, order=4)
                butterworth_filters.append((b, a))
            else:
                butterworth_filters.append(None)
    
    print('Designed gammatone bank with center freqs (Hz):')
    print(np.round(center_freqs, 1))
    if use_butterworth_high:
        print(f'Using Butterworth bandpass for top {n_butterworth_bands} bands')

    return h_bank, center_freqs, butterworth_filters


def apply_gammatone_bank(x, h_bank, butterworth_filters=None):
    """
    Apply input x with each column of h_bank.
    If butterworth_filters is provided, use IIR filtering for those bands.

    Returns:
        bands: 2D array [len(x), N]
    """
    n_samples = len(x)
    N = h_bank.shape[1]
    bands = np.zeros((n_samples, N), dtype=np.float32)

    for k in range(N):
        # Check if this band uses Butterworth
        if butterworth_filters is not None and butterworth_filters[k] is not None:
            b, a = butterworth_filters[k]
            y = lfilter(b, a, x)
            bands[:, k] = y.astype(np.float32)
        else:
            # Use gammatone FIR
            h = h_bank[:, k]
            y = fftconvolve(x, h, mode="same")
            bands[:, k] = y.astype(np.float32)

    return bands


def extract_envelopes(bands, fs, fc_lp=400.0, order=4):
    """
    Envelope extraction:
    - full-wave rectify each channel
    - lowpass at fc_lp (Butterworth low pass, order = 'order')

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

def run_phase2_pipeline(audio_data, fs, file_output_dir, base_filename, 
                       bandwidth_extension=1.0, use_butterworth_high=False, 
                       n_butterworth_bands=0, envelope_cutoff=400.0, envelope_order=4,
                       band_edges=BAND_EDGES_12): 
    print('\n--- Phase 2: Designing gammatone filterbank ---')
    h_bank, center_freqs, butterworth_filters = design_gammatone_bank(
        fs, 
        band_edges=band_edges,
        bandwidth_extension=bandwidth_extension,
        use_butterworth_high=use_butterworth_high,
        n_butterworth_bands=n_butterworth_bands
    )

    print('Filtering audio through gammatone bank (Task 5)...')
    bands = apply_gammatone_bank(audio_data, h_bank, butterworth_filters)

    print(f'Extracting envelopes (fc_lp={envelope_cutoff} Hz, order={envelope_order})...')
    envelopes = extract_envelopes(bands, fs, fc_lp=envelope_cutoff, order=envelope_order)

    print('Plotting Phase 2 results (Tasks 6 & 9)...')
    plot_phase2_results(audio_data, fs, bands, envelopes,
                        center_freqs, file_output_dir, base_filename)

    return bands, envelopes, center_freqs

def run_phase3_pipeline(audio_data, fs, envelopes, center_freqs,file_output_dir, base_filename,play_output=True):
    """
    Phase 3: Tasks 10–13
    Task 10: For each channel, generate a cosine at that channel's center freq.
    Task 11: Amplitude modulate each cosine with that channel's envelope.
    Task 12: Sum all channels and normalize.
    Task 13: Play and save output sound.
    """
    print('\n--- Phase 3: Carrier generation and synthesis (Tasks 10–13) ---')

    num_samples, num_ch = envelopes.shape
    t = np.arange(num_samples) / fs

    # ----- Task 10: cosine carriers for each channel -----
    cos_carriers = np.zeros_like(envelopes, dtype=np.float32)
    for k, fc in enumerate(center_freqs):
        cos_carriers[:, k] = np.cos(2.0 * np.pi * fc * t)

    # ----- Task 11: amplitude modulation with envelopes -----
    # Using envelopes from Task 8 as the modulator.
    # If your instructor insists on using raw rectified bands instead,
    # you could replace `envelopes` with np.abs(bands) here.
    modulated = envelopes * cos_carriers  # element-wise multiply

    # ----- Task 12: sum across channels and normalize -----
    y_out = np.sum(modulated, axis=1)
    max_val = np.max(np.abs(y_out)) + 1e-12
    y_out = y_out / max_val

    # ----- Task 13: play and save -----
    if play_output:
        print('Playing synthesized Phase 3 output...')
        try:
            sd.play(y_out, fs)
            sd.wait()
        except Exception as e:
            print(f'Could not play Phase 3 output: {e}')

    # Save to WAV
    out_filename = os.path.join(file_output_dir,
                                f'{base_filename}_phase3_output.wav')
    y_int16 = (y_out * 32767).astype(np.int16)
    wavfile.write(out_filename, fs, y_int16)
    print(f'Phase 3 output saved to: {out_filename}')

    # Optional: plot final output waveform
    t_full = np.arange(len(y_out)) / fs
    plt.figure(figsize=(12, 4))
    plt.plot(t_full, y_out)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Phase 3 Synthesized Output Waveform')
    plt.grid(True)
    plt.tight_layout()
    out_plot_path = os.path.join(file_output_dir,
                                 f'{base_filename}_phase3_output_waveform.png')
    plt.savefig(out_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Phase 3 output waveform plot saved to: {out_plot_path}')

    return y_out


def process_audio_file(filename, run_phase2=True, run_phase3=False,
                       play_audio=True, play_phase3=True,
                       iteration=None, output_subdir=None):
    """
    High-level function that:
    - Runs Phase 1 on the file
    - Optionally runs Phase 2 on the processed audio
    - Optionally runs Phase 3 synthesis (Tasks 10–13)
    
    Parameters:
        iteration: 'iteration1' or 'iteration2' for specific configurations
        output_subdir: Custom output subdirectory (e.g., 'iteration1/child_quiet_single_fast')
    """
    audio_data, fs, file_output_dir, base_filename = \
        process_audio_file_phase1(filename, play_audio=play_audio)
    
    # Override output directory if specified
    if output_subdir is not None:
        file_output_dir = os.path.join('data/output', output_subdir)
        os.makedirs(file_output_dir, exist_ok=True)

    bands = envelopes = center_freqs = None

    # Set parameters based on iteration
    bandwidth_extension = 1.0
    use_butterworth_high = False
    n_butterworth_bands = 0
    envelope_cutoff = 400.0
    envelope_order = 4
    band_edges = BAND_EDGES_12  # default to 12 bands
    
    if iteration == 'iteration1':
        # Iteration 1: Band overlap + Butterworth high-freq filters
        bandwidth_extension = 1.15  # 15% bandwidth extension for overlap
        use_butterworth_high = True
        n_butterworth_bands = 4  # Top 4 bands use Butterworth
        envelope_cutoff = 400.0
        envelope_order = 4
        band_edges = BAND_EDGES_12
        print('\n=== ITERATION 1 Configuration ===')
        print(f'- Bandwidth extension: {bandwidth_extension} (15% overlap)')
        print(f'- Top {n_butterworth_bands} bands use Butterworth filters')
        print(f'- Envelope cutoff: {envelope_cutoff} Hz')
        
    elif iteration == 'iteration2':
        # Iteration 2: Higher envelope cutoff, use Iteration 1 filter bank
        bandwidth_extension = 1.15  # Same as Iteration 1
        use_butterworth_high = True
        n_butterworth_bands = 4
        envelope_cutoff = 600.0  # Increased from 400 Hz
        envelope_order = 16  # Increased from 4
        band_edges = BAND_EDGES_12
        print('\n=== ITERATION 2 Configuration ===')
        print(f'- Using Iteration 1 filter bank (bandwidth={bandwidth_extension})')
        print(f'- Envelope cutoff: {envelope_cutoff} Hz (increased from 400 Hz)')
        print(f'- Envelope order: {envelope_order} taps (increased from 4)')
        
    elif iteration == 'iteration3':
        # Iteration 3: 8 channels instead of 12, building off iteration 2
        bandwidth_extension = 1.15  # Same as Iteration 2
        use_butterworth_high = True
        n_butterworth_bands = 3  # Top 3 of 8 bands use Butterworth
        envelope_cutoff = 600.0  # Same as Iteration 2
        envelope_order = 16  # Same as Iteration 2
        band_edges = BAND_EDGES_8  # 8 bands instead of 12
        print('\n=== ITERATION 3 Configuration ===')
        print(f'- Reduced to 8 channels (from 12)')
        print(f'- Bandwidth extension: {bandwidth_extension}')
        print(f'- Top {n_butterworth_bands} bands use Butterworth filters')
        print(f'- Envelope cutoff: {envelope_cutoff} Hz')
        print(f'- Envelope order: {envelope_order} taps')

    if run_phase2:
        bands, envelopes, center_freqs = \
            run_phase2_pipeline(audio_data, fs, file_output_dir, base_filename,
                              bandwidth_extension=bandwidth_extension,
                              use_butterworth_high=use_butterworth_high,
                              n_butterworth_bands=n_butterworth_bands,
                              envelope_cutoff=envelope_cutoff,
                              envelope_order=envelope_order,
                              band_edges=band_edges)

    if run_phase3 and envelopes is not None and center_freqs is not None:
        run_phase3_pipeline(audio_data, fs, envelopes, center_freqs,
                            file_output_dir, base_filename,
                            play_output=play_phase3)

    # Keep return simple for now; Phase 3 output is saved to file
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

    # Process all files for Iteration 3 (8 channels, building off iteration 2)
    print('\n' + '='*80)
    print('PROCESSING ALL FILES FOR ITERATION 3 (8 CHANNELS)')
    print('='*80)
    for filename in files:
        # Extract the category name from path
        category = os.path.basename(os.path.dirname(filename))
        output_subdir = f'iteration3/{category}'
        
        print(f'\nProcessing: {filename}')
        try:
            process_audio_file(
                filename,
                run_phase2=True,
                run_phase3=True,
                play_audio=False,
                play_phase3=False,
                iteration='iteration3',
                output_subdir=output_subdir
            )
        except Exception as e:
            print(f'Error processing {filename}: {e}')
    
    print('\n' + '='*80)
    print('PROCESSING COMPLETE')
    print('='*80)
