# pip install -r requirements.txt

"""
SYDE 252 - Phase 1, Task 3: Audio File Processing
Python implementation for cochlear implant project
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import resample
import sounddevice as sd
import os


def process_audio_file(filename):
    """
    Process audio file for cochlear implant signal processor
    
    Parameters:
    filename (str): Path to audio file (.wav format)
    
    Returns:
    tuple: (processed_audio, sampling_rate)
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
    
    # Task 3.2: Convert stereo to mono if necessary
    if len(audio_data.shape) == 2:
        num_channels = audio_data.shape[1]
        if num_channels == 2:
            print('Stereo audio detected. Converting to mono...')
            audio_data = audio_data[:, 0] + audio_data[:, 1]  # Add both channels
        else:
            audio_data = audio_data[:, 0]  # Take first channel
    else:
        print('Mono audio detected.')
    
    # Task 3.6: Resample to 16 kHz if necessary
    target_fs = 16000  # Target sampling rate
    
    if fs_original != target_fs:
        print(f'Resampling from {fs_original} Hz to {target_fs} Hz...')
        num_samples = int(len(audio_data) * target_fs / fs_original)
        audio_data = resample(audio_data, num_samples)
        fs = target_fs
    else:
        print('Sampling rate already at 16 kHz.')
        fs = fs_original
    
    # Normalize audio
    audio_data = audio_data / np.max(np.abs(audio_data))
    
    # Task 3.3: Play the sound
    print('Playing audio...')
    try:
        sd.play(audio_data, fs)
        sd.wait()  # Wait for playback to finish
    except Exception as e:
        print(f'Could not play audio: {e}')
    
    # Task 3.4: Write sound to new file
    # Create a subfolder named after the input file
    base_filename = os.path.splitext(os.path.basename(filename))[0]  # Get filename without extension
    file_output_dir = os.path.join('data/output', base_filename)
    os.makedirs(file_output_dir, exist_ok=True)  # Create file-specific output directory
    
    output_filename = os.path.join(file_output_dir, 'processed_' + os.path.basename(filename))
    # Convert back to int16 for saving
    audio_int16 = (audio_data * 32767).astype(np.int16)
    wavfile.write(output_filename, fs, audio_int16)
    print(f'Audio saved to: {output_filename}')
    
    # Task 3.5: Plot waveform as function of sample number
    plt.figure(figsize=(12, 4))
    plt.plot(np.arange(len(audio_data)), audio_data)
    plt.xlabel('Sample Number')
    plt.ylabel('Amplitude')
    plt.title(f'Audio Waveform: {filename}')
    plt.grid(True)
    plt.tight_layout()
    waveform_path = os.path.join(file_output_dir, f'{base_filename}_waveform.png')
    plt.savefig(waveform_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Task 3.7: Generate 1 kHz cosine signal
    freq = 1000  # 1 kHz
    duration = len(audio_data) / fs  # Match duration of input
    t = np.arange(len(audio_data)) / fs  # Time vector
    
    cosine_signal = np.cos(2 * np.pi * freq * t)
    
    # Play the cosine signal
    print('Playing 1 kHz cosine signal...')
    try:
        sd.play(cosine_signal, fs)
        sd.wait()
    except Exception as e:
        print(f'Could not play cosine: {e}')
    
    # Plot two cycles of cosine waveform vs time
    samples_per_cycle = fs / freq  # Samples in one cycle
    two_cycles = int(2 * samples_per_cycle)  # Samples for 2 cycles
    
    plt.figure(figsize=(10, 4))
    plt.plot(t[:two_cycles], cosine_signal[:two_cycles], 'b-', linewidth=1.5)
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.title('1 kHz Cosine Signal - Two Cycles')
    plt.grid(True)
    plt.tight_layout()
    cosine_path = os.path.join(file_output_dir, f'{base_filename}_cosine.png')
    plt.savefig(cosine_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return audio_data, fs


def process_multiple_files(file_list):
    """
    Process multiple audio files
    
    Parameters:
    file_list (list): List of audio file paths
    """
    for i, filename in enumerate(file_list):
        print(f'\n{"="*60}')
        print(f'Processing file {i+1}/{len(file_list)}: {filename}')
        print(f'{"="*60}')
        try:
            process_audio_file(filename)
        except Exception as e:
            print(f'Error processing {filename}: {e}')


# Example usage
if __name__ == "__main__":
    # Process a single file
    #processed_audio, fs = process_audio_file('audio_files/100981__mo_damage__atari-speech.wav')
    #processed_audio, fs = process_audio_file('data/input/child_quiet_single_fast/child_quiet_single_fast.wav')
    # Or process multiple files:
    files = [
        'data/input/child_quiet_single_fast/child_quiet_single_fast.wav',
        'data/input/female_noisy_single_neutral/female_noisy_single_neutral.wav',
        'data/input/female_quiet_single_neutral/female_quiet_single_neutral.wav',
        'data/input/male_quiet_single_slow/male_quiet_single_slow.wav'
    ]
    
    # Uncomment to run:
    process_multiple_files(files)