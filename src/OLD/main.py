import argparse
import sys
from pathlib import Path
from src.audio_io import load_mono, save_wav, cosine_tone
from src.filterbank import filterbank_signals
from src.envelope import envelopes_for_bands
from src.synth import synthesize_from_envelopes
from src.plot_utils import plot_waveform, plot_two




def process_file(in_path, out_dir, N=8, spacing="log", kind="fir", order=512, lp_cut=400.0, fs_target=16000):
    x, fs = load_mono(in_path, target_fs=fs_target)
    stem = Path(in_path).stem

    # waveform + 1 kHz cosine (same length)
    plot_waveform(x, fs, f"{stem} waveform", Path(out_dir) / f"{stem}.waveform.png")
    cos = cosine_tone(len(x), fs, freq=1000.0)
    plot_two(x, fs, cos, "Input (mono, 16 kHz)", "1 kHz cosine", Path(out_dir) / f"{stem}.cosine_compare.png")

    # filterbank → envelopes
    bands, centers, edges = filterbank_signals(x, fs, N=N, spacing=spacing, kind=kind, order=order)
    envs = envelopes_for_bands(bands, fs, cutoff=lp_cut, order=4, zero_phase=True)

    # plots: lowest & highest band + their envelopes
    plot_waveform(bands[0], fs, f"Lowest band {edges[0]:.0f}-{edges[1]:.0f} Hz", Path(out_dir) / f"{stem}.low_band.png")
    plot_waveform(bands[-1], fs, f"Highest band {edges[-2]:.0f}-{edges[-1]:.0f} Hz", Path(out_dir) / f"{stem}.high_band.png")
    plot_waveform(envs[0], fs, "Envelope (low band)", Path(out_dir) / f"{stem}.low_env.png")
    plot_waveform(envs[-1], fs, "Envelope (high band)", Path(out_dir) / f"{stem}.high_env.png")

    # synthesize vocoded output
    y = synthesize_from_envelopes(envs, centers, fs)
    save_wav(Path(out_dir) / f"{stem}_vocoded.wav", y, fs)




def main():
    ap = argparse.ArgumentParser(description="Cochlear CI vocoder (offline)")
    ap.add_argument("input", help="Path to a .wav file OR a directory of .wav files")
    ap.add_argument("--out_dir", default="data/output", help="Output directory for audio + plots")
    ap.add_argument("--N", type=int, default=8, help="# of bands")
    ap.add_argument("--spacing", choices=["log", "linear"], default="log")
    ap.add_argument("--kind", choices=["fir", "iir"], default="fir")
    ap.add_argument("--order", type=int, default=512, help="FIR taps or IIR order")
    ap.add_argument("--lp_cut", type=float, default=400.0, help="Envelope LPF cutoff (Hz)")
    ap.add_argument("--fs", type=int, default=16000, help="Target sample rate")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if in_path.is_dir():
        wavs = list(in_path.glob("*.wav"))
        if not wavs:
            print("No .wav files found.", file=sys.stderr)
            sys.exit(1)
        for p in wavs:
            process_file(p, out_dir, args.N, args.spacing, args.kind, args.order, args.lp_cut, args.fs)
    else:
        process_file(in_path, out_dir, args.N, args.spacing, args.kind, args.order, args.lp_cut, args.fs)




if __name__ == "__main__":
    main()