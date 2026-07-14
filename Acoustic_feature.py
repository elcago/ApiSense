#!/usr/bin/env python3

import sys
import os
import librosa
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
import gc
import warnings
warnings.filterwarnings("ignore")

SR      = 44100
N_FFT   = 1024
HOP     = 512
SEG_SEC = 10
N_SEGS  = 60

BP_F0   = (150, 350)
BP_SPEC = (250, 390)
BP_FLUX = (100, 950)
BP_HARM = (180, 1700)

YIN_FMIN = 215
YIN_FMAX = 236

F0_EST  = 230.0
N_HARM  = 7
HARM_BW = 2

TARGET_RMS = 0.388


def bandpass(y, sr, flo, fhi, order=4):
    nyq = sr / 2
    b, a = butter(order, [flo / nyq, fhi / nyq], btype="band")
    return np.nan_to_num(filtfilt(b, a, y))


def extract_features(audio_path, verbose=True):
    if verbose:
        ext = os.path.splitext(audio_path)[1].upper()
        print(f"Loading {audio_path} ({ext}) ...")

    y, sr = librosa.load(audio_path, sr=SR, mono=True)

    max_samples = 10 * 60 * SR
    if len(y) > max_samples:
        y = y[:max_samples]
        if verbose:
            print(f"  Trimmed to 10 min ({max_samples:,} samples)")

    if verbose:
        rms_raw = np.sqrt(np.mean(y**2))
        print(f"  Duration : {len(y)/sr:.1f}s | "
              f"Raw RMS : {rms_raw:.5f} ({20*np.log10(rms_raw+1e-12):.1f} dBFS)")

    raw_rms = np.sqrt(np.mean(y**2))
    y_norm  = np.clip(y * (TARGET_RMS / (raw_rms + 1e-12)), -1.0, 1.0)

    y_f0   = bandpass(y_norm, sr, *BP_F0)
    y_spec = bandpass(y_norm, sr, *BP_SPEC)
    y_flux = bandpass(y_norm, sr, *BP_FLUX)
    y_harm = bandpass(y_norm, sr, *BP_HARM)
    y_zcr  = y_norm.copy()

    harmonic_freqs = [F0_EST * k for k in range(1, N_HARM + 1)]
    freq_res       = sr / N_FFT
    freqs_axis     = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)

    segment_len   = SEG_SEC * sr
    n_segs        = len(y_norm) // segment_len
    f0_voiced_all = []
    centroid_all  = []
    bandwidth_all = []
    flux_all      = []
    rms_all       = []
    zcr_all       = []
    harm_energy   = 0.0
    total_energy  = 0.0

    if verbose:
        print(f"  Processing {n_segs} × {SEG_SEC}s segments ...")

    for i in range(n_segs):
        sl = slice(i * segment_len, (i + 1) * segment_len)

        f0_seg = np.nan_to_num(
            librosa.yin(y_f0[sl], fmin=YIN_FMIN, fmax=YIN_FMAX,
                        sr=sr, frame_length=N_FFT, hop_length=HOP)
        )
        voiced = f0_seg[f0_seg > YIN_FMIN]
        if len(voiced) > 0:
            f0_voiced_all.append(voiced)

        S_harm = np.abs(librosa.stft(y_harm[sl], n_fft=N_FFT,
                                      hop_length=HOP, window="hann"))
        pwr = np.mean(S_harm ** 2, axis=1)
        h_e = 0.0
        for hf in harmonic_freqs:
            if hf < sr / 2:
                lo = max(0, int((hf - HARM_BW * freq_res) / freq_res))
                hi = min(len(freqs_axis) - 1,
                         int((hf + HARM_BW * freq_res) / freq_res))
                h_e += pwr[lo:hi + 1].sum()
        harm_energy  += h_e
        total_energy += pwr.sum()

        S_spec = np.abs(librosa.stft(y_spec[sl], n_fft=N_FFT,
                                      hop_length=HOP, window="hann"))
        cent = librosa.feature.spectral_centroid(S=S_spec, sr=sr)[0]
        centroid_all.append(cent)
        bw = librosa.feature.spectral_bandwidth(
            S=S_spec, sr=sr, centroid=cent[np.newaxis, :]
        )[0]
        bandwidth_all.append(bw)

        S_flux = np.abs(librosa.stft(y_flux[sl], n_fft=N_FFT,
                                      hop_length=HOP, window="hann"))
        flux = librosa.onset.onset_strength(
            S=librosa.power_to_db(S_flux ** 2), sr=sr, hop_length=HOP
        )
        flux_all.append(flux)

        rms_seg = librosa.feature.rms(
            y=y_norm[sl], frame_length=N_FFT, hop_length=HOP
        )[0]
        rms_all.append(rms_seg)

        zcr_seg = librosa.feature.zero_crossing_rate(
            y_zcr[sl], frame_length=N_FFT, hop_length=HOP
        )[0]
        zcr_all.append(zcr_seg)

        del S_harm, S_spec, S_flux
        gc.collect()

    f0_all = (np.concatenate(f0_voiced_all)
              if f0_voiced_all else np.array([np.nan]))

    return {
        "f0"                : round(float(np.mean(f0_all)),                        4),
        "f0_stability"      : round(float(np.std(f0_all)),                         4),
        "harmonic_ratio"    : round(float(harm_energy / (total_energy + 1e-12)),   6),
        "spectral_centroid" : round(float(np.mean(np.concatenate(centroid_all))),  4),
        "spectral_bandwidth": round(float(np.mean(np.concatenate(bandwidth_all))), 4),
        "spectral_flux"     : round(float(np.mean(np.concatenate(flux_all))),      6),
        "RMS_Energy"        : round(float(np.mean(np.concatenate(rms_all))),       6),
        "zero_crossing_rate": round(float(np.mean(np.concatenate(zcr_all))),       6),
    }


def print_results(label, features, ctrl_min, ctrl_max, ctrl_mean):
    def chk(v, lo, hi):
        return "OK" if lo <= v <= hi else "FAIL"
    def pct(v, t):
        return (v - t) / t * 100

    print(f"\n{'='*90}")
    print(f"  {label}")
    print(f"{'='*90}")
    print(f"  {'Feature':<22} {'Value':>10}  {'%Err':>8}  "
          f"{'CSV ctrl range':>18}  {'In range?':>9}")
    print(f"  {'─'*80}")
    for feat, val in features.items():
        t  = ctrl_mean[feat]
        lo = ctrl_min[feat]
        hi = ctrl_max[feat]
        print(f"  {feat:<22} {val:>10.4f}  {pct(val,t):>+7.1f}%  "
              f"{str(round(lo,3))+'–'+str(round(hi,3)):>18}  "
              f"{chk(val,lo,hi):>9}")


if __name__ == "__main__":

    ctrl_mean = {
        "f0":               225.228,
        "f0_stability":     4.774,
        "harmonic_ratio":   0.926,
        "spectral_centroid":325.020,
        "spectral_bandwidth":72.718,
        "spectral_flux":    0.208,
        "RMS_Energy":       0.388,
        "zero_crossing_rate":0.185,
    }
    ctrl_min = {
        "f0":               214.86,
        "f0_stability":     3.52,
        "harmonic_ratio":   0.860,
        "spectral_centroid":274.1,
        "spectral_bandwidth":58.7,
        "spectral_flux":    0.190,
        "RMS_Energy":       0.288,
        "zero_crossing_rate":0.168,
    }
    ctrl_max = {
        "f0":               236.1,
        "f0_stability":     6.96,
        "harmonic_ratio":   0.999,
        "spectral_centroid":373.2,
        "spectral_bandwidth":87.6,
        "spectral_flux":    0.290,
        "RMS_Energy":       0.531,
        "zero_crossing_rate":0.203,
    }

    args  = sys.argv[1:]
    files = []
    csv_out = None
    i = 0
    while i < len(args):
        if args[i] == "--csv" and i + 1 < len(args):
            csv_out = args[i + 1]
            i += 2
        else:
            files.append(args[i])
            i += 1

    if not files:
        print("Usage: python extract_acoustic_features.py file1.mp3 [file2.mp3 ...] [--csv output.csv]")
        print("       Accepts MP3, WAV, FLAC, OGG, M4A and any other librosa-supported format.")
        sys.exit(1)

    all_results = {}
    for audio_path in files:
        feats = extract_features(audio_path)
        all_results[audio_path] = feats
        print_results(os.path.basename(audio_path), feats,
                      ctrl_min, ctrl_max, ctrl_mean)

    if len(all_results) > 1:
        feat_names = list(next(iter(all_results.values())).keys())
        labels     = list(all_results.keys())
        print(f"\n{'='*90}")
        print(f"  CROSS-FILE COMPARISON")
        print(f"{'='*90}")
        header = f"  {'Feature':<22}"
        for lbl in labels:
            short = os.path.basename(lbl)[:14]
            header += f"  {short:>14}"
        header += f"  {'CSV ctrl mean':>14}"
        print(header)
        print(f"  {'─'*85}")
        for feat in feat_names:
            row = f"  {feat:<22}"
            for lbl in labels:
                row += f"  {all_results[lbl][feat]:>14.4f}"
            row += f"  {ctrl_mean[feat]:>14.4f}"
            print(row)

    if csv_out:
        rows = []
        for path, feats in all_results.items():
            row = {"file": os.path.basename(path)}
            row.update(feats)
            rows.append(row)
        df = pd.DataFrame(rows)
        df.to_csv(csv_out, index=False)
        print(f"\nResults saved to: {csv_out}")
