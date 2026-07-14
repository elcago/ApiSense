import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from pathlib import Path
from scipy.signal import butter, filtfilt


COUNTS_CSV = "data/counts.csv"
AUDIO_DIR = "data/audio"
OUTPUT_CSV = "data/features.csv"

SAMPLE_RATE = 44100
INTERVAL_SECONDS = 600
HOP_LENGTH = 512
N_FFT = 1024
WINGBEAT_LOW_HZ = 180
YIN_FMIN = 80.0
YIN_FMAX = 500.0
YIN_THRESHOLD = 0.1
N_HARMONICS = 7


def compute_foraging_features(counts_df):
    counts_df = counts_df.copy()
    counts_df["time_h"] = counts_df["hour"] + counts_df["minute"] / 60

    records = []

    for (colony_id, date), day_df in counts_df.groupby(["colony_id", "date"]):
        day_df = day_df.sort_values(["hour", "minute"])

        first_departure = day_df.loc[day_df["exit"] > 0, "time_h"].min()
        last_return = day_df.loc[day_df["entrance"] > 0, "time_h"].max()

        trip_window = (
            last_return - first_departure
            if pd.notna(first_departure) and pd.notna(last_return)
            else np.nan
        )

        total_exits = day_df["exit"].sum()
        total_entrances = day_df["entrance"].sum()
        homing_failure = max(0, total_exits - total_entrances) / total_exits if total_exits > 0 else 0.0

        for _, row in day_df.iterrows():
            exits = row["exit"]
            entrances = row["entrance"]
            records.append({
                "colony_id": colony_id,
                "date": date,
                "hour": row["hour"],
                "minute": row["minute"],
                "exit_entry_ratio": exits / entrances if entrances > 0 else np.nan,
                "mean_flight_time": INTERVAL_SECONDS / exits if exits > 0 else np.nan,
                "first_departure": first_departure,
                "last_return": last_return,
                "trip_window": trip_window,
                "homing_failure_rate": homing_failure,
            })

    return pd.DataFrame(records)


def yin_f0(frame, sr):
    tau_min = int(sr / YIN_FMAX)
    tau_max = int(sr / YIN_FMIN)
    n = len(frame)

    if tau_max >= n:
        return np.nan

    df = np.zeros(tau_max)
    for tau in range(1, tau_max):
        diff = frame[:n - tau] - frame[tau:]
        df[tau] = np.sum(diff ** 2)

    cmndf = np.zeros(tau_max)
    cmndf[0] = 1.0
    running_sum = 0.0
    for tau in range(1, tau_max):
        running_sum += df[tau]
        cmndf[tau] = df[tau] * tau / running_sum if running_sum > 0 else 1.0

    for tau in range(tau_min, tau_max - 1):
        if cmndf[tau] < YIN_THRESHOLD and cmndf[tau] < cmndf[tau + 1]:
            return float(sr / tau)

    return float(sr / (np.argmin(cmndf[tau_min:tau_max]) + tau_min))


def extract_acoustic_features(audio_path):
    y, sr = sf.read(audio_path)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != SAMPLE_RATE:
        y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
        sr = SAMPLE_RATE

    rms_global = np.sqrt(np.mean(y ** 2))
    if rms_global > 0:
        y = y / rms_global

    b, a = butter(4, WINGBEAT_LOW_HZ / (sr / 2), btype="high")
    y = filtfilt(b, a, y)

    interval_samples = INTERVAL_SECONDS * sr
    n_intervals = len(y) // interval_samples
    records = []

    for i in range(n_intervals):
        segment = y[i * interval_samples: (i + 1) * interval_samples]

        magnitude = np.abs(librosa.stft(segment, n_fft=N_FFT, hop_length=HOP_LENGTH, window="hann"))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)

        frames = librosa.util.frame(segment, frame_length=N_FFT, hop_length=HOP_LENGTH).T
        f0_vals = np.array([yin_f0(f, sr) for f in frames])
        f0_vals = f0_vals[np.isfinite(f0_vals)]
        f0_mean = float(np.mean(f0_vals)) if len(f0_vals) > 0 else np.nan
        f0_stability = float(np.std(f0_vals)) if len(f0_vals) > 0 else np.nan

        if np.isfinite(f0_mean) and f0_mean > 0:
            harmonic_freqs = [f0_mean * k for k in range(1, N_HARMONICS + 1) if f0_mean * k < sr / 2]
            harmonic_power = sum(magnitude[np.argmin(np.abs(freqs - hf)), :].mean() for hf in harmonic_freqs)
            total_power = magnitude.mean() * magnitude.shape[0]
            harmonic_ratio = harmonic_power / total_power if total_power > 0 else np.nan
        else:
            harmonic_ratio = np.nan

        power = magnitude ** 2
        power_sum = power.sum(axis=0) + 1e-10
        spectral_centroid = float(np.mean((freqs[:, None] * power).sum(axis=0) / power_sum))
        spectral_bandwidth = float(np.mean(np.sqrt(((freqs[:, None] - spectral_centroid) ** 2 * power).sum(axis=0) / power_sum)))

        prev_mag = np.roll(magnitude, 1, axis=1)
        prev_mag[:, 0] = 0

        records.append({
            "interval_index": i,
            "f0": f0_mean,
            "f0_stability": f0_stability,
            "harmonic_ratio": harmonic_ratio,
            "spectral_centroid": spectral_centroid,
            "spectral_bandwidth": spectral_bandwidth,
            "spectral_flux": float(np.mean(np.sum((magnitude - prev_mag) ** 2, axis=0))),
            "RMS_Energy": float(np.sqrt(np.mean(segment ** 2))),
            "zero_crossing_rate": float(np.mean(librosa.feature.zero_crossing_rate(segment, hop_length=HOP_LENGTH))),
        })

    return records


if __name__ == "__main__":
    print("Computing foraging features...")
    counts_df = pd.read_csv(COUNTS_CSV)
    foraging_df = compute_foraging_features(counts_df)

    print("Extracting acoustic features...")
    acoustic_records = []
    for audio_file in sorted(Path(AUDIO_DIR).glob("*.wav")):
        colony_id, date = audio_file.stem.rsplit("_", 1)
        records = extract_acoustic_features(str(audio_file))
        for r in records:
            r["colony_id"] = colony_id
            r["date"] = date
        acoustic_records.extend(records)
        print(f"  {audio_file.name}: {len(records)} intervals")

    acoustic_df = pd.DataFrame(acoustic_records)
    merged = foraging_df.merge(acoustic_df, on=["colony_id", "date"], how="inner")
    merged.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved {len(merged)} records to {OUTPUT_CSV}")
