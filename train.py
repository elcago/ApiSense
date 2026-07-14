import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

torch.set_num_threads(1)

DATA_CSV = "data/ApiSense_v17.csv"
CHECKPOINT_DIR = "checkpoints"

FORAGING_COLS = [
    "exit_entry_ratio", "mean_flight_time",
    "first_departure", "last_return", "trip_window", "homing_failure_rate",
]
ACOUSTIC_COLS = [
    "f0", "f0_stability", "harmonic_ratio",
    "spectral_centroid", "spectral_bandwidth", "spectral_flux",
    "RMS_Energy", "zero_crossing_rate",
]
WEATHER_COLS = ["temp_C", "humidity_pct", "wind_ms", "rain_in"]
LABEL_COL = "bifenthrin"
SORT_COLS = ["day_num", "hour", "minute"]

LR = 1e-4
BATCH_SIZE = 16
N_EPOCHS = 100
PATIENCE = 20
N_SEEDS = 10
VAL_FRACTION = 0.2

HIDDEN_DIM = 16
N_HEADS = 2
DROPOUT = 0.5


class ModalityEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=HIDDEN_DIM, n_heads=N_HEADS):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        x = self.input_proj(x)
        attn_out, _ = self.attn(x, x, x)
        return self.norm(x + attn_out).mean(dim=1)


class WeatherGate(nn.Module):
    def __init__(self, weather_dim=len(WEATHER_COLS), hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.foraging_gate = nn.Linear(weather_dim, hidden_dim)
        self.audio_gate = nn.Linear(weather_dim, hidden_dim)

    def forward(self, z_foraging, z_audio, z_weather):
        return (
            z_foraging * torch.sigmoid(self.foraging_gate(z_weather)),
            z_audio * torch.sigmoid(self.audio_gate(z_weather)),
        )


class BidirectionalCrossAttention(nn.Module):
    def __init__(self, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.fwd_attn = nn.MultiheadAttention(hidden_dim, num_heads=1, batch_first=True)
        self.bwd_attn = nn.MultiheadAttention(hidden_dim, num_heads=1, batch_first=True)
        self.fuse = nn.Linear(hidden_dim * 2, hidden_dim * 2)

    def forward(self, z_foraging, z_audio):
        zf = z_foraging.unsqueeze(1)
        za = z_audio.unsqueeze(1)
        fwd, _ = self.fwd_attn(query=zf, key=za, value=za)
        bwd, _ = self.bwd_attn(query=za, key=zf, value=zf)
        return self.fuse(torch.cat([fwd.squeeze(1), bwd.squeeze(1)], dim=-1))


class WAMNet(nn.Module):
    def __init__(
        self,
        foraging_dim=len(FORAGING_COLS),
        acoustic_dim=len(ACOUSTIC_COLS),
        weather_dim=len(WEATHER_COLS),
        hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT,
        use_weather_gating=True,
        use_cross_attention=True,
    ):
        super().__init__()
        self.use_weather_gating = use_weather_gating
        self.use_cross_attention = use_cross_attention

        self.foraging_encoder = ModalityEncoder(foraging_dim, hidden_dim)
        self.acoustic_encoder = ModalityEncoder(acoustic_dim, hidden_dim)

        if use_weather_gating:
            self.weather_encoder = nn.Linear(weather_dim, weather_dim)
            self.weather_gate = WeatherGate(weather_dim, hidden_dim)

        if use_cross_attention:
            self.cross_attn = BidirectionalCrossAttention(hidden_dim)
            self.fuse = None
        else:
            self.fuse = None

        combined_dim = hidden_dim * 2

        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, foraging, acoustic, weather):
        z_foraging = self.foraging_encoder(foraging)
        z_acoustic = self.acoustic_encoder(acoustic)

        if self.use_weather_gating:
            z_weather = self.weather_encoder(weather.mean(dim=1))
            z_foraging, z_acoustic = self.weather_gate(z_foraging, z_acoustic, z_weather)

        if self.use_cross_attention:
            z_combined = self.cross_attn(z_foraging, z_acoustic)
        else:
            z_combined = torch.cat([z_foraging, z_acoustic], dim=-1)

        return self.classifier(z_combined).squeeze(-1)


def fit_scalers(df, train_colony_ids):
    train_rows = df[df["colony_id"].isin(train_colony_ids)]
    scalers = {}
    for name, cols in [("foraging", FORAGING_COLS), ("acoustic", ACOUSTIC_COLS), ("weather", WEATHER_COLS)]:
        scaler = StandardScaler()
        scaler.fit(train_rows[cols].fillna(0).values)
        scalers[name] = scaler
    return scalers


def build_sequences(df, colony_id, scalers=None):
    colony = df[df["colony_id"] == colony_id].sort_values(SORT_COLS)

    foraging_raw = colony[FORAGING_COLS].fillna(0).values
    acoustic_raw = colony[ACOUSTIC_COLS].fillna(0).values
    weather_raw = colony[WEATHER_COLS].fillna(0).values

    if scalers is not None:
        foraging_raw = scalers["foraging"].transform(foraging_raw)
        acoustic_raw = scalers["acoustic"].transform(acoustic_raw)
        weather_raw = scalers["weather"].transform(weather_raw)

    foraging = torch.tensor(foraging_raw, dtype=torch.float32)
    acoustic = torch.tensor(acoustic_raw, dtype=torch.float32)
    weather = torch.tensor(weather_raw, dtype=torch.float32)
    label = int(colony[LABEL_COL].iloc[0])
    return foraging, acoustic, weather, label


def collate(batch):
    f = torch.stack([x[0] for x in batch])
    a = torch.stack([x[1] for x in batch])
    w = torch.stack([x[2] for x in batch])
    l = torch.tensor([x[3] for x in batch], dtype=torch.float32)
    return f, a, w, l


def train_one_fold(train_data, test_colony, device, seed=42):
    torch.manual_seed(seed)

    n_val = max(1, int(len(train_data) * VAL_FRACTION))
    rng = np.random.RandomState(seed)
    val_idx = set(rng.choice(len(train_data), size=n_val, replace=False))
    fit_data = [x for i, x in enumerate(train_data) if i not in val_idx]
    val_data = [x for i, x in enumerate(train_data) if i in val_idx]

    train_loader = DataLoader(fit_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)
    test_loader = DataLoader([test_colony], batch_size=1, collate_fn=collate)

    model = WAMNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.BCELoss()

    best_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(N_EPOCHS):
        model.train()
        for f, a, w, labels in train_loader:
            f, a, w, labels = f.to(device), a.to(device), w.to(device), labels.to(device)
            optimizer.zero_grad()
            criterion(model(f, a, w), labels).backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for f, a, w, labels in val_loader:
                f, a, w, labels = f.to(device), a.to(device), w.to(device), labels.to(device)
                val_loss += criterion(model(f, a, w), labels).item()

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break

    model.load_state_dict(best_state)
    model.eval()

    preds, labels_all = [], []
    with torch.no_grad():
        for f, a, w, labels in test_loader:
            p = model(f.to(device), a.to(device), w.to(device)).cpu().numpy()
            preds.extend((p > 0.5).astype(int))
            labels_all.extend(labels.numpy())

    result = {
        "accuracy": accuracy_score(labels_all, preds),
        "precision": precision_score(labels_all, preds, zero_division=0),
        "recall": recall_score(labels_all, preds, zero_division=0),
        "f1": f1_score(labels_all, preds, zero_division=0),
        "model_state": {k: v.clone() for k, v in best_state.items()},
    }

    del model, optimizer, train_loader, val_loader, test_loader, best_state
    gc.collect()

    return result


if __name__ == "__main__":
    import pathlib

    pathlib.Path(CHECKPOINT_DIR).mkdir(exist_ok=True)
    pathlib.Path("results").mkdir(exist_ok=True)

    df = pd.read_csv(DATA_CSV)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    colony_ids = df["colony_id"].unique()
    all_results = []
    best_overall_f1 = -1
    best_state = None

    for test_id in colony_ids:
        train_ids = [c for c in colony_ids if c != test_id]
        scalers = fit_scalers(df, train_ids)
        train_data = [build_sequences(df, cid, scalers) for cid in train_ids]
        test_colony = build_sequences(df, test_id, scalers)

        seed_results = [train_one_fold(train_data, test_colony, device, seed=s) for s in range(N_SEEDS)]
        mean_metrics = {k: np.mean([r[k] for r in seed_results]) for k in ["accuracy", "precision", "recall", "f1"]}
        mean_metrics["colony_id"] = test_id
        all_results.append(mean_metrics)

        if mean_metrics["f1"] > best_overall_f1:
            best_overall_f1 = mean_metrics["f1"]
            best_state = seed_results[0]["model_state"]

        print(f"Colony {test_id}: acc={mean_metrics['accuracy']:.3f} f1={mean_metrics['f1']:.3f}")

    results_df = pd.DataFrame(all_results)
    results_df.to_csv("results/loocv_results.csv", index=False)
    torch.save(best_state, f"{CHECKPOINT_DIR}/wam_net_best.pt")

    print("\nOverall:")
    print(results_df[["accuracy", "precision", "recall", "f1"]].mean().round(4))
    print(f"\nBest checkpoint saved to {CHECKPOINT_DIR}/wam_net_best.pt")
