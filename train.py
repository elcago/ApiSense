import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

DATA_CSV = "data/ApiSense_v17.csv"
CHECKPOINT_DIR = "checkpoints"

FORAGING_COLS = [
    "exit_entry_ratio", "mean_trip_duration",
    "first_departure", "last_return", "trip_window", "homing_failure_rate",
]
ACOUSTIC_COLS = [
    "f0", "f0_stability", "harmonic_ratio",
    "spectral_centroid", "spectral_bandwidth", "spectral_flux",
    "rms_energy", "zero_crossing_rate",
]
WEATHER_COLS = ["temperature", "humidity", "wind_speed", "precipitation"]
LABEL_COL = "exposed"

LR = 1e-4
BATCH_SIZE = 16
N_EPOCHS = 100
PATIENCE = 20
N_SEEDS = 10
VAL_FRACTION = 0.2

ENCODER_HIDDEN_DIM = 16
ENCODER_N_HEADS = 2
CLASSIFIER_HIDDEN_DIM = 128
CLASSIFIER_DROPOUT = 0.5


class SelfAttentionEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=ENCODER_HIDDEN_DIM, n_heads=ENCODER_N_HEADS):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.self_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=n_heads, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        h = self.input_proj(x)
        attn_out, _ = self.self_attn(h, h, h)
        h = self.norm1(h + attn_out)
        h = self.norm2(h + self.feed_forward(h))
        return h.mean(dim=1)


class WeatherEncoder(nn.Module):
    def __init__(self, input_dim=len(WEATHER_COLS), output_dim=len(WEATHER_COLS)):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, w):
        return self.linear(w.mean(dim=1))


class WeatherGate(nn.Module):
    def __init__(self, weather_dim, modality_dim):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(weather_dim, modality_dim), nn.Sigmoid())

    def forward(self, z_weather, z_modality):
        return z_modality * self.gate(z_weather)


class CrossModalAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.scale = dim ** 0.5

    def forward(self, z_query_src, z_kv_src):
        q = self.query(z_query_src)
        k = self.key(z_kv_src)
        v = self.value(z_kv_src)
        score = (q * k).sum(dim=-1, keepdim=True) / self.scale
        weight = torch.softmax(score, dim=-1)
        return weight * v


class WAMNet(nn.Module):
    def __init__(
        self,
        foraging_dim=len(FORAGING_COLS),
        acoustic_dim=len(ACOUSTIC_COLS),
        weather_dim=len(WEATHER_COLS),
        hidden_dim=ENCODER_HIDDEN_DIM,
        classifier_hidden_dim=CLASSIFIER_HIDDEN_DIM,
        dropout=CLASSIFIER_DROPOUT,
        use_weather_gating=True,
        use_cross_attention=True,
    ):
        super().__init__()
        self.use_weather_gating = use_weather_gating
        self.use_cross_attention = use_cross_attention

        self.foraging_encoder = SelfAttentionEncoder(foraging_dim, hidden_dim)
        self.acoustic_encoder = SelfAttentionEncoder(acoustic_dim, hidden_dim)
        self.weather_encoder = WeatherEncoder(weather_dim)

        if use_weather_gating:
            self.foraging_gate = WeatherGate(weather_dim, hidden_dim)
            self.acoustic_gate = WeatherGate(weather_dim, hidden_dim)

        if use_cross_attention:
            self.cross_forward = CrossModalAttention(hidden_dim)
            self.cross_backward = CrossModalAttention(hidden_dim)

        combined_dim = hidden_dim * 2

        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, classifier_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, foraging, acoustic, weather):
        z_foraging = self.foraging_encoder(foraging)
        z_acoustic = self.acoustic_encoder(acoustic)

        if self.use_weather_gating:
            z_weather = self.weather_encoder(weather)
            z_foraging = self.foraging_gate(z_weather, z_foraging)
            z_acoustic = self.acoustic_gate(z_weather, z_acoustic)

        if self.use_cross_attention:
            z_fwd = self.cross_forward(z_foraging, z_acoustic)
            z_bwd = self.cross_backward(z_acoustic, z_foraging)
            z_combined = torch.cat([z_fwd, z_bwd], dim=-1)
        else:
            z_combined = torch.cat([z_foraging, z_acoustic], dim=-1)

        return self.classifier(z_combined).squeeze(-1)


def build_sequences(df, colony_id):
    colony = df[df["colony_id"] == colony_id].sort_values("interval_start")
    foraging = torch.tensor(colony[FORAGING_COLS].fillna(0).values, dtype=torch.float32)
    acoustic = torch.tensor(colony[ACOUSTIC_COLS].fillna(0).values, dtype=torch.float32)
    weather = torch.tensor(colony[WEATHER_COLS].fillna(0).values, dtype=torch.float32)
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

    return {
        "accuracy": accuracy_score(labels_all, preds),
        "precision": precision_score(labels_all, preds, zero_division=0),
        "recall": recall_score(labels_all, preds, zero_division=0),
        "f1": f1_score(labels_all, preds, zero_division=0),
        "model_state": best_state,
    }


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
        train_data = [build_sequences(df, cid) for cid in train_ids]
        test_colony = build_sequences(df, test_id)

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
