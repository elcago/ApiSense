import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from train import build_sequences, FORAGING_COLS, ACOUSTIC_COLS, WEATHER_COLS, LABEL_COL, fit_scalers
from train import LR, BATCH_SIZE, N_EPOCHS, PATIENCE, N_SEEDS, collate

DATA_CSV = "data/ApiSense_v17.csv"
HIDDEN_DIM = 16
N_HEADS = 2
DROPOUT = 0.5


class NoCrossAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.foraging_proj = nn.Linear(len(FORAGING_COLS), HIDDEN_DIM)
        self.acoustic_proj = nn.Linear(len(ACOUSTIC_COLS), HIDDEN_DIM)
        self.weather_enc = nn.Linear(len(WEATHER_COLS), len(WEATHER_COLS))
        self.foraging_attn = nn.MultiheadAttention(HIDDEN_DIM, N_HEADS, batch_first=True)
        self.acoustic_attn = nn.MultiheadAttention(HIDDEN_DIM, N_HEADS, batch_first=True)
        self.foraging_gate = nn.Linear(len(WEATHER_COLS), HIDDEN_DIM)
        self.audio_gate = nn.Linear(len(WEATHER_COLS), HIDDEN_DIM)
        self.classifier = nn.Sequential(
            nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM), nn.ReLU(),
            nn.Dropout(DROPOUT), nn.Linear(HIDDEN_DIM, 1), nn.Sigmoid(),
        )

    def forward(self, foraging, acoustic, weather):
        zw = self.weather_enc(weather.mean(dim=1))
        zf = self.foraging_proj(foraging)
        zf, _ = self.foraging_attn(zf, zf, zf)
        zf = zf.mean(dim=1) * torch.sigmoid(self.foraging_gate(zw))
        za = self.acoustic_proj(acoustic)
        za, _ = self.acoustic_attn(za, za, za)
        za = za.mean(dim=1) * torch.sigmoid(self.audio_gate(zw))
        return self.classifier(torch.cat([zf, za], dim=-1)).squeeze(-1)


class NoWeatherGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.foraging_proj = nn.Linear(len(FORAGING_COLS), HIDDEN_DIM)
        self.acoustic_proj = nn.Linear(len(ACOUSTIC_COLS), HIDDEN_DIM)
        self.foraging_attn = nn.MultiheadAttention(HIDDEN_DIM, N_HEADS, batch_first=True)
        self.acoustic_attn = nn.MultiheadAttention(HIDDEN_DIM, N_HEADS, batch_first=True)
        self.fwd_attn = nn.MultiheadAttention(HIDDEN_DIM, 1, batch_first=True)
        self.bwd_attn = nn.MultiheadAttention(HIDDEN_DIM, 1, batch_first=True)
        self.fuse = nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM * 2)
        self.classifier = nn.Sequential(
            nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM), nn.ReLU(),
            nn.Dropout(DROPOUT), nn.Linear(HIDDEN_DIM, 1), nn.Sigmoid(),
        )

    def forward(self, foraging, acoustic, weather):
        zf = self.foraging_proj(foraging)
        zf, _ = self.foraging_attn(zf, zf, zf)
        zf = zf.mean(dim=1).unsqueeze(1)
        za = self.acoustic_proj(acoustic)
        za, _ = self.acoustic_attn(za, za, za)
        za = za.mean(dim=1).unsqueeze(1)
        fwd, _ = self.fwd_attn(zf, za, za)
        bwd, _ = self.bwd_attn(za, zf, zf)
        fused = self.fuse(torch.cat([fwd.squeeze(1), bwd.squeeze(1)], dim=-1))
        return self.classifier(fused).squeeze(-1)


class NoWeatherInput(nn.Module):
    def __init__(self):
        super().__init__()
        self.foraging_proj = nn.Linear(len(FORAGING_COLS), HIDDEN_DIM)
        self.acoustic_proj = nn.Linear(len(ACOUSTIC_COLS), HIDDEN_DIM)
        self.foraging_attn = nn.MultiheadAttention(HIDDEN_DIM, N_HEADS, batch_first=True)
        self.acoustic_attn = nn.MultiheadAttention(HIDDEN_DIM, N_HEADS, batch_first=True)
        self.foraging_gate = nn.Linear(len(WEATHER_COLS), HIDDEN_DIM)
        self.audio_gate = nn.Linear(len(WEATHER_COLS), HIDDEN_DIM)
        self.fwd_attn = nn.MultiheadAttention(HIDDEN_DIM, 1, batch_first=True)
        self.bwd_attn = nn.MultiheadAttention(HIDDEN_DIM, 1, batch_first=True)
        self.fuse = nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM * 2)
        self.classifier = nn.Sequential(
            nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM), nn.ReLU(),
            nn.Dropout(DROPOUT), nn.Linear(HIDDEN_DIM, 1), nn.Sigmoid(),
        )

    def forward(self, foraging, acoustic, weather):
        zw = torch.zeros(weather.size(0), weather.size(2), device=weather.device)
        zf = self.foraging_proj(foraging)
        zf, _ = self.foraging_attn(zf, zf, zf)
        zf = zf.mean(dim=1) * torch.sigmoid(self.foraging_gate(zw))
        za = self.acoustic_proj(acoustic)
        za, _ = self.acoustic_attn(za, za, za)
        za = za.mean(dim=1) * torch.sigmoid(self.audio_gate(zw))
        fwd, _ = self.fwd_attn(zf.unsqueeze(1), za.unsqueeze(1), za.unsqueeze(1))
        bwd, _ = self.bwd_attn(za.unsqueeze(1), zf.unsqueeze(1), zf.unsqueeze(1))
        fused = self.fuse(torch.cat([fwd.squeeze(1), bwd.squeeze(1)], dim=-1))
        return self.classifier(fused).squeeze(-1)


class AudioOnly(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(len(ACOUSTIC_COLS), HIDDEN_DIM)
        self.attn = nn.MultiheadAttention(HIDDEN_DIM, N_HEADS, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ReLU(),
            nn.Dropout(DROPOUT), nn.Linear(HIDDEN_DIM, 1), nn.Sigmoid(),
        )

    def forward(self, foraging, acoustic, weather):
        z, _ = self.attn(self.proj(acoustic), self.proj(acoustic), self.proj(acoustic))
        return self.classifier(z.mean(dim=1)).squeeze(-1)


class ForagingOnly(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(len(FORAGING_COLS), HIDDEN_DIM)
        self.attn = nn.MultiheadAttention(HIDDEN_DIM, N_HEADS, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ReLU(),
            nn.Dropout(DROPOUT), nn.Linear(HIDDEN_DIM, 1), nn.Sigmoid(),
        )

    def forward(self, foraging, acoustic, weather):
        z, _ = self.attn(self.proj(foraging), self.proj(foraging), self.proj(foraging))
        return self.classifier(z.mean(dim=1)).squeeze(-1)


ABLATION_MODELS = {
    "w/o Cross-Attention": NoCrossAttn,
    "w/o Weather Gate":    NoWeatherGate,
    "w/o Weather Input":   NoWeatherInput,
    "Audio-Only":          AudioOnly,
    "Foraging-Only":       ForagingOnly,
}


def train_fold(model_cls, train_data, test_colony, device, seed=42):
    torch.manual_seed(seed)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)

    model = model_cls().to(device)
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
            for f, a, w, labels in train_loader:
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

    f, a, w, label = test_colony
    with torch.no_grad():
        p = model(f.unsqueeze(0).to(device), a.unsqueeze(0).to(device), w.unsqueeze(0).to(device)).item()
    pred = int(p > 0.5)

    return {"accuracy": int(pred == label), "pred": pred, "label": label}


if __name__ == "__main__":
    import pathlib
    pathlib.Path("results").mkdir(exist_ok=True)

    df = pd.read_csv(DATA_CSV)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    colony_ids = df["colony_id"].unique()
    summary = []

    for model_name, model_cls in ABLATION_MODELS.items():
        print(f"\n{model_name}")
        all_preds, all_labels = [], []

        for test_id in colony_ids:
            train_ids = [c for c in colony_ids if c != test_id]
            scalers = fit_scalers(df, train_ids)
            train_data = [build_sequences(df, cid, scalers) for cid in train_ids]
            test_colony = build_sequences(df, test_id, scalers)

            seed_results = [train_fold(model_cls, train_data, test_colony, device, seed=s) for s in range(N_SEEDS)]
            votes = [r["pred"] for r in seed_results]
            all_preds.append(int(np.round(np.mean(votes))))
            all_labels.append(test_colony[3])

        result = {
            "model": model_name,
            "accuracy": accuracy_score(all_labels, all_preds),
            "precision": precision_score(all_labels, all_preds, zero_division=0),
            "recall": recall_score(all_labels, all_preds, zero_division=0),
            "f1": f1_score(all_labels, all_preds, zero_division=0),
        }
        summary.append(result)
        print(f"  acc={result['accuracy']:.3f}  f1={result['f1']:.3f}")

    results_df = pd.DataFrame(summary).set_index("model")
    results_df.to_csv("results/ablation_results.csv")
    print("\nAblation Summary:")
    print(results_df.round(4))
