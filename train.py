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
    from wam_net import WAMNet
    torch.manual_seed(seed)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
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

        print(f"Colony {test_id}: acc={mean_metrics['accuracy']:.3f}  f1={mean_metrics['f1']:.3f}")

    results_df = pd.DataFrame(all_results)
    results_df.to_csv("results/loocv_results.csv", index=False)

    torch.save(best_state, f"{CHECKPOINT_DIR}/wam_net_best.pt")

    print("\nOverall:")
    print(results_df[["accuracy", "precision", "recall", "f1"]].mean().round(4))
    print(f"\nBest checkpoint saved to {CHECKPOINT_DIR}/wam_net_best.pt")
