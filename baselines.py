import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
from train import build_sequences, FORAGING_COLS, ACOUSTIC_COLS, WEATHER_COLS, LABEL_COL, SORT_COLS, fit_scalers
from train import LR, BATCH_SIZE, N_EPOCHS, PATIENCE, N_SEEDS, VAL_FRACTION

torch.set_num_threads(1)


DATA_CSV = "data/ApiSense_v17.csv"
ALL_FEATURE_COLS = FORAGING_COLS + ACOUSTIC_COLS + WEATHER_COLS
INPUT_DIM = len(ALL_FEATURE_COLS)
HIDDEN_DIM = 32
DROPOUT = 0.5
RF_N_ESTIMATORS = 100
XGB_N_ESTIMATORS = 100
XGB_MAX_DEPTH = 4


class LSTMBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(INPUT_DIM, HIDDEN_DIM, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Dropout(DROPOUT), nn.Linear(HIDDEN_DIM, 1), nn.Sigmoid()
        )

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.classifier(h.squeeze(0)).squeeze(-1)


class GRUBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(INPUT_DIM, HIDDEN_DIM, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Dropout(DROPOUT), nn.Linear(HIDDEN_DIM, 1), nn.Sigmoid()
        )

    def forward(self, x):
        _, h = self.gru(x)
        return self.classifier(h.squeeze(0)).squeeze(-1)


class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=padding)
        self.relu = nn.ReLU()
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        return self.relu(self.conv(x)[:, :, :x.size(2)] + self.residual(x))


class TCNBaseline(nn.Module):
    def __init__(self, n_layers=4):
        super().__init__()
        layers = []
        for i in range(n_layers):
            in_ch = INPUT_DIM if i == 0 else HIDDEN_DIM
            layers.append(TCNBlock(in_ch, HIDDEN_DIM, dilation=2 ** i))
        self.tcn = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Dropout(DROPOUT), nn.Linear(HIDDEN_DIM, 1), nn.Sigmoid()
        )

    def forward(self, x):
        return self.classifier(self.tcn(x.permute(0, 2, 1))[:, :, -1]).squeeze(-1)


SEQUENTIAL_MODELS = {
    "LSTM": LSTMBaseline,
    "GRU": GRUBaseline,
    "TCN": TCNBaseline,
}

SKLEARN_MODELS = {
    "SVM": lambda: SVC(kernel="rbf", probability=True),
    "XGBoost": lambda: XGBClassifier(n_estimators=XGB_N_ESTIMATORS, max_depth=XGB_MAX_DEPTH, use_label_encoder=False, eval_metric="logloss", verbosity=0),
    "Random Forest": lambda: RandomForestClassifier(n_estimators=RF_N_ESTIMATORS, random_state=42),
}


def build_flat_sequence(df, colony_id, scalers=None):
    colony = df[df["colony_id"] == colony_id].sort_values(SORT_COLS)

    if scalers is not None:
        foraging = scalers["foraging"].transform(colony[FORAGING_COLS].fillna(0).values)
        acoustic = scalers["acoustic"].transform(colony[ACOUSTIC_COLS].fillna(0).values)
        weather = scalers["weather"].transform(colony[WEATHER_COLS].fillna(0).values)
        x = torch.tensor(np.concatenate([foraging, acoustic, weather], axis=1), dtype=torch.float32)
    else:
        x = torch.tensor(colony[ALL_FEATURE_COLS].fillna(0).values, dtype=torch.float32)

    label = int(colony[LABEL_COL].iloc[0])
    return x, label


def collate_flat(batch):
    x = torch.stack([b[0] for b in batch])
    l = torch.tensor([b[1] for b in batch], dtype=torch.float32)
    return x, l


def train_sequential_fold(model_cls, train_data, test_sample, device, seed=42):
    torch.manual_seed(seed)

    n_val = max(1, int(len(train_data) * VAL_FRACTION))
    rng = np.random.RandomState(seed)
    val_idx = set(rng.choice(len(train_data), size=n_val, replace=False))
    fit_data = [x for i, x in enumerate(train_data) if i not in val_idx]
    val_data = [x for i, x in enumerate(train_data) if i in val_idx]

    train_loader = DataLoader(fit_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_flat)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_flat)

    model = model_cls().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.BCELoss()

    best_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(N_EPOCHS):
        model.train()
        for x, labels in train_loader:
            x, labels = x.to(device), labels.to(device)
            optimizer.zero_grad()
            criterion(model(x), labels).backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, labels in val_loader:
                x, labels = x.to(device), labels.to(device)
                val_loss += criterion(model(x), labels).item()

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

    x_test, label = test_sample
    with torch.no_grad():
        p = model(x_test.unsqueeze(0).to(device)).item()
    pred = int(p > 0.5)

    del model, optimizer, train_loader, val_loader, best_state
    gc.collect()

    return pred, label


if __name__ == "__main__":
    import pathlib
    pathlib.Path("results").mkdir(exist_ok=True)

    df = pd.read_csv(DATA_CSV)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    colony_ids = df["colony_id"].unique()
    summary = []

    for model_name, model_cls in SEQUENTIAL_MODELS.items():
        print(f"\n{model_name}")
        all_preds, all_labels = [], []

        for test_id in colony_ids:
            train_ids = [c for c in colony_ids if c != test_id]
            scalers = fit_scalers(df, train_ids)
            train_data = [build_flat_sequence(df, cid, scalers) for cid in train_ids]
            test_sample = build_flat_sequence(df, test_id, scalers)

            votes = [train_sequential_fold(model_cls, train_data, test_sample, device, seed=s)[0] for s in range(N_SEEDS)]
            all_preds.append(int(np.round(np.mean(votes))))
            all_labels.append(test_sample[1])
            del train_data, test_sample
            gc.collect()

        result = {
            "model": model_name,
            "accuracy": accuracy_score(all_labels, all_preds),
            "precision": precision_score(all_labels, all_preds, zero_division=0),
            "recall": recall_score(all_labels, all_preds, zero_division=0),
            "f1": f1_score(all_labels, all_preds, zero_division=0),
        }
        summary.append(result)
        print(f"  acc={result['accuracy']:.3f}  f1={result['f1']:.3f}")

    for model_name, model_factory in SKLEARN_MODELS.items():
        print(f"\n{model_name}")
        all_preds, all_labels = [], []

        for test_id in colony_ids:
            train_ids = [c for c in colony_ids if c != test_id]
            X_train = np.array([df[df["colony_id"] == cid][ALL_FEATURE_COLS].fillna(0).values.flatten() for cid in train_ids])
            y_train = [int(df[df["colony_id"] == cid][LABEL_COL].iloc[0]) for cid in train_ids]
            X_test = df[df["colony_id"] == test_id][ALL_FEATURE_COLS].fillna(0).values.flatten().reshape(1, -1)
            y_test = int(df[df["colony_id"] == test_id][LABEL_COL].iloc[0])

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            clf = model_factory()
            clf.fit(X_train, y_train)
            all_preds.append(clf.predict(X_test)[0])
            all_labels.append(y_test)

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
    results_df.to_csv("results/baseline_results.csv")
    print("\nBaseline Summary:")
    print(results_df.round(4))
