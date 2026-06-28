import torch
import torch.nn as nn
import pandas as pd
from train import build_sequences, LABEL_COL


DATA_CSV = "data/ApiSense_v17.csv"
WEIGHTS = "checkpoints/wam_net_best.pt"

FORAGING_DIM = 6
ACOUSTIC_DIM = 8
WEATHER_DIM = 4
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
    def __init__(self, weather_dim=WEATHER_DIM, hidden_dim=HIDDEN_DIM):
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
    def __init__(self):
        super().__init__()
        self.foraging_encoder = ModalityEncoder(FORAGING_DIM)
        self.acoustic_encoder = ModalityEncoder(ACOUSTIC_DIM)
        self.weather_encoder = nn.Linear(WEATHER_DIM, WEATHER_DIM)
        self.weather_gate = WeatherGate()
        self.cross_attn = BidirectionalCrossAttention()
        self.classifier = nn.Sequential(
            nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIM, 1),
            nn.Sigmoid(),
        )

    def forward(self, foraging_seq, audio_seq, weather_seq):
        z_foraging = self.foraging_encoder(foraging_seq)
        z_audio = self.acoustic_encoder(audio_seq)
        z_weather = self.weather_encoder(weather_seq.mean(dim=1))
        z_foraging, z_audio = self.weather_gate(z_foraging, z_audio, z_weather)
        fused = self.cross_attn(z_foraging, z_audio)
        return self.classifier(fused).squeeze(-1)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(DATA_CSV)

    model = WAMNet().to(device)
    model.load_state_dict(torch.load(WEIGHTS, map_location=device))
    model.eval()

    print(f"{'Colony':<12} {'p(exposed)':>12} {'Predicted':>12} {'Actual':>10}")
    print("-" * 50)

    for colony_id in df["colony_id"].unique():
        foraging, acoustic, weather, label = build_sequences(df, colony_id)
        foraging = foraging.unsqueeze(0).to(device)
        acoustic = acoustic.unsqueeze(0).to(device)
        weather = weather.unsqueeze(0).to(device)

        with torch.no_grad():
            p = model(foraging, acoustic, weather).item()

        predicted = "exposed" if p > 0.5 else "control"
        actual = "exposed" if label == 1 else "control"
        print(f"{colony_id:<12} {p:>12.3f} {predicted:>12} {actual:>10}")
