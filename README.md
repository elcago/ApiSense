# ApiSense: Pesticide Detection Using Honey Bees as Biosensors

Honey bees forage up to 95 km² daily and are highly sensitive to sublethal pesticide concentrations, making them natural distributed sensors for environmental contamination. ApiSense is a solar-powered field platform that monitors bee foraging behavior and colony acoustics continuously, and uses a weather-aware deep learning model (WAM-Net) to detect bifenthrin exposure before visible colony damage occurs.

---

## How It Works

A camera was mounted above the hive entrance, a microphone was inserted into the hive, and weather data was measured from a nearby weather station. The three data streams were synchronized and fed into WAM-Net to detect pesticide exposure. Because bees respond to bad weather similarly to pesticide exposure, WAM-Net includes a weather-gating layer that filters out behavioral changes driven by weather before making an exposure prediction. A bidirectional cross-attention module then combines foraging and acoustic signals into a single classification.

---

## Components

**1. Bee Counting Pipeline** — `bee_counting.py`

Detects bees at the hive entrance with a fine-tuned YOLOv8s model (94.8% precision, 98.6% mAP@0.5), tracks individuals across frames with OC-SORT (88.7% MOTA), and classifies each trajectory as an entrance or exit using a virtual counting line (F1: 93.3% in / 91.7% out).

**2. Feature Extraction** — `features.py`

Extracts six foraging features capturing activity levels, trip timing, and homing failure rate, and eight acoustic features capturing wingbeat frequency stability, spectral structure, and energy.

**3. Training** — `train.py`

Trains with leave-one-out cross-validation across all 32 colonies, averaged over 10 random seeds.


**4. Inference** — `inference.py`

Predicts bifenthrin exposure using three stages: modality-specific self-attention encoders to capture cumulative behavioral changes over time, a weather-gating mechanism to separate pesticide-driven changes from weather-driven ones, and bidirectional cross-modal attention to jointly model foraging and acoustic signals.

**5. Ablation Study** — `ablation.py`

Removes each architectural component in turn — cross-attention, weather gate, and weather input — and evaluates single-modality models to quantify each component's contribution.

**6. Baseline Comparison** — `baselines.py`

Benchmarks WAM-Net against LSTM, GRU, TCN, SVM, XGBoost, and Random Forest, all trained on the same 14 input features.

**7. Feature Importance** — `interpretability.py`

Computes Integrated Gradients attribution scores across the 15-day exposure window. Identifies f0 stability and exit-entry ratio as the earliest behavioral biomarkers of bifenthrin exposure, emerging before disrupted foraging schedules and homing failure.

---

## Results

| Model | Mean Accuracy | Days to 75% Accuracy |
|---|---|---|
| **WAM-Net (Ours)** | **80.9%** | **2** |
| LSTM | 73.2% | 4 |
| GRU | 72.3% | 4 |
| TCN | 71.5% | 5 |
| SVM | 68.5% | — |
| XGBoost | 67.5% | — |
| Random Forest | 65.1% | — |

---

## Installation

```bash
pip install -r requirements.txt
```

Requirements: PyTorch, ultralytics, librosa, soundfile, scipy, scikit-learn, xgboost, opencv-python, pandas, numpy

