# ApiSense: Pesticide Detection Using Honey Bees as Biosensors

Honey bees forage up to 95 km² daily and are sensitive to sublethal pesticide concentrations, making them naturally distributed sensors for environmental contamination. ApiSense is a multi-sensor field platform that monitors bee foraging behavior, colony acoustics, and weather conditions continuously, and uses a weather-aware multimodal neural network (WAM-Net) to detect bifenthrin exposure before visible colony damage occurs.

---

## How It Works

A camera was mounted above the hive entrance to collect bee foraging activity, a microphone was inserted into the hive to collect bee acoustics, and a nearby weather station was used to collect weather data. Then the three data streams were fed into WAM-Net, which includes self-attention encoders to track long-term behavioral changes, a weather gating layer that filters out behavioral changes driven by weather, and a bidirectional cross-attention mechanism that combines foraging and acoustic signals. Finally, the model output a bifenthrin exposure detection.

---

## Components

**1. Bee Counting Pipeline** — `bee_counting.py`

Detects bees in each frame using a fine-tuned YOLOv8s model, tracks individual bees across frames using OC-SORT, and classifies each trajectory as an entrance or exit using a virtual counting line in front of the hive entrance.


**2. Feature Extraction** — `features.py`

Extracts six foraging features including activity levels, flight trip timing, and homing failure rate, and eight acoustic features including wingbeat frequency stability, spectral structure, and energy.

**3. Training** — `train.py`

Trains the WAM-Net model.


**4. Inference** — `inference.py`

Predicts bifenthrin exposure using the trained WAM-Net model.

**5. Ablation Study** — `ablation.py`

Removes each architectural component in turn, including cross-attention, weather gate, and weather input, and evaluates single-modality models to quantify each component's contribution.

**6. Baseline Comparison** — `baselines.py`

Compares with WAM-Net against LSTM, GRU, TCN, SVM, XGBoost, and Random Forest, all trained on the same input features.

**7. Feature Importance** — `interpretability.py`

Computes integrated gradients (IG) attribution scores for each feature throughout the full exposure window. Identifies f0 stability and exit-entry ratio as the earliest behavioral biomarkers of bifenthrin exposure.



---

## Results

| Model | Mean Accuracy | Days to 75% Accuracy |
|---|---|---|
| **WAM-Net (Ours)** | **79.4%** | **2** |
| LSTM | 74.2% | 4 |
| GRU | 73.8% | 4 |
| TCN | 73.5% | 5 |
| SVM | 70.9% | — |
| XGBoost | 69.1% | — |
| Random Forest | 66.6% | — |

---

## Data

The bee detection training data is available on Kaggle: https://www.kaggle.com/datasets/rowlettheowlet/bee-detection-from-video-frames

The processed feature dataset contains 25,920 records across 24 colonies and 15 days and is included in this repository.

---

## Installation

```bash
pip install -r requirements.txt
```

Requirements: PyTorch, ultralytics, librosa, soundfile, scipy, scikit-learn, xgboost, opencv-python, pandas, numpy

