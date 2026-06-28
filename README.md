**A Weather-Aware Multimodal System for Pesticide Exposure Detection Using Honey Bees as Biosensors**


This repository is the official implementation of "A Weather-Aware Multimodal System for Pesticide Exposure Detection Using Honey Bees as Biosensors" 



## Requirements

To install requirements, run:

```
pip install -r requirements.txt
```

with Python 3.9+.

## Data

Data were collected from 32 honey bee colonies (16 control, 16 bifenthrin-treated) in Santa Clara County, California, monitored continuously over 15 days. Raw video and audio files are not publicly available due to size (~2 TB). A processed feature CSV (`data/sample/ApiSense_v17.csv`) with 34,560 records is provided for running the model directly.

To obtain input features from raw video and audio, annotate hive entrance frames using [Label Studio](https://labelstud.io/) and export in YOLO format, then run `extract_features.py`.

An example feature CSV from one colony-day is in `data/sample/`.

## Data Preprocessing

To run the full pipeline from raw video and audio to feature CSV:

```
python extract_features.py --video_dir data/raw/videos/ \
                              --audio_dir data/raw/audio/ \
                              --weather_csv data/raw/weather.csv \
                              --train_yolo --annotations data/raw/annotations/ \
                              --output data/sample/ApiSense_v17.csv
```

## Training

To train WAM-Net with leave-one-out cross-validation and run all ablations and baselines:

```
python train.py --data_path data/sample/ApiSense_v17.csv --seed 42
```

or use:

```
python train.py
```

for default settings. Hyperparameters are defined in `model.py`.

## Evaluation

To reproduce all tables and figures from the paper:

```
python evaluation.py --data_path data/sample/ApiSense_v17.csv --out_dir figures/
```

This produces:
- Table 3: per-day detection performance (printed to stdout)
- Figure 5: confusion matrices at Days 2, 3, 4, and overall
- Figure 6: ROC curves
- Figure 7: temporal feature attribution and overall feature importance (Integrated Gradients)
- Figure 8: feature trajectories with 95% CI bands
- Figure 9: UMAP projection across three exposure phases
- Figure 10: Spearman weather-feature correlations
- Figure S2: pairwise Pearson correlation matrix

## References

If you use this code in your research, please cite our paper:

```
@article{liang2026apisense,
  title={A Weather-Aware Multimodal System for Pesticide Exposure Detection Using Honey Bees as Biosensors},
  author={Liang, Ethan},
  journal={Sensors},
  publisher={MDPI},
  year={2026},
  note={Submitted}
}
```

These resources were used within the code:

- **YOLOv8** [paper](https://arxiv.org/abs/2304.00501) [code](https://github.com/ultralytics/ultralytics)
- **OC-SORT** [paper](https://arxiv.org/abs/2203.14360) [code](https://github.com/noahcao/OC_SORT)
- **librosa** [paper](https://zenodo.org/record/591533) [code](https://github.com/librosa/librosa)
- **Captum (Integrated Gradients)** [paper](https://arxiv.org/abs/2009.07896) [code](https://github.com/pytorch/captum)

## Contact for Questions

Ethan Liang, elcago2020@gmail.com
