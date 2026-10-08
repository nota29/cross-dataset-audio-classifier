# Cross-Dataset Audio Classifier

A multi-task classifier for everyday sounds with a **shared residual body** and two task-specific heads, trained simultaneously on the heterogeneous datasets **BSD10K** (single-label) and **FSD50K** (multi-label). 

---

## Architecture

```
Input embedding
      │
 Linear projection
      │
 N × ResidualBlock       ← shared body (GELU + LayerNorm)
      │
 ┌────┴────┐
 ▼         ▼
Head B    Head F          ← two-layer MLP per task
(BSD10K)  (FSD50K)
single-   multi-
label     label
```

**Total loss:**

```
L = w_b · CrossEntropy(task_b)
  + w_f · BCEWithLogits(task_f)
  + α   · NTXent(features_b, features_f)
```

Default weights: `w_b = 0.15`, `w_f = 1.0`, `α = 0.0001`

---

## Input Embeddings

Audio embeddings for both datasets are pre-computed using [ONE-PEACE](https://github.com/OFA-Sys/ONE-PEACE), a large multimodal foundation model. Each audio clip is represented as a **1536-dim** vector extracted from ONE-PEACE's audio encoder, stored as a `.npy` file named `{sound_id}.npy`.

---

## Datasets

| Dataset | Task | Labels | Loss |
|---|---|---|---|
| [BSD10K](https://zenodo.org/records/17250001) | Single-label classification | ~23 sound event classes | CrossEntropy |
| [FSD50K](https://zenodo.org/record/4060432) | Multi-label classification | ~200 AudioSet classes | BCEWithLogits |

---

## Evaluation Metrics

For each dataset the evaluation script reports per-class precision, recall, and F1, plus:
- **Micro-F1** — aggregated over all samples
- **Macro-F1** — average over all classes
- **Macro-mAP** — mean Average Precision across classes

Macro-mAP is the primary metric: with the default configuration, joint training reached **0.7247** on BSD10K and **0.6549** on FSD50K.


---


## Setup

```bash
git clone https://github.com/nota29/cross-dataset-audio-classifier.git
cd cross-dataset-audio-classifier
pip install -r requirements.txt
```

Place your pre-computed embedding folders and metadata CSVs under `data/` as shown above (or update paths in `configs/config.yaml`).

---

## Usage

### Train + evaluate (default config)
```bash
python main.py
```

### Custom config file
```bash
python main.py --config configs/config.yaml
```

### Override hyperparameters from the command line
```bash
python main.py --epochs 50 --lr 1e-4 --batch_size 32
```

Results are saved to `outputs/classification_report_bsd.csv` and `outputs/classification_report_fsd.csv`.

---

## Configuration

All settings live in `configs/config.yaml`. Key options:

| Section | Key | Default | Description |
|---|---|---|---|
| `training` | `num_epochs` | 50 | Training epochs |
| `training` | `learning_rate` | 1e-4 | AdamW LR |
| `training` | `task_weights.task_b` | 0.15 | BSD10K loss weight |
| `training` | `task_weights.task_f` | 1.0 | FSD50K loss weight |
| `training` | `alpha_align` | 0.0001 | NTXent loss weight |
| `training` | `ntxent_temperature` | 0.1 | NTXent temperature |
| `model` | `hidden_dim` | 512 | Hidden layer width |
| `model` | `depth` | 6 | Number of residual blocks |

---

## Repository Structure

```
cross-dataset-audio-classifier/
├── main.py                  # entry point: train + evaluate
├── configs/
│   └── config.yaml          # all hyperparameters and data paths
├── src/
│   ├── data_loader.py       # embedding and label loading for BSD10K and FSD50K
│   ├── model.py             # ResidualBlock + DualHeadClassifier
│   ├── losses.py            # task losses + NTXent alignment
│   ├── train.py             # training loop
│   └── evaluate.py          # evaluation + CSV report
├── data/
│   ├── bsd10k_metadata_dev.csv
│   ├── bsd10k_metadata_eval.csv
│   ├── fsd50k_metadata_dev.csv
│   ├── fsd50k_metadata_eval.csv
│   ├── embeddings_bsd10k/   # pre-computed .npy embeddings (not tracked by git)
│   └── embeddings_fsd50k/   # pre-computed .npy embeddings (not tracked by git)
├── outputs/                 # evaluation CSVs written here
├── requirements.txt
└── .gitignore
```

