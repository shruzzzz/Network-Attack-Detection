# Flowline — Detection and Classification of Multi-Type Network Attacks Using Time-Based Features

An end-to-end network intrusion detection system built around three
complementary classifiers trained on **time-based flow features** (inter-arrival
time statistics, packet rate, flow duration, burst patterns). Implements the
methodology from the accompanying dissertation:

- **Approach 1 — Basic Random Forest**: 100-estimator RF, default hyperparameters, full 78-feature set.
- **Approach 2 — Tuned Random Forest**: 200-estimator RF with regularization (`max_depth=20`, `min_samples_split=5`, `min_samples_leaf=2`).
- **Approach 3 — Stacking Ensemble + PCA**: RF + KNN(k=3) + Calibrated LinearSVC → Logistic Regression meta-learner, over a 10-component PCA-reduced feature space.

The system ships with a training pipeline, a Flask REST API + real-time
browser dashboard, and a live packet capture module.

## Project layout

```
network-attack-detection/
├── src/
│   ├── generate_sample_data.py   # synthetic CIC-IDS-2018-shaped dataset generator
│   ├── train_model.py            # trains all 3 approaches, saves metrics + models
│   ├── app.py                    # Flask REST API + dashboard server
│   └── live_capture.py           # live packet capture / flow classification
├── templates/
│   └── index.html                # dashboard UI
├── static/                       # (reserved for extra CSS/JS/images)
├── data/                         # place your CIC-IDS-2018 CSV(s) here
├── model_artifacts/              # trained models + metrics.json land here
├── notebooks/                    # optional EDA/experiments
├── requirements.txt
└── README.md
```

## Getting the data

This project targets the **CIC-IDS-2018** dataset (78 flow-level features per
record, 15 traffic classes: BENIGN + 14 attack families). Two ways to get data:

1. **Use the real dataset.** Download the CSVs from the Canadian Institute for
   Cybersecurity (search "CSE-CIC-IDS2018"), merge them into a single CSV with
   a `Label` column, and drop it into `data/`.
2. **Use the bundled synthetic generator** to try the whole pipeline immediately
   without any download:

   ```bash
   cd src
   python generate_sample_data.py --rows 60000 --out ../data/sample_cicids2018.csv
   ```

   This creates a dataset with the same 78 columns and 15 class labels, with
   per-class feature distributions shaped to loosely mimic the behavioral
   patterns described in the dissertation (e.g. DDoS = very high `Flow Byts/s`,
   slow-HTTP = very long `Flow Duration`, web attacks = tiny packets on port 80).
   It's for exercising the pipeline end-to-end, not for reproducing the
   dissertation's exact reported numbers — for that you need the real dataset.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Train the models

```bash
cd src
python train_model.py --data ../data/sample_cicids2018.csv --outdir ../model_artifacts
```

This runs the full pipeline (load → clean → encode/scale → stratified 80/20
split → train all 3 approaches → evaluate → serialize) and writes to
`model_artifacts/`:

| File | Description |
|---|---|
| `model1.pkl` | Basic Random Forest |
| `model2.pkl` | Tuned Random Forest |
| `model3.pkl` | Stacking Ensemble |
| `pca.pkl` | PCA projection used by Approach 3 |
| `scaler.pkl` | StandardScaler fit on training features |
| `encoder.pkl` | LabelEncoder mapping class names ↔ integer indices |
| `feature_columns.pkl` | Ordered list of the 78 feature names |
| `metrics.json` | Accuracy, macro/weighted P/R/F1, per-class breakdown, confusion matrix, training time — for all 3 approaches |

Use `--skip-stacking` for a quick smoke test (Approach 3 takes noticeably
longer due to its cross-validated stacking).

## Run the dashboard + API

```bash
cd src
python app.py
```

Open **http://localhost:5000**. The dashboard shows:

- Approach selector cards with accuracy / macro F1 / training time
- Per-class precision/recall/F1 table with inline bars
- Aggregate metrics bar chart across all 3 approaches
- Batch classification: upload a CSV of flow features, get a benign/attack
  breakdown, class distribution doughnut chart, and per-row top-3 probabilities
- Live monitor panel showing real-time classified flows over Server-Sent Events

### REST API

- `GET /api/model-info` — all trained models' metadata + metrics
- `POST /api/predict?approach=1` — multipart file upload (`file` field, CSV) **or** JSON body `{"rows": [...]}`; returns predicted classes, summary stats, and per-row top-3 probabilities
- `POST /api/live/start` — body `{"interface": "any", "approach": "1"}`, starts the live capture subprocess
- `POST /api/live/stop` — stops it
- `GET /sse` — Server-Sent Events stream consumed by the dashboard

## Live packet capture

```bash
cd src
sudo python live_capture.py --interface eth0 --approach 1
```

Requires root / `CAP_NET_RAW` for raw socket access (standard for any
packet-capture tool). If that's unavailable — e.g. in a sandboxed dev
environment — it automatically falls back to `--simulate` mode, which
generates a realistic mix of synthetic flows so the live dashboard panel can
still be demonstrated end-to-end. You can also force this explicitly:

```bash
python live_capture.py --simulate --approach 1
```

## Notes on reproducing the dissertation's reported numbers

The dissertation reports results on the real CIC-IDS-2018 benchmark (Basic RF:
95.84% accuracy / 90.43% macro F1; full details in the report). To reproduce
those numbers, train on the actual CIC-IDS-2018 CSVs rather than the synthetic
generator here — the synthetic generator is a structural stand-in so the code
can be exercised without a multi-gigabyte download.

## Extending this project

Ideas from the dissertation's future-work chapter that map directly onto this
codebase:

- Add `class_weight="balanced"` or SMOTE/ADASYN in `train_model.py` to address
  class imbalance (Web Attack – XSS / SQL Injection / Infiltration).
- Add a SHAP/LIME explainability endpoint alongside `/api/predict`.
- Swap `live_capture.py`'s flow completion logic for an async/streaming
  design if you need to sustain high packet rates.
