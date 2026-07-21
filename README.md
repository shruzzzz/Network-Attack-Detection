 Detection and Classification of Multi-Type Network Attacks Using Time-Based Features

An end-to-end network intrusion detection system built around three
complementary classifiers trained on **time-based flow features** (inter-arrival
time statistics, packet rate, flow duration, burst patterns). Implements the
methodology from the accompanying dissertation:

1 — Basic Random Forest**: 100-estimator RF, default hyperparameters, full 78-feature set.
2 — Tuned Random Forest**: 200-estimator RF with regularization (`max_depth=20`, `min_samples_split=5`, `min_samples_leaf=2`).
3 — Stacking Ensemble + PCA**: RF + KNN(k=3) + Calibrated LinearSVC → Logistic Regression meta-learner, over a 10-component PCA-reduced feature space.


Project layout

network-attack-detection
── src
── generate_sample_data.py   
── train_model.py            
── app.py                   
── live_capture.py           
── templates
── index.html                
── static                     
── data                         
── model_artifacts             
── notebooks                 
── requirements.txt
── README.md

Data

This project targets the **CIC-IDS-2018** dataset (78 flow-level features per
record, 15 traffic classes: BENIGN + 14 attack families). Two ways to get data:

1. *Use the real dataset.** Download the CSVs from the Canadian Institute for
   Cybersecurity (search "CSE-CIC-IDS2018"), merge them into a single CSV with
   a `Label` column, and drop it into `data/`.
2. Use the bundled synthetic generator** to try the whole pipeline immediately
   without any download:

   This creates a dataset with the same 78 columns and 15 class labels, with
   per-class feature distributions shaped to loosely mimic the behavioral
   patterns described in the dissertation (e.g. DDoS = very high `Flow Byts/s`,
   slow-HTTP = very long `Flow Duration`, web attacks = tiny packets on port 80).
   It's for exercising the pipeline end-to-end, not for reproducing the
   dissertation's exact reported numbers — for that you need the real dataset.

Open http://localhost:5000. The dashboard shows:

- Approach selector cards with accuracy / macro F1 / training time
- Per-class precision/recall/F1 table with inline bars
- Aggregate metrics bar chart across all 3 approaches
- Batch classification: upload a CSV of flow features, get a benign/attack
  breakdown, class distribution doughnut chart, and per-row top-3 probabilities
- Live monitor panel showing real-time classified flows over Server-Sent Events


Notes on reproducing the dissertation's reported numbers

The dissertation reports results on the real CIC-IDS-2018 benchmark (Basic RF:
95.84% accuracy  90.43% macro F1; full details in the report). To reproduce
those numbers, train on the actual CIC-IDS-2018 CSVs rather than the synthetic
generator here — the synthetic generator is a structural stand-in so the code
can be exercised without a multi-gigabyte download.

