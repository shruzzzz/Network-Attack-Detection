"""
generate_sample_data.py
------------------------
Generates a synthetic CIC-IDS-2018-shaped dataset so the full pipeline
(train -> evaluate -> serve -> dashboard) can be run end-to-end without
first downloading the real ~6GB CIC-IDS-2018 dataset.

The 78 columns and 15 class names match the real dataset's schema. Class
proportions and per-class feature distributions are designed to loosely
mimic the behavioral patterns described in the dissertation (e.g. DDoS =
very high Flow Bytes/s, slow-HTTP = long flow duration, web attacks =
tiny packets on port 80), so the trained models show broadly similar
qualitative behavior (strong on volumetric attacks, weaker on XSS/SQL
Injection/Infiltration) without claiming to reproduce the paper's exact
numbers.

Usage:
    python generate_sample_data.py --rows 60000 --out ../data/sample_cicids2018.csv
"""

import argparse
import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "Dst Port", "Protocol", "Flow Duration", "Tot Fwd Pkts", "Tot Bwd Pkts",
    "TotLen Fwd Pkts", "TotLen Bwd Pkts", "Fwd Pkt Len Max", "Fwd Pkt Len Min",
    "Fwd Pkt Len Mean", "Fwd Pkt Len Std", "Bwd Pkt Len Max", "Bwd Pkt Len Min",
    "Bwd Pkt Len Mean", "Bwd Pkt Len Std", "Flow Byts/s", "Flow Pkts/s",
    "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "Fwd IAT Tot", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min",
    "Bwd IAT Tot", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
    "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags",
    "Fwd Header Len", "Bwd Header Len", "Fwd Pkts/s", "Bwd Pkts/s",
    "Pkt Len Min", "Pkt Len Max", "Pkt Len Mean", "Pkt Len Std", "Pkt Len Var",
    "FIN Flag Cnt", "SYN Flag Cnt", "RST Flag Cnt", "PSH Flag Cnt",
    "ACK Flag Cnt", "URG Flag Cnt", "CWE Flag Count", "ECE Flag Cnt",
    "Down/Up Ratio", "Pkt Size Avg", "Fwd Seg Size Avg", "Bwd Seg Size Avg",
    "Fwd Byts/b Avg", "Fwd Pkts/b Avg", "Fwd Blk Rate Avg", "Bwd Byts/b Avg",
    "Bwd Pkts/b Avg", "Bwd Blk Rate Avg", "Subflow Fwd Pkts", "Subflow Fwd Byts",
    "Subflow Bwd Pkts", "Subflow Bwd Byts", "Init Fwd Win Byts",
    "Init Bwd Win Byts", "Fwd Act Data Pkts", "Fwd Seg Size Min",
    "Active Mean", "Active Std", "Active Max", "Active Min",
    "Idle Mean", "Idle Std", "Idle Max", "Idle Min",
]
assert len(FEATURE_COLUMNS) == 78, len(FEATURE_COLUMNS)

# (class name, approx proportion, dst_port, flow_bytes_s range, flow_duration range, iat_mean range, psh_ratio)
CLASS_PROFILES = [
    ("BENIGN",              0.34, None,        (2e4, 3e6),   (1e3, 5e6),   (1e3, 5e4),  0.30),
    ("DDoS",                0.07, 80,          (5e6, 5e7),   (10, 5e4),    (1, 50),     0.55),
    ("DoS GoldenEye",       0.05, 80,          (2e5, 3e6),   (1e3, 2e5),   (5, 200),    0.65),
    ("DoS Hulk",            0.05, 80,          (3e6, 4e7),   (10, 3e4),    (1, 30),     0.60),
    ("DoS Slowhttptest",    0.05, 80,          (50, 500),    (5e6, 1.2e8), (1e4, 3e5),  0.20),
    ("DoS slowloris",       0.05, 80,          (30, 400),    (5e6, 1.2e8), (1e4, 4e5),  0.20),
    ("Bot",                 0.05, None,        (500, 5e4),   (1e5, 1e7),   (3.9e5, 4.1e5), 0.35),
    ("FTP-Patator",         0.05, 21,          (1e3, 2e4),   (1e3, 1e6),   (50, 3e3),   0.10),
    ("SSH-Patator",         0.05, 22,          (1e3, 2e4),   (1e3, 1e6),   (50, 3e3),   0.10),
    ("Heartbleed",          0.005, 443,        (2e3, 1e4),   (1e5, 5e6),   (1e3, 1e4),  0.15),
    ("Infiltration",        0.005, None,       (1e4, 5e5),   (1e4, 1e7),   (1e2, 1e4),  0.30),
    ("PortScan",            0.05, None,        (100, 3e3),   (1, 5e4),     (1, 200),    0.05),
    ("Web Attack - Brute Force", 0.05, 80,     (100, 1e3),   (1e3, 2e5),   (1e2, 5e3),  0.90),
    ("Web Attack - SQL Injection", 0.003, 80,  (80, 400),    (1e3, 2e5),   (1e2, 5e3),  0.92),
    ("Web Attack - XSS",    0.017, 80,         (90, 450),    (1e3, 2e5),   (1e2, 5e3),  0.91),
]

RNG = np.random.default_rng(42)


def _sample_range(low, high, n):
    return RNG.uniform(low, high, size=n)


def _make_rows(label, n, port_hint):
    profile = next(p for p in CLASS_PROFILES if p[0] == label)
    _, _, dst_port_fixed, byts_range, dur_range, iat_range, psh_ratio = profile

    data = {}
    if dst_port_fixed is not None:
        data["Dst Port"] = np.full(n, dst_port_fixed)
    else:
        data["Dst Port"] = RNG.integers(1024, 65535, size=n)

    data["Protocol"] = RNG.choice([6, 17], size=n, p=[0.85, 0.15])
    data["Flow Duration"] = _sample_range(*dur_range, n)
    data["Flow Byts/s"] = _sample_range(*byts_range, n)
    data["Flow IAT Mean"] = _sample_range(*iat_range, n)
    data["Flow IAT Std"] = data["Flow IAT Mean"] * RNG.uniform(0.05, 0.5, n)
    data["Flow IAT Max"] = data["Flow IAT Mean"] * RNG.uniform(1.5, 4, n)
    data["Flow IAT Min"] = data["Flow IAT Mean"] * RNG.uniform(0.01, 0.5, n)
    data["Flow Pkts/s"] = data["Flow Byts/s"] / RNG.uniform(50, 500, n)

    data["Tot Fwd Pkts"] = RNG.integers(1, 500, n).astype(float)
    data["Tot Bwd Pkts"] = RNG.integers(0, 500, n).astype(float)
    data["TotLen Fwd Pkts"] = data["Tot Fwd Pkts"] * RNG.uniform(20, 1500, n)
    data["TotLen Bwd Pkts"] = data["Tot Bwd Pkts"] * RNG.uniform(20, 1500, n)

    for c in ["Fwd Pkt Len Max", "Fwd Pkt Len Min", "Fwd Pkt Len Mean", "Fwd Pkt Len Std",
              "Bwd Pkt Len Max", "Bwd Pkt Len Min", "Bwd Pkt Len Mean", "Bwd Pkt Len Std",
              "Pkt Len Min", "Pkt Len Max", "Pkt Len Mean", "Pkt Len Std", "Pkt Len Var",
              "Pkt Size Avg", "Fwd Seg Size Avg", "Bwd Seg Size Avg"]:
        data[c] = _sample_range(20, 1500, n)

    data["Fwd IAT Tot"] = data["Flow IAT Mean"] * data["Tot Fwd Pkts"]
    data["Fwd IAT Mean"] = data["Flow IAT Mean"] * RNG.uniform(0.8, 1.2, n)
    data["Fwd IAT Std"] = data["Flow IAT Std"] * RNG.uniform(0.8, 1.2, n)
    data["Fwd IAT Max"] = data["Flow IAT Max"] * RNG.uniform(0.8, 1.2, n)
    data["Fwd IAT Min"] = data["Flow IAT Min"] * RNG.uniform(0.8, 1.2, n)
    data["Bwd IAT Tot"] = data["Flow IAT Mean"] * data["Tot Bwd Pkts"]
    data["Bwd IAT Mean"] = data["Flow IAT Mean"] * RNG.uniform(0.8, 1.2, n)
    data["Bwd IAT Std"] = data["Flow IAT Std"] * RNG.uniform(0.8, 1.2, n)
    data["Bwd IAT Max"] = data["Flow IAT Max"] * RNG.uniform(0.8, 1.2, n)
    data["Bwd IAT Min"] = data["Flow IAT Min"] * RNG.uniform(0.8, 1.2, n)

    data["Fwd PSH Flags"] = (RNG.uniform(0, 1, n) < psh_ratio).astype(float)
    data["Bwd PSH Flags"] = np.zeros(n)
    data["Fwd URG Flags"] = np.zeros(n)
    data["Bwd URG Flags"] = np.zeros(n)
    data["Fwd Header Len"] = data["Tot Fwd Pkts"] * 20
    data["Bwd Header Len"] = data["Tot Bwd Pkts"] * 20
    data["Fwd Pkts/s"] = data["Tot Fwd Pkts"] / np.maximum(data["Flow Duration"] / 1e6, 1e-6)
    data["Bwd Pkts/s"] = data["Tot Bwd Pkts"] / np.maximum(data["Flow Duration"] / 1e6, 1e-6)

    data["FIN Flag Cnt"] = RNG.integers(0, 2, n).astype(float)
    data["SYN Flag Cnt"] = RNG.integers(0, 2, n).astype(float)
    data["RST Flag Cnt"] = np.zeros(n)
    data["PSH Flag Cnt"] = (RNG.uniform(0, 1, n) < psh_ratio).astype(float)
    data["ACK Flag Cnt"] = RNG.integers(0, 5, n).astype(float)
    data["URG Flag Cnt"] = np.zeros(n)
    data["CWE Flag Count"] = np.zeros(n)
    data["ECE Flag Cnt"] = np.zeros(n)
    data["Down/Up Ratio"] = RNG.uniform(0.2, 3, n)

    data["Fwd Byts/b Avg"] = np.zeros(n)
    data["Fwd Pkts/b Avg"] = np.zeros(n)
    data["Fwd Blk Rate Avg"] = np.zeros(n)
    data["Bwd Byts/b Avg"] = np.zeros(n)
    data["Bwd Pkts/b Avg"] = np.zeros(n)
    data["Bwd Blk Rate Avg"] = np.zeros(n)

    data["Subflow Fwd Pkts"] = data["Tot Fwd Pkts"]
    data["Subflow Fwd Byts"] = data["TotLen Fwd Pkts"]
    data["Subflow Bwd Pkts"] = data["Tot Bwd Pkts"]
    data["Subflow Bwd Byts"] = data["TotLen Bwd Pkts"]

    if label == "Web Attack - SQL Injection" or label == "Web Attack - XSS" or label == "Web Attack - Brute Force":
        data["Init Fwd Win Byts"] = np.full(n, 29200.0)
        data["Init Bwd Win Byts"] = np.full(n, 28960.0)
    else:
        data["Init Fwd Win Byts"] = RNG.choice([-1, 8192, 29200, 65535], size=n).astype(float)
        data["Init Bwd Win Byts"] = RNG.choice([-1, 8192, 29200, 65535], size=n).astype(float)

    data["Fwd Act Data Pkts"] = RNG.integers(0, 100, n).astype(float)
    data["Fwd Seg Size Min"] = RNG.choice([20, 32], size=n).astype(float)

    for c in ["Active Mean", "Active Std", "Active Max", "Active Min",
              "Idle Mean", "Idle Std", "Idle Max", "Idle Min"]:
        data[c] = _sample_range(0, 1e5, n)

    df = pd.DataFrame(data)
    df = df[FEATURE_COLUMNS]
    df["Label"] = label
    return df


def generate(n_rows: int) -> pd.DataFrame:
    frames = []
    for label, prop, *_ in CLASS_PROFILES:
        n = max(2, int(n_rows * prop))
        frames.append(_make_rows(label, n, None))
    df = pd.concat(frames, ignore_index=True)
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    # Sprinkle a small percentage of inf/NaN to mimic zero-duration-flow artifacts
    n_dirty = int(0.02 * len(df))
    dirty_idx = RNG.choice(df.index, size=n_dirty, replace=False)
    df.loc[dirty_idx, "Flow Byts/s"] = np.inf
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a synthetic CIC-IDS-2018-shaped dataset.")
    parser.add_argument("--rows", type=int, default=60000, help="Approximate total number of rows to generate.")
    parser.add_argument("--out", type=str, default="../data/sample_cicids2018.csv", help="Output CSV path.")
    args = parser.parse_args()

    df = generate(args.rows)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} rows x {len(df.columns)} columns to {args.out}")
    print(df["Label"].value_counts())
