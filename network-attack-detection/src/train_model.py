"""
train_model.py
--------------
Master training pipeline for "Detection and Classification of Multi-Type
Network Attacks Using Time-Based Features".

Implements three classification approaches on the CIC-IDS-2018 (or
CIC-IDS-2018-shaped) dataset:

  Approach 1 - Basic Random Forest      (100 trees, default depth)
  Approach 2 - Tuned Random Forest      (200 trees, max_depth=20, regularized)
  Approach 3 - Stacking Ensemble + PCA  (RF + KNN + Calibrated LinearSVC -> LogReg)

Usage:
    python train_model.py --data ../data/sample_cicids2018.csv --outdir ../model_artifacts
"""

import argparse
import json
import os
import pickle
import time

# Prevent BLAS/OpenMP thread pools from competing with joblib's own thread
# management -- must be set BEFORE importing numpy/sklearn.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC


# --------------------------------------------------------------------------- #
# Data loading & preprocessing (FR1-FR4)
# --------------------------------------------------------------------------- #

def load_and_clean(path: str) -> pd.DataFrame:
    print(f"[1/7] Loading data from {path} ...")
    df = pd.read_csv(path, low_memory=False)
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore")

    print("[2/7] Cleaning (inf/NaN removal, label whitespace strip) ...")
    df = df.replace([np.inf, -np.inf], np.nan)
    before = len(df)
    df = df.dropna()
    after = len(df)
    print(f"       Dropped {before - after} rows containing NaN/inf ({(before - after) / max(before,1):.1%})")

    df["Label"] = df["Label"].astype(str).str.strip()
    print(f"       Final shape: {df.shape}")
    print(df["Label"].value_counts().to_string())
    return df


def encode_and_scale(df: pd.DataFrame):
    print("[3/7] Encoding labels and scaling features ...")
    feature_columns = [c for c in df.columns if c != "Label"]

    encoder = LabelEncoder()
    y = encoder.fit_transform(df["Label"].values)

    scaler = StandardScaler()
    X = scaler.fit_transform(df[feature_columns].values)

    return X, y, encoder, scaler, feature_columns


def split_data(X, y):
    print("[4/7] Stratified 80/20 train/test split ...")
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# --------------------------------------------------------------------------- #
# Shared metrics helper
# --------------------------------------------------------------------------- #

def metrics_dict(y_true, y_pred, classes, approach_name, description, training_time):
    acc = accuracy_score(y_true, y_pred)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    _, _, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    per_class_p, per_class_r, per_class_f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0, labels=range(len(classes))
    )
    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))

    per_class = {}
    for i, cls in enumerate(classes):
        per_class[cls] = {
            "precision": round(float(per_class_p[i]) * 100, 2),
            "recall": round(float(per_class_r[i]) * 100, 2),
            "f1": round(float(per_class_f1[i]) * 100, 2),
            "support": int(support[i]),
        }

    return {
        "approach_name": approach_name,
        "description": description,
        "accuracy": round(float(acc) * 100, 2),
        "macro_precision": round(float(macro_p) * 100, 2),
        "macro_recall": round(float(macro_r) * 100, 2),
        "macro_f1": round(float(macro_f1) * 100, 2),
        "weighted_f1": round(float(weighted_f1) * 100, 2),
        "training_time_seconds": round(training_time, 3),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "classes": list(classes),
    }


# --------------------------------------------------------------------------- #
# Approach 1 - Basic Random Forest
# --------------------------------------------------------------------------- #

def train_approach1(X_train, y_train, X_test, y_test, classes):
    print("\n[5/7] Training Approach 1: Basic Random Forest ...")
    model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    t0 = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - t0
    y_pred = model.predict(X_test)
    metrics = metrics_dict(
        y_test, y_pred, classes,
        "Basic Random Forest",
        "100-estimator RF, default hyperparameters, full 78-feature standardized set.",
        elapsed,
    )
    print(f"       Done in {elapsed:.2f}s | Accuracy: {metrics['accuracy']}% | Macro F1: {metrics['macro_f1']}%")
    return model, metrics


# --------------------------------------------------------------------------- #
# Approach 2 - Tuned Random Forest
# --------------------------------------------------------------------------- #

def train_approach2(X_train, y_train, X_test, y_test, classes):
    print("\n[5/7] Training Approach 2: Tuned Random Forest ...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42,
    )
    t0 = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - t0
    y_pred = model.predict(X_test)
    metrics = metrics_dict(
        y_test, y_pred, classes,
        "Tuned Random Forest",
        "200-estimator RF, max_depth=20, min_samples_split=5, min_samples_leaf=2.",
        elapsed,
    )
    print(f"       Done in {elapsed:.2f}s | Accuracy: {metrics['accuracy']}% | Macro F1: {metrics['macro_f1']}%")
    return model, metrics


# --------------------------------------------------------------------------- #
# Approach 3 - Stacking Ensemble + PCA
# --------------------------------------------------------------------------- #

def train_approach3(X_train, y_train, X_test, y_test, classes):
    print("\n[5/7] Training Approach 3: Stacking Ensemble + PCA ...")

    pca = PCA(n_components=10, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    base_estimators = [
        ("rf", RandomForestClassifier(n_estimators=100, n_jobs=1, random_state=42)),
        ("knn", KNeighborsClassifier(n_neighbors=3)),
        ("svm", CalibratedClassifierCV(LinearSVC(random_state=42, max_iter=5000), cv=2)),
    ]
    meta_learner = LogisticRegression(max_iter=1000)

    model = StackingClassifier(
        estimators=base_estimators,
        final_estimator=meta_learner,
        cv=2,
        n_jobs=-1,
    )

    t0 = time.time()
    model.fit(X_train_pca, y_train)
    elapsed = time.time() - t0
    y_pred = model.predict(X_test_pca)
    metrics = metrics_dict(
        y_test, y_pred, classes,
        "Stacking Ensemble + PCA",
        "RF + KNN(k=3) + Calibrated LinearSVC -> Logistic Regression meta-learner, 10-component PCA.",
        elapsed,
    )
    print(f"       Done in {elapsed:.2f}s | Accuracy: {metrics['accuracy']}% | Macro F1: {metrics['macro_f1']}%")
    return model, pca, metrics


# --------------------------------------------------------------------------- #
# Main orchestration
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Train all three network attack classification approaches.")
    parser.add_argument("--data", type=str, required=True, help="Path to labeled CIC-IDS-2018-shaped CSV.")
    parser.add_argument("--outdir", type=str, default="../model_artifacts", help="Directory to save model artifacts.")
    parser.add_argument("--skip-stacking", action="store_true", help="Skip Approach 3 (useful for quick smoke tests).")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = load_and_clean(args.data)
    X, y, encoder, scaler, feature_columns = encode_and_scale(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    classes = list(encoder.classes_)

    all_metrics = {}

    model1, metrics1 = train_approach1(X_train, y_train, X_test, y_test, classes)
    with open(os.path.join(args.outdir, "model1.pkl"), "wb") as f:
        pickle.dump(model1, f)
    all_metrics["approach1"] = metrics1
    del model1

    model2, metrics2 = train_approach2(X_train, y_train, X_test, y_test, classes)
    with open(os.path.join(args.outdir, "model2.pkl"), "wb") as f:
        pickle.dump(model2, f)
    all_metrics["approach2"] = metrics2
    del model2

    if not args.skip_stacking:
        model3, pca, metrics3 = train_approach3(X_train, y_train, X_test, y_test, classes)
        with open(os.path.join(args.outdir, "model3.pkl"), "wb") as f:
            pickle.dump(model3, f)
        with open(os.path.join(args.outdir, "pca.pkl"), "wb") as f:
            pickle.dump(pca, f)
        all_metrics["approach3"] = metrics3
        del model3

    print("\n[6/7] Saving shared preprocessing artifacts ...")
    with open(os.path.join(args.outdir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(args.outdir, "encoder.pkl"), "wb") as f:
        pickle.dump(encoder, f)
    with open(os.path.join(args.outdir, "feature_columns.pkl"), "wb") as f:
        pickle.dump(feature_columns, f)

    print("[7/7] Writing metrics.json ...")
    with open(os.path.join(args.outdir, "metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)

    print("\nTraining complete. Artifacts written to:", os.path.abspath(args.outdir))
    for name, m in all_metrics.items():
        print(f"  {name}: accuracy={m['accuracy']}%  macro_f1={m['macro_f1']}%  "
              f"time={m['training_time_seconds']}s")


if __name__ == "__main__":
    main()
