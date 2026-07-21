"""
app.py
------
Flask REST API + browser dashboard for the Network Attack Detection system.

Routes:
    GET  /                    -> dashboard (templates/index.html)
    GET  /api/model-info      -> summary of all trained models + metrics
    POST /api/predict         -> classify uploaded CSV of flow feature vectors
    POST /api/live/start      -> start live packet capture (subprocess)
    POST /api/live/stop       -> stop live packet capture
    GET  /sse                 -> Server-Sent Events stream for live classification

Run:
    python app.py
    (then open http://localhost:5000)
"""

import io
import json
import os
import pickle
import queue
import subprocess
import sys
import threading
import time

import numpy as np
import pandas as pd
from flask import Flask, Response, jsonify, render_template, request

ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "..", "model_artifacts")

app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(__file__), "..", "templates"),
    static_folder=os.path.join(os.path.dirname(__file__), "..", "static"),
)

# --------------------------------------------------------------------------- #
# Load all artifacts once at startup
# --------------------------------------------------------------------------- #

_artifacts = {"models": {}, "scaler": None, "encoder": None, "feature_columns": None,
              "pca": None, "metrics": {}}


def _load_pickle(name):
    path = os.path.join(ARTIFACT_DIR, name)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def load_artifacts():
    print("Loading model artifacts from", os.path.abspath(ARTIFACT_DIR))
    _artifacts["models"]["1"] = _load_pickle("model1.pkl")
    _artifacts["models"]["2"] = _load_pickle("model2.pkl")
    _artifacts["models"]["3"] = _load_pickle("model3.pkl")
    _artifacts["scaler"] = _load_pickle("scaler.pkl")
    _artifacts["encoder"] = _load_pickle("encoder.pkl")
    _artifacts["feature_columns"] = _load_pickle("feature_columns.pkl")
    _artifacts["pca"] = _load_pickle("pca.pkl")

    metrics_path = os.path.join(ARTIFACT_DIR, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            _artifacts["metrics"] = json.load(f)

    loaded = [k for k, v in _artifacts["models"].items() if v is not None]
    print(f"Loaded models: {loaded or 'NONE (run train_model.py first)'}")


load_artifacts()

# --------------------------------------------------------------------------- #
# SSE plumbing for live capture
# --------------------------------------------------------------------------- #

_sse_clients = []
_live_process = None
_live_lock = threading.Lock()


def _broadcast(event: dict):
    data = f"data: {json.dumps(event)}\n\n"
    for q in list(_sse_clients):
        try:
            q.put_nowait(data)
        except queue.Full:
            pass


@app.route("/sse")
def sse():
    def stream():
        q = queue.Queue(maxsize=100)
        _sse_clients.append(q)
        try:
            yield "data: " + json.dumps({"type": "status", "message": "connected"}) + "\n\n"
            while True:
                data = q.get()
                yield data
        finally:
            _sse_clients.remove(q)

    return Response(stream(), mimetype="text/event-stream")


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/model-info")
def model_info():
    approaches = []
    labels = {"1": "approach1", "2": "approach2", "3": "approach3"}
    badges = {"1": "Fast", "2": "Balanced", "3": "Best Ensemble"}
    for key, metrics_key in labels.items():
        m = _artifacts["metrics"].get(metrics_key)
        available = _artifacts["models"].get(key) is not None
        approaches.append({
            "id": key,
            "available": available,
            "badge": badges[key],
            "metrics": m,
        })

    classes = list(_artifacts["encoder"].classes_) if _artifacts["encoder"] is not None else []

    return jsonify({
        "approaches": approaches,
        "classes": classes,
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    approach = request.args.get("approach", request.form.get("approach", "1"))

    model = _artifacts["models"].get(approach)
    scaler = _artifacts["scaler"]
    encoder = _artifacts["encoder"]
    feature_columns = _artifacts["feature_columns"]
    pca = _artifacts["pca"]

    if model is None or scaler is None or encoder is None:
        return jsonify({"error": "Model artifacts not found. Run train_model.py first."}), 400

    # Accept either a file upload or a JSON body of feature vectors
    if "file" in request.files:
        file = request.files["file"]
        df = pd.read_csv(file)
    else:
        body = request.get_json(silent=True)
        if not body or "rows" not in body:
            return jsonify({"error": "Provide a CSV file upload or JSON {'rows': [...]}"}), 400
        df = pd.DataFrame(body["rows"])

    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore")
    has_labels = "Label" in df.columns
    true_labels = df["Label"].astype(str).str.strip() if has_labels else None
    df = df.drop(columns=["Label"], errors="ignore")

    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        return jsonify({"error": f"Missing required feature columns: {missing[:5]}..."}), 400

    df = df[feature_columns]
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    X = scaler.transform(df.values)
    if approach == "3" and pca is not None:
        X = pca.transform(X)

    y_pred = model.predict(X)
    probs = model.predict_proba(X) if hasattr(model, "predict_proba") else None
    pred_labels = encoder.inverse_transform(y_pred)

    class_counts = pd.Series(pred_labels).value_counts().to_dict()
    total = len(pred_labels)
    benign_count = class_counts.get("BENIGN", 0)
    attack_count = total - benign_count

    log_rows = []
    for i in range(min(total, 2000)):  # cap the returned log for payload size
        row = {"row": i, "predicted": pred_labels[i]}
        if probs is not None:
            top3_idx = np.argsort(probs[i])[::-1][:3]
            row["top3"] = [
                {"class": encoder.classes_[j], "probability": round(float(probs[i][j]), 4)}
                for j in top3_idx
            ]
        if has_labels:
            row["actual"] = true_labels.iloc[i]
        log_rows.append(row)

    response = {
        "approach": approach,
        "summary": {
            "total": total,
            "benign": int(benign_count),
            "attack": int(attack_count),
            "benign_pct": round(benign_count / total * 100, 2) if total else 0,
            "attack_pct": round(attack_count / total * 100, 2) if total else 0,
        },
        "class_distribution": class_counts,
        "log": log_rows,
    }
    return jsonify(response)


@app.route("/api/live/start", methods=["POST"])
def live_start():
    global _live_process
    with _live_lock:
        if _live_process is not None and _live_process.poll() is None:
            return jsonify({"status": "already_running"})

        interface = request.json.get("interface", "any") if request.is_json else "any"
        approach = request.json.get("approach", "1") if request.is_json else "1"

        script = os.path.join(os.path.dirname(__file__), "live_capture.py")
        try:
            _live_process = subprocess.Popen(
                [sys.executable, script, "--interface", interface, "--approach", approach],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except Exception as e:
            return jsonify({"error": str(e)}), 500

        def reader():
            for line in _live_process.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    event = {"type": "log", "message": line}
                _broadcast(event)

        threading.Thread(target=reader, daemon=True).start()

    _broadcast({"type": "status", "message": f"live capture starting on interface={interface}"})
    return jsonify({"status": "started"})


@app.route("/api/live/stop", methods=["POST"])
def live_stop():
    global _live_process
    with _live_lock:
        if _live_process is not None and _live_process.poll() is None:
            _live_process.terminate()
            try:
                _live_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                _live_process.kill()
        _live_process = None
    _broadcast({"type": "status", "message": "live capture stopped"})
    return jsonify({"status": "stopped"})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000, threaded=True)
