"""
live_capture.py
----------------
Real-time packet capture, flow reconstruction, and classification module.

Groups captured packets into flows by the 5-tuple (src IP, src port, dst IP,
dst port, protocol), accumulates per-flow statistics incrementally, and
classifies each completed flow using the selected trained model.

Emits one JSON object per line to stdout, which app.py reads and forwards to
connected browsers via Server-Sent Events. This keeps live capture in its own
process so a capture crash (e.g. missing root privileges) never takes down
the Flask server.

Requires elevated privileges for raw packet capture:
    sudo python live_capture.py --interface eth0 --approach 1

If Scapy / raw sockets are unavailable (no root, sandboxed environment,
etc.) the module automatically falls back to a --simulate mode that
generates synthetic flows so the dashboard's live panel can still be
demonstrated end-to-end.

Usage:
    python live_capture.py --interface any --approach 1
    python live_capture.py --simulate --approach 1
"""

import argparse
import json
import os
import pickle
import sys
import time
from collections import defaultdict

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "..", "model_artifacts")
IDLE_TIMEOUT_SECONDS = 5.0


def emit(event: dict):
    print(json.dumps(event), flush=True)


def _load_pickle(name):
    path = os.path.join(ARTIFACT_DIR, name)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


class FlowStats:
    """Accumulates per-flow packet statistics for the 5-tuple key."""

    def __init__(self, key, first_ts, proto, dst_port):
        self.key = key
        self.start_ts = first_ts
        self.last_ts = first_ts
        self.proto = proto
        self.dst_port = dst_port
        self.fwd_pkts = 0
        self.bwd_pkts = 0
        self.fwd_bytes = 0
        self.bwd_bytes = 0
        self.fwd_iats = []
        self.bwd_iats = []
        self.last_fwd_ts = None
        self.last_bwd_ts = None
        self.psh_count = 0
        self.syn_count = 0
        self.fin_count = 0
        self.rst_count = 0
        self.init_win_fwd = -1
        self.init_win_bwd = -1

    def add_packet(self, ts, direction, length, flags, win_size):
        self.last_ts = ts
        if direction == "fwd":
            if self.last_fwd_ts is not None:
                self.fwd_iats.append(ts - self.last_fwd_ts)
            self.last_fwd_ts = ts
            self.fwd_pkts += 1
            self.fwd_bytes += length
            if self.init_win_fwd == -1:
                self.init_win_fwd = win_size
        else:
            if self.last_bwd_ts is not None:
                self.bwd_iats.append(ts - self.last_bwd_ts)
            self.last_bwd_ts = ts
            self.bwd_pkts += 1
            self.bwd_bytes += length
            if self.init_win_bwd == -1:
                self.init_win_bwd = win_size

        if "P" in flags:
            self.psh_count += 1
        if "S" in flags:
            self.syn_count += 1
        if "F" in flags:
            self.fin_count += 1
        if "R" in flags:
            self.rst_count += 1

    def is_complete(self, now):
        return self.fin_count > 0 or self.rst_count > 0 or (now - self.last_ts) > IDLE_TIMEOUT_SECONDS

    def to_feature_row(self, feature_columns):
        duration = max((self.last_ts - self.start_ts) * 1e6, 1.0)  # microseconds
        total_pkts = self.fwd_pkts + self.bwd_pkts
        total_bytes = self.fwd_bytes + self.bwd_bytes
        all_iats = self.fwd_iats + self.bwd_iats

        def stat(vals, fn, default=0.0):
            return float(fn(vals)) if vals else default

        row = {c: 0.0 for c in feature_columns}
        row["Dst Port"] = self.dst_port
        row["Protocol"] = self.proto
        row["Flow Duration"] = duration
        row["Tot Fwd Pkts"] = self.fwd_pkts
        row["Tot Bwd Pkts"] = self.bwd_pkts
        row["TotLen Fwd Pkts"] = self.fwd_bytes
        row["TotLen Bwd Pkts"] = self.bwd_bytes
        row["Flow Byts/s"] = total_bytes / (duration / 1e6)
        row["Flow Pkts/s"] = total_pkts / (duration / 1e6)
        row["Flow IAT Mean"] = stat(all_iats, np.mean) * 1e6
        row["Flow IAT Std"] = stat(all_iats, np.std) * 1e6
        row["Flow IAT Max"] = stat(all_iats, max) * 1e6
        row["Flow IAT Min"] = stat(all_iats, min) * 1e6
        row["Fwd IAT Mean"] = stat(self.fwd_iats, np.mean) * 1e6
        row["Fwd IAT Std"] = stat(self.fwd_iats, np.std) * 1e6
        row["Bwd IAT Mean"] = stat(self.bwd_iats, np.mean) * 1e6
        row["Bwd IAT Std"] = stat(self.bwd_iats, np.std) * 1e6
        row["Fwd Pkts/s"] = self.fwd_pkts / (duration / 1e6)
        row["Bwd Pkts/s"] = self.bwd_pkts / (duration / 1e6)
        row["PSH Flag Cnt"] = self.psh_count
        row["FIN Flag Cnt"] = self.fin_count
        row["SYN Flag Cnt"] = self.syn_count
        row["RST Flag Cnt"] = self.rst_count
        row["Init Fwd Win Byts"] = self.init_win_fwd
        row["Init Bwd Win Byts"] = self.init_win_bwd
        row["Subflow Fwd Pkts"] = self.fwd_pkts
        row["Subflow Fwd Byts"] = self.fwd_bytes
        row["Subflow Bwd Pkts"] = self.bwd_pkts
        row["Subflow Bwd Byts"] = self.bwd_bytes
        return row


class Classifier:
    def __init__(self, approach: str):
        self.approach = approach
        self.model = _load_pickle(f"model{approach}.pkl")
        self.scaler = _load_pickle("scaler.pkl")
        self.encoder = _load_pickle("encoder.pkl")
        self.feature_columns = _load_pickle("feature_columns.pkl")
        self.pca = _load_pickle("pca.pkl") if approach == "3" else None
        self.ready = all([self.model, self.scaler, self.encoder, self.feature_columns])

    def classify(self, feature_row: dict):
        if not self.ready:
            return None, None
        vec = np.array([[feature_row.get(c, 0.0) for c in self.feature_columns]])
        vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
        X = self.scaler.transform(vec)
        if self.pca is not None:
            X = self.pca.transform(X)
        pred = self.model.predict(X)[0]
        label = self.encoder.inverse_transform([pred])[0]
        confidence = None
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X)[0]
            confidence = float(np.max(proba))
        return label, confidence


def run_simulation(classifier: Classifier, feature_columns):
    """Fallback mode: synthesize plausible flows so the dashboard can be
    demoed without root / raw-socket access."""
    from generate_sample_data import _make_rows  # local import, reuses profiles

    emit({"type": "status", "message": "running in --simulate mode (no raw socket access needed)"})
    labels_cycle = ["BENIGN", "BENIGN", "DDoS", "PortScan", "BENIGN",
                    "SSH-Patator", "BENIGN", "Web Attack - Brute Force", "BENIGN"]
    i = 0
    while True:
        label = labels_cycle[i % len(labels_cycle)]
        row_df = _make_rows(label, 1, None)
        row = row_df.drop(columns=["Label"]).iloc[0].to_dict()
        pred_label, confidence = classifier.classify(row)
        emit({
            "type": "flow",
            "timestamp": time.time(),
            "dst_port": int(row.get("Dst Port", -1)),
            "protocol": int(row.get("Protocol", 6)),
            "predicted": pred_label,
            "confidence": confidence,
            "ground_truth_hint": label,  # only present in simulate mode
        })
        i += 1
        time.sleep(1.2)


def run_capture(interface: str, classifier: Classifier, feature_columns):
    try:
        from scapy.all import sniff, IP, TCP, UDP
    except Exception as e:
        emit({"type": "status", "message": f"scapy unavailable ({e}); falling back to --simulate mode"})
        run_simulation(classifier, feature_columns)
        return

    flows = {}

    def flush_complete(now):
        done_keys = [k for k, f in flows.items() if f.is_complete(now)]
        for k in done_keys:
            f = flows.pop(k)
            row = f.to_feature_row(feature_columns)
            label, confidence = classifier.classify(row)
            emit({
                "type": "flow",
                "timestamp": now,
                "dst_port": f.dst_port,
                "protocol": f.proto,
                "predicted": label,
                "confidence": confidence,
            })

    def handle_packet(pkt):
        if IP not in pkt:
            return
        now = time.time()
        proto = 6 if TCP in pkt else (17 if UDP in pkt else pkt[IP].proto)
        sport = pkt.sport if (TCP in pkt or UDP in pkt) else 0
        dport = pkt.dport if (TCP in pkt or UDP in pkt) else 0
        length = len(pkt)
        flags = str(pkt[TCP].flags) if TCP in pkt else ""
        win = int(pkt[TCP].window) if TCP in pkt else -1

        fwd_key = (pkt[IP].src, sport, pkt[IP].dst, dport, proto)
        bwd_key = (pkt[IP].dst, dport, pkt[IP].src, sport, proto)

        if fwd_key in flows:
            flows[fwd_key].add_packet(now, "fwd", length, flags, win)
        elif bwd_key in flows:
            flows[bwd_key].add_packet(now, "bwd", length, flags, win)
        else:
            f = FlowStats(fwd_key, now, proto, dport)
            f.add_packet(now, "fwd", length, flags, win)
            flows[fwd_key] = f

        if len(flows) % 20 == 0:
            flush_complete(now)

    emit({"type": "status", "message": f"capturing on interface={interface} (requires root/CAP_NET_RAW)"})
    try:
        sniff(iface=None if interface in ("any", "") else interface,
              prn=handle_packet, store=False)
    except PermissionError:
        emit({"type": "status", "message": "permission denied for raw capture; falling back to --simulate mode"})
        run_simulation(classifier, feature_columns)


def main():
    parser = argparse.ArgumentParser(description="Live network flow capture and classification.")
    parser.add_argument("--interface", type=str, default="any")
    parser.add_argument("--approach", type=str, default="1", choices=["1", "2", "3"])
    parser.add_argument("--simulate", action="store_true", help="Force synthetic flow simulation mode.")
    args = parser.parse_args()

    classifier = Classifier(args.approach)
    feature_columns = classifier.feature_columns or []

    if not classifier.ready:
        emit({"type": "status", "message": "Model artifacts missing - run train_model.py first."})
        return

    if args.simulate:
        run_simulation(classifier, feature_columns)
    else:
        run_capture(args.interface, classifier, feature_columns)


if __name__ == "__main__":
    main()
