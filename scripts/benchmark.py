"""Mesures de performance (Partie VIII).

Protocole :
* matériel et logiciels relevés automatiquement ;
* préchauffage de 200 appels non mesurés ;
* latence unitaire : 2 000 répétitions, une transaction par appel ;
* débit : lots de 100 / 1 000 / 10 000 transactions, 20 répétitions ;
* trois niveaux distingués :
  (1) pipeline seul (predict_proba scikit-learn),
  (2) predict_fraud : validation du contrat + pipeline + alerte de domaine,
  (3) requête HTTP locale : sérialisation JSON + HTTP + validation stricte + prédiction ;
* mémoire : RSS du processus (psutil) avant/après chargement et pendant un lot de 56 746 lignes.
Horloge : time.perf_counter.

Usage : python scripts/benchmark.py --out reports
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import psutil
import sklearn
from _common import ROOT, dump_json

from app import config
from app.prediction import handle_request, load_model, predict_fraud
from app.synthetic import make_synthetic_data

WARMUP, N_SINGLE, N_HTTP, BATCH_REPEATS = 200, 2000, 1000, 20


def pct(values_ms):
    a = np.asarray(values_ms)
    return {
        "p50_ms": float(np.percentile(a, 50)),
        "p95_ms": float(np.percentile(a, 95)),
        "p99_ms": float(np.percentile(a, 99)),
        "mean_ms": float(a.mean()),
        "max_ms": float(a.max()),
        "n": int(a.size),
    }


def time_calls(fn, n):
    for _ in range(WARMUP):
        fn()
    out = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        out.append((time.perf_counter() - t0) * 1000)
    return pct(out)


def cpu_name() -> str:
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "inconnu"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports")
    ap.add_argument("--model-path", default="models/model.joblib")
    args = ap.parse_args()
    proc = psutil.Process(os.getpid())
    model_path = ROOT / args.model_path

    gc.collect()
    rss_before = proc.memory_info().rss
    bundle = load_model(model_path)
    rss_loaded = proc.memory_info().rss

    # Données d'entrée au schéma réel (synthétiques : seule la forme compte ici).
    data = make_synthetic_data(n=60_000, fraud_rate=0.002, seed=3)
    X = data.drop(columns="Class")
    one = X.iloc[[0]]
    pipe = bundle.pipeline

    single_pipeline = time_calls(lambda: pipe.predict_proba(one), N_SINGLE)
    single_predict = time_calls(lambda: predict_fraud(bundle, one), N_SINGLE)
    body_one = json.dumps({"transactions": one.to_dict(orient="records")})
    single_handler = time_calls(lambda: handle_request(body_one, bundle), N_SINGLE)

    # ---- HTTP local ----
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            if length > 2_000_000:
                self.send_response(413)
                self.end_headers()
                return
            status, resp = handle_request(self.rfile.read(length), bundle)
            payload = json.dumps(resp).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *a):  # pas de journalisation des requêtes
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{port}/predict"

    body_one_bytes = body_one.encode()

    def http_call(body=body_one_bytes):
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as r:
            r.read()

    for _ in range(WARMUP // 4):
        http_call()
    http_lat = []
    for _ in range(N_HTTP):
        t0 = time.perf_counter()
        http_call()
        http_lat.append((time.perf_counter() - t0) * 1000)
    http_single = pct(http_lat)
    bad = json.dumps({"transactions": [{"Amount": "x"}]}).encode()
    try:
        http_call(bad)
        http_bad_status = 200
    except urllib.error.HTTPError as e:
        http_bad_status = e.code
    server.shutdown()

    # ---- Débit par lots ----
    throughput = {}
    for size in (100, 1_000, 10_000):
        batch = X.iloc[:size]
        predict_fraud(bundle, batch)
        times = []
        for _ in range(BATCH_REPEATS):
            t0 = time.perf_counter()
            predict_fraud(bundle, batch)
            times.append(time.perf_counter() - t0)
        med = float(np.median(times))
        throughput[str(size)] = {
            "median_s": med,
            "p95_s": float(np.percentile(times, 95)),
            "tx_per_s": size / med,
        }

    peak = rss_loaded
    stop = threading.Event()

    def sampler():
        nonlocal peak
        while not stop.is_set():
            peak = max(peak, proc.memory_info().rss)
            time.sleep(0.002)

    th = threading.Thread(target=sampler)
    th.start()
    for _ in range(5):
        predict_fraud(bundle, X.iloc[:56_746])
    stop.set()
    th.join()

    size_mb = model_path.stat().st_size / 1e6
    peak_mb = peak / 1e6
    results = {
        "environment": {
            "cpu": cpu_name(),
            "logical_cores": psutil.cpu_count(),
            "ram_gb": round(psutil.virtual_memory().total / 1e9, 1),
            "os": platform.platform(),
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "note": "conteneur Linux partagé ; mono-processus ; aucun autre calcul simultané",
        },
        "protocol": {
            "warmup": WARMUP,
            "n_single": N_SINGLE,
            "n_http": N_HTTP,
            "batch_repeats": BATCH_REPEATS,
            "clock": "time.perf_counter",
        },
        "single": {
            "pipeline_only": single_pipeline,
            "predict_fraud": single_predict,
            "request_handler_no_http": single_handler,
            "http_local": http_single,
        },
        "http_invalid_request_status": http_bad_status,
        "throughput": throughput,
        "model_size_mb": size_mb,
        "rss_mb": {
            "before_load": rss_before / 1e6,
            "after_load": rss_loaded / 1e6,
            "peak_batch_56746": peak_mb,
        },
        "criteria": config.PERF,
        "checks": {
            "p95_single_predict_fraud": single_predict["p95_ms"]
            <= config.PERF["p95_single_ms_max"],
            "p95_http_local": http_single["p95_ms"] <= config.PERF["p95_single_ms_max"],
            "throughput_batch_1000": throughput["1000"]["tx_per_s"]
            >= config.PERF["throughput_min_per_s"],
            "model_size": size_mb <= config.PERF["model_size_mb_max"],
            "rss_peak": peak_mb <= config.PERF["rss_mb_max"],
        },
    }
    dump_json(results, ROOT / args.out / "results" / "benchmark.json")
    print(
        json.dumps({k: results[k] for k in ("single", "throughput", "rss_mb", "checks")}, indent=1)
    )
    print("taille modèle (Mo):", round(size_mb, 4), "HTTP invalide ->", http_bad_status)


if __name__ == "__main__":
    main()
