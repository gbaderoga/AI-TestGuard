"""Non-régression sur le jeu réel (Partie VI.2).

1. `--freeze` : fige la baseline v1.0 (modèle de models/model.joblib) dans
   reports/baselines/baseline_v1.0.json, uniquement si elle n'existe pas déjà.
2. Sans option : ré-entraîne chaque version candidate, mesure ses métriques sur
   le MÊME jeu de validation figé et produit une décision accepter / investiguer / rejeter.

La baseline n'est jamais recalculée automatiquement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
import sklearn
from _common import ROOT, dump_json, load_splits

from app import __version__, config
from app.evaluation import compare_to_baseline, evaluate_model
from app.prediction import load_model
from app.training import select_threshold, train_model

BASELINE = ROOT / "reports" / "baselines" / "baseline_v1.0.json"

CANDIDATES = {
    "v1.1-a : régression logistique, C = 0.1 (au lieu de 0.01)": ("logreg", {"C": 0.1}),
    "v1.1-b : régression logistique sans pondération des classes": (
        "logreg",
        {"C": 0.01, "class_weight": None},
    ),
    "v1.1-c : arbre de décision (profondeur 8)": (
        "tree",
        {"max_depth": 8, "min_samples_leaf": 20},
    ),
    "v1.1-d : régression logistique, graine différente (contrôle)": (
        "logreg",
        {"C": 0.01, "random_state": 7},
    ),
}

KEYS = ("average_precision", "roc_auc", "recall", "precision", "fpr", "f1", "tp", "fp", "fn")


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "non-versionné"


def val_fingerprint(X: pd.DataFrame, y) -> str:
    h = hashlib.sha256(pd.util.hash_pandas_object(X, index=False).to_numpy().tobytes())
    h.update(np.asarray(y).tobytes())
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/creditcard.csv")
    ap.add_argument("--out", default="reports")
    ap.add_argument("--freeze", action="store_true")
    args = ap.parse_args()
    df, splits = load_splits(args.data)
    X_tr, y_tr = splits["train"]
    X_va, y_va = splits["val"]
    fp = val_fingerprint(X_va, y_va)

    if args.freeze:
        if BASELINE.exists():
            sys.exit(f"{BASELINE} existe déjà : la baseline ne se recalcule pas.")
        bundle = load_model(ROOT / "models" / "model.joblib")
        p = bundle.pipeline.predict_proba(X_va)[:, 1]
        m = evaluate_model(y_va, (p >= bundle.threshold).astype(int), p)
        dump_json(
            {
                "version": "v1.0",
                "code_version": __version__,
                "git_commit": git_commit(),
                "model": bundle.metadata["model"],
                "params": bundle.metadata["params"],
                "seed": config.SEED,
                "threshold": bundle.threshold,
                "library_versions": {
                    "python": platform.python_version(),
                    "scikit-learn": sklearn.__version__,
                    "numpy": np.__version__,
                    "pandas": pd.__version__,
                    "joblib": joblib.__version__,
                },
                "dataset": {
                    "source": "Kaggle mlg-ulb/creditcardfraud",
                    "sha256_csv": hashlib.sha256((ROOT / args.data).read_bytes()).hexdigest(),
                    "n_raw": len(df),
                    "n_fraud_raw": int(df["Class"].sum()),
                    "split": "stratifié 60/20/20, doublons retirés, graine 42",
                    "n_val": len(y_va),
                    "n_val_fraud": int(y_va.sum()),
                    "val_fingerprint": fp,
                },
                "metrics_val": {k: m[k] for k in KEYS},
                "tolerance": config.REGRESSION_TOLERANCE,
            },
            BASELINE,
        )
        print(f"Baseline figée : {BASELINE}")
        return

    base = json.loads(BASELINE.read_text())
    if base["dataset"]["val_fingerprint"] != fp:
        sys.exit("Le jeu de validation diffère de celui de la baseline : comparaison invalide.")

    report = {"baseline": base, "candidates": {}}
    for name, (model, params) in CANDIDATES.items():
        pipe = train_model(X_tr, y_tr, model, params)
        p = pipe.predict_proba(X_va)[:, 1]
        thr = select_threshold(y_va, p)  # règle de seuil inchangée, sur la validation
        m = evaluate_model(y_va, (p >= thr).astype(int), p)
        cmp = compare_to_baseline(m, base["metrics_val"])
        report["candidates"][name] = {
            "model": model,
            "params": params,
            "threshold": thr,
            "metrics_val": {k: m[k] for k in KEYS},
            **cmp,
        }
        print(
            f"{name:65s} -> {cmp['decision']:11s} "
            + " ".join(
                f"{k}={m[k]:.4f}"
                for k in ("average_precision", "recall", "precision", "fpr", "roc_auc")
            )
        )
    dump_json(report, ROOT / args.out / "results" / "regression_comparison.json")


if __name__ == "__main__":
    main()
