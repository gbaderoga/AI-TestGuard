"""Crée (ou met à jour, sur décision explicite) la baseline de non-régression de la CI.

Usage : python scripts/make_ci_baseline.py [--force]

Cette commande ne doit être lancée qu'après une décision documentée
(Pull Request dédiée). Les tests ne la recalculent jamais.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402

from app import __version__, config  # noqa: E402
from app.evaluation import evaluate_model  # noqa: E402
from app.preprocessing import preprocess_data, split_data  # noqa: E402
from app.synthetic import make_synthetic_data  # noqa: E402
from app.training import select_threshold, train_model  # noqa: E402

OUT = ROOT / "tests" / "baselines" / "ci_baseline.json"
DATASET = {"n": 6000, "fraud_rate": 0.02, "seed": 0}
MODEL, PARAMS = "logreg", {"C": 0.1}


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "non-versionné"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="écraser la baseline existante")
    args = parser.parse_args()
    if OUT.exists() and not args.force:
        sys.exit(f"{OUT} existe déjà. Utiliser --force après décision documentée.")

    df = make_synthetic_data(**DATASET)
    train, val, _ = split_data(df, seed=config.SEED)
    (X_tr, y_tr), (X_va, y_va) = preprocess_data(train), preprocess_data(val)
    pipe = train_model(X_tr, y_tr, MODEL, PARAMS, config.SEED)
    proba = pipe.predict_proba(X_va)[:, 1]
    threshold = select_threshold(y_va, proba)
    metrics = evaluate_model(y_va, (proba >= threshold).astype(int), proba)

    baseline = {
        "description": "Référence CI (données synthétiques) — ne pas recalculer automatiquement",
        "code_version": __version__,
        "git_commit": git_commit(),
        "model": MODEL,
        "params": PARAMS,
        "seed": config.SEED,
        "threshold": threshold,
        "library_versions": {
            "python": sys.version.split()[0],
            "scikit-learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "joblib": joblib.__version__,
        },
        "dataset": {
            **DATASET,
            "kind": "synthétique (app/synthetic.py)",
            "n_val": len(y_va),
            "n_val_fraud": int(y_va.sum()),
        },
        "metrics": metrics,
        "proba_head": proba[:20].tolist(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(baseline, indent=2, ensure_ascii=False) + "\n")
    print(f"Baseline écrite : {OUT}")


if __name__ == "__main__":
    main()
