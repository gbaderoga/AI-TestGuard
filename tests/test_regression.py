"""Tests de non-régression (T06, T26-T27).

La référence `tests/baselines/ci_baseline.json` est FIGÉE et versionnée.
Elle n'est jamais recalculée automatiquement : seule une décision humaine,
tracée dans une Pull Request, peut la mettre à jour
(`python scripts/make_ci_baseline.py`), sinon les régressions seraient masquées.

Échec volontaire : définir AITG_SIMULATE_REGRESSION=1 dégrade
délibérément l'entraînement (labels d'entraînement permutés) pour démontrer
que la CI détecte et signale la régression.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import sklearn

from app import config
from app.evaluation import compare_to_baseline, evaluate_model
from app.preprocessing import preprocess_data, split_data
from app.synthetic import make_synthetic_data
from app.training import select_threshold, train_model

BASELINE_PATH = Path(__file__).parent / "baselines" / "ci_baseline.json"


def train_reference_candidate(simulate_regression: bool = False) -> dict:
    """Reproduit exactement la procédure de référence (données, graine, paramètres)."""
    baseline = json.loads(BASELINE_PATH.read_text())
    ds = baseline["dataset"]
    df = make_synthetic_data(n=ds["n"], fraud_rate=ds["fraud_rate"], seed=ds["seed"])
    train, val, _ = split_data(df, seed=baseline["seed"])
    (X_tr, y_tr), (X_va, y_va) = preprocess_data(train), preprocess_data(val)
    if simulate_regression:
        y_tr = y_tr.sample(frac=1.0, random_state=0).reset_index(drop=True)
    pipe = train_model(X_tr, y_tr, baseline["model"], baseline["params"], baseline["seed"])
    proba = pipe.predict_proba(X_va)[:, 1]
    # Seuil figé de la baseline : on compare la même règle de décision.
    pred = (proba >= baseline["threshold"]).astype(int)
    metrics = evaluate_model(y_va, pred, proba)
    metrics["threshold_reselected"] = select_threshold(y_va, proba)
    metrics["proba_head"] = proba[:20].tolist()
    return metrics


@pytest.fixture(scope="module")
def baseline() -> dict:
    return json.loads(BASELINE_PATH.read_text())


@pytest.fixture(scope="module")
def candidate() -> dict:
    return train_reference_candidate(os.environ.get("AITG_SIMULATE_REGRESSION") == "1")


def test_baseline_records_required_context(baseline):
    for key in (
        "code_version",
        "git_commit",
        "model",
        "params",
        "seed",
        "threshold",
        "library_versions",
        "dataset",
        "metrics",
    ):
        assert key in baseline, key
    assert {"n", "fraud_rate", "seed", "n_val", "n_val_fraud"} <= set(baseline["dataset"])


def test_no_performance_regression(baseline, candidate):
    result = compare_to_baseline(candidate, baseline["metrics"])
    failing = {k: v for k, v in result["details"].items() if v["status"] == "rejeter"}
    assert result["decision"] != "rejeter", f"Régression détectée : {json.dumps(failing, indent=2)}"


def test_validation_set_is_unchanged(baseline, candidate):
    assert candidate["n"] == baseline["dataset"]["n_val"]
    assert candidate["n_fraud"] == baseline["dataset"]["n_val_fraud"]


def test_predictions_match_baseline_numerically(baseline, candidate):
    """Même code + mêmes versions => mêmes probabilités (tolérance REPRO_ATOL).

    Si les versions des bibliothèques diffèrent, on tolère un écart plus large
    (1e-6) : des optimisations numériques peuvent changer les derniers chiffres.
    """
    same_libs = baseline["library_versions"]["scikit-learn"] == sklearn.__version__
    atol = config.REPRO_ATOL if same_libs else 1e-6
    np.testing.assert_allclose(candidate["proba_head"], baseline["proba_head"], atol=atol, rtol=0)


def test_threshold_selection_is_stable(baseline, candidate):
    assert candidate["threshold_reselected"] == pytest.approx(baseline["threshold"], abs=1e-6)
