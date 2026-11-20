"""Tests d'intégration bout en bout (T05, T06, T17-T19).

Scénario : CSV -> validation -> partition -> ajustement (train) -> seuil
(validation) -> sauvegarde -> rechargement -> prédiction (test) -> rapport.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app import config
from app.evaluation import evaluate_model
from app.prediction import ModelBundle, ModelLoadError, load_model, predict_fraud, save_model
from app.preprocessing import (
    FEATURE_COLUMNS,
    PCA_COLUMNS,
    load_data,
    preprocess_data,
    split_data,
)
from app.synthetic import make_synthetic_data
from app.training import select_model, select_threshold

REAL_DATA = Path(os.environ.get("AITG_DATA_PATH", "data/creditcard.csv"))


def run_pipeline(csv_path: Path, model_dir: Path, model_name: str = "logreg") -> dict:
    df = load_data(csv_path)
    train, val, test = split_data(df)
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = map(preprocess_data, (train, val, test))
    sel = select_model(X_tr, y_tr, X_va, y_va, model_name)
    threshold = select_threshold(y_va, sel.pipeline.predict_proba(X_va)[:, 1])
    bundle = ModelBundle.from_training(
        sel.pipeline, threshold, X_tr, {"model": model_name, "params": sel.params}
    )
    path = model_dir / "model.joblib"
    digest = save_model(bundle, path)
    reloaded = load_model(path)
    before = predict_fraud(bundle, X_te)
    after = predict_fraud(reloaded, X_te)
    metrics = evaluate_model(y_te, after["is_fraud"], after["fraud_probability"])
    report = model_dir / "metrics.json"
    report.write_text(json.dumps(metrics, indent=2))
    return {
        "bundle": bundle,
        "reloaded": reloaded,
        "before": before,
        "after": after,
        "metrics": metrics,
        "digest": digest,
        "splits": {"train": X_tr, "val": X_va, "test": X_te},
        "report": report,
        "path": path,
    }


@pytest.fixture(scope="module")
def pipeline_run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("run")
    csv_path = tmp / "data.csv"
    make_synthetic_data(n=4000, fraud_rate=0.02, seed=0).to_csv(csv_path, index=False)
    return run_pipeline(csv_path, tmp)


def test_end_to_end_produces_report(pipeline_run):
    m = json.loads(pipeline_run["report"].read_text())
    for key in ("tp", "fp", "fn", "tn", "precision", "recall", "roc_auc", "average_precision"):
        assert key in m
    assert m["tp"] + m["fp"] + m["fn"] + m["tn"] == len(pipeline_run["splits"]["test"])


def test_dimensions_are_consistent(pipeline_run):
    X_te = pipeline_run["splits"]["test"]
    Z = pipeline_run["bundle"].pipeline.named_steps["preprocess"].transform(X_te)
    assert Z.shape == (len(X_te), len(FEATURE_COLUMNS))
    assert np.isfinite(Z).all()
    assert len(pipeline_run["after"]) == len(X_te)


def test_predictions_identical_after_reload(pipeline_run):
    b, a = pipeline_run["before"], pipeline_run["after"]
    np.testing.assert_allclose(
        b["fraud_probability"], a["fraud_probability"], atol=config.REPRO_ATOL, rtol=0
    )
    assert (b["is_fraud"] == a["is_fraud"]).all()
    assert pipeline_run["reloaded"].threshold == pipeline_run["bundle"].threshold
    assert pipeline_run["reloaded"].metadata == pipeline_run["bundle"].metadata


def test_scaler_is_fitted_on_training_data_only(pipeline_run):
    prep = pipeline_run["bundle"].pipeline.named_steps["preprocess"]
    X_tr, X_te = pipeline_run["splits"]["train"], pipeline_run["splits"]["test"]
    other = prep.named_transformers_["other"]
    amount = prep.named_transformers_["amount"].named_steps["scale"]
    cols = ["Time", *PCA_COLUMNS]
    np.testing.assert_allclose(other.mean_, X_tr[cols].mean().to_numpy(), rtol=1e-10)
    np.testing.assert_allclose(amount.mean_, [np.log1p(X_tr["Amount"]).mean()], rtol=1e-10)
    assert other.n_samples_seen_ == len(X_tr)
    # Les statistiques ne correspondent ni au test ni à l'ensemble complet.
    full = pd.concat([X_tr, pipeline_run["splits"]["val"], X_te])
    assert not np.allclose(other.mean_, X_te[cols].mean().to_numpy())
    assert not np.allclose(other.mean_, full[cols].mean().to_numpy())


def test_threshold_was_chosen_on_validation(pipeline_run):
    """Le seuil doit être une probabilité observée sur la validation (ou 0.5)."""
    bundle, X_va = pipeline_run["bundle"], pipeline_run["splits"]["val"]
    proba_val = bundle.pipeline.predict_proba(X_va)[:, 1]
    assert np.isclose(proba_val, bundle.threshold, atol=1e-15).any() or bundle.threshold == 0.5


def test_training_is_reproducible(tmp_path_factory, pipeline_run):
    tmp = tmp_path_factory.mktemp("rerun")
    csv_path = tmp / "data.csv"
    make_synthetic_data(n=4000, fraud_rate=0.02, seed=0).to_csv(csv_path, index=False)
    second = run_pipeline(csv_path, tmp)
    np.testing.assert_allclose(
        second["after"]["fraud_probability"],
        pipeline_run["after"]["fraud_probability"],
        atol=config.REPRO_ATOL,
        rtol=0,
    )
    assert second["bundle"].threshold == pipeline_run["bundle"].threshold


def test_tree_pipeline_end_to_end(tmp_path):
    csv_path = tmp_path / "data.csv"
    make_synthetic_data(n=3000, fraud_rate=0.03, seed=5).to_csv(csv_path, index=False)
    run = run_pipeline(csv_path, tmp_path, model_name="tree")
    assert run["metrics"]["roc_auc"] > 0.5
    assert (run["before"]["is_fraud"] == run["after"]["is_fraud"]).all()


# --------------------------------------------------------------------------- #
# Erreurs de chargement                                                       #
# --------------------------------------------------------------------------- #
def test_load_missing_model_is_explicit(tmp_path):
    with pytest.raises(ModelLoadError, match="introuvable"):
        load_model(tmp_path / "nope.joblib")


def test_load_corrupted_model_is_explicit(tmp_path, pipeline_run):
    path = tmp_path / "corrupt.joblib"
    path.write_bytes(pipeline_run["path"].read_bytes()[:200])
    with pytest.raises(ModelLoadError, match="corrompu"):
        load_model(path)


def test_load_wrong_object_is_rejected(tmp_path):
    import joblib

    path = tmp_path / "other.joblib"
    joblib.dump({"not": "a bundle"}, path)
    with pytest.raises(ModelLoadError, match="inattendu"):
        load_model(path)


def test_checksum_is_verified_before_deserialisation(tmp_path, pipeline_run):
    path, digest = pipeline_run["path"], pipeline_run["digest"]
    assert load_model(path, expected_sha256=digest).threshold == pipeline_run["bundle"].threshold
    tampered = tmp_path / "tampered.joblib"
    tampered.write_bytes(path.read_bytes() + b"\x00")
    with pytest.raises(ModelLoadError, match="Empreinte"):
        load_model(tampered, expected_sha256=digest)


def test_save_rejects_non_bundle(tmp_path, pipeline_run):
    with pytest.raises(TypeError):
        save_model(pipeline_run["bundle"].pipeline, tmp_path / "x.joblib")


def test_version_mismatch_is_logged(tmp_path, pipeline_run, caplog):
    bundle = pipeline_run["bundle"]
    tampered = ModelBundle(
        bundle.pipeline,
        bundle.threshold,
        bundle.feature_min,
        bundle.feature_max,
        {**bundle.metadata, "sklearn_version": "0.0.0"},
    )
    path = tmp_path / "old.joblib"
    save_model(tampered, path)
    with caplog.at_level("WARNING"):
        load_model(path)
    assert "Version scikit-learn différente" in caplog.text


# --------------------------------------------------------------------------- #
# Jeu réel (exécuté seulement si le fichier Kaggle est présent)               #
# --------------------------------------------------------------------------- #
@pytest.mark.slow
@pytest.mark.skipif(not REAL_DATA.is_file(), reason="creditcard.csv absent (non versionné)")
def test_real_dataset_contract():
    df = load_data(REAL_DATA)
    assert len(df) == 284_807
    assert int(df["Class"].sum()) == 492
