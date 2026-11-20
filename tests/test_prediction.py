"""Tests de l'entraînement, de la prédiction, des métriques et de l'API (T04, T11-T16)."""

from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd
import pytest

from app import config
from app.evaluation import (
    bootstrap_ci,
    check_acceptance,
    compare_to_baseline,
    evaluate_model,
)
from app.prediction import (
    MAX_BATCH,
    ModelBundle,
    handle_request,
    parse_api_payload,
    predict_fraud,
)
from app.preprocessing import FEATURE_COLUMNS, DataValidationError
from app.training import (
    build_pipeline,
    expected_cost,
    select_model,
    select_threshold,
    train_model,
)


# --------------------------------------------------------------------------- #
# T04 - contrat de sortie                                                     #
# --------------------------------------------------------------------------- #
def test_prediction_probabilities_are_valid(bundle, X_test):
    out = predict_fraud(bundle, X_test)
    p = out["fraud_probability"].to_numpy()
    assert np.isfinite(p).all()
    assert ((p >= 0) & (p <= 1)).all()
    assert set(out["is_fraud"].unique()) <= {0, 1}
    assert out.index.equals(X_test.index)
    assert len(out) == len(X_test)


def test_decision_is_consistent_with_threshold(bundle, X_test):
    out = predict_fraud(bundle, X_test)
    expected = (out["fraud_probability"] >= bundle.threshold).astype(int)
    assert (out["is_fraud"] == expected).all()


def test_explicit_threshold_overrides_bundle(bundle, X_test):
    assert predict_fraud(bundle, X_test, threshold=0.0)["is_fraud"].all()
    assert predict_fraud(bundle, X_test, threshold=1.0)["is_fraud"].sum() <= 1


@pytest.mark.parametrize("threshold", [-0.1, 1.5])
def test_invalid_threshold_is_rejected(bundle, X_test, threshold):
    with pytest.raises(ValueError, match="seuil"):
        predict_fraud(bundle, X_test, threshold=threshold)


def test_prediction_on_raw_pipeline_uses_default_threshold(bundle, X_test):
    out = predict_fraud(bundle.pipeline, X_test)
    assert "out_of_domain_alert" not in out.columns
    assert (out["is_fraud"] == (out["fraud_probability"] >= 0.5)).all()


def test_prediction_is_row_order_invariant(bundle, X_test):
    a = predict_fraud(bundle, X_test)
    perm = X_test.sample(frac=1.0, random_state=0)
    b = predict_fraud(bundle, perm).loc[a.index]
    np.testing.assert_allclose(a["fraud_probability"], b["fraud_probability"], atol=1e-12)


def test_single_row_equals_batch_prediction(bundle, X_test):
    batch = predict_fraud(bundle, X_test)["fraud_probability"]
    for i in X_test.index[:10]:
        single = predict_fraud(bundle, X_test.loc[[i]])["fraud_probability"].iloc[0]
        assert abs(single - batch.loc[i]) < 1e-12


def test_prediction_rejects_invalid_input(bundle, X_test):
    bad = X_test.copy()
    bad.iloc[0, 0] = np.nan
    with pytest.raises(DataValidationError):
        predict_fraud(bundle, bad)
    with pytest.raises(DataValidationError, match="absentes"):
        predict_fraud(bundle, X_test.drop(columns="V5"))


def test_out_of_domain_alert(bundle, X_test):
    X = X_test.iloc[:5].copy()
    X.loc[X.index[0], "Amount"] = bundle.feature_max["Amount"] * 100
    out = predict_fraud(bundle, X)
    assert out["out_of_domain_alert"].iloc[0]
    assert np.isfinite(out["fraud_probability"]).all()


def test_bundle_records_metadata(bundle):
    for key in ("code_version", "sklearn_version", "numpy_version", "python_version"):
        assert key in bundle.metadata
    assert 0.0 <= bundle.threshold <= 1.0
    assert set(bundle.feature_min) == set(FEATURE_COLUMNS)


def test_bundle_rejects_invalid_threshold(bundle, splits):
    with pytest.raises(ValueError):
        ModelBundle.from_training(bundle.pipeline, 1.2, splits["train"][0], {})


# --------------------------------------------------------------------------- #
# Entraînement                                                                #
# --------------------------------------------------------------------------- #
def test_unknown_model_is_rejected():
    with pytest.raises(ValueError, match="inconnu"):
        build_pipeline("xgboost")


def test_train_rejects_single_class(splits):
    X, _ = splits["train"]
    with pytest.raises(ValueError, match="deux classes"):
        train_model(X, np.zeros(len(X), dtype=int))


def test_train_rejects_misaligned_inputs(splits):
    X, y = splits["train"]
    with pytest.raises(ValueError, match="incohérentes"):
        train_model(X, y.iloc[:-1])


def test_train_rejects_non_dataframe(splits):
    X, y = splits["train"]
    with pytest.raises(TypeError):
        train_model(X.to_numpy(), y)


def test_train_rejects_missing_feature(splits):
    X, y = splits["train"]
    with pytest.raises(ValueError, match="absentes"):
        train_model(X.drop(columns="Amount"), y)


@pytest.mark.parametrize("model_name", ["logreg", "tree"])
def test_both_models_learn_better_than_chance(splits, model_name):
    X_train, y_train = splits["train"]
    X_val, y_val = splits["val"]
    res = select_model(X_train, y_train, X_val, y_val, model_name)
    prevalence = float(np.mean(y_val))
    assert res.val_average_precision > 5 * prevalence
    assert len(res.history) == len(config.PARAM_GRIDS[model_name])


# --------------------------------------------------------------------------- #
# Seuil                                                                       #
# --------------------------------------------------------------------------- #
def test_threshold_minimises_cost_brute_force(rng):
    y = rng.binomial(1, 0.05, 500)
    p = np.clip(0.6 * y + rng.normal(0.2, 0.2, 500), 0, 1)
    t = select_threshold(y, p, cost_fn=100, cost_fp=5)
    brute = min(expected_cost(y, p, c, 100, 5) for c in np.unique(np.r_[p, 0.5]))
    assert expected_cost(y, p, t, 100, 5) == brute


def test_threshold_moves_with_cost_ratio(rng):
    y = rng.binomial(1, 0.05, 2000)
    p = np.clip(0.4 * y + rng.normal(0.3, 0.15, 2000), 0, 1)
    t_fn_expensive = select_threshold(y, p, cost_fn=1000, cost_fp=1)
    t_fp_expensive = select_threshold(y, p, cost_fn=1, cost_fp=1000)
    assert t_fn_expensive < t_fp_expensive


def test_threshold_rejects_bad_inputs():
    with pytest.raises(ValueError):
        select_threshold([], [])
    with pytest.raises(ValueError):
        select_threshold([0, 1], [0.2])


# --------------------------------------------------------------------------- #
# Métriques (evaluate_model)                                                  #
# --------------------------------------------------------------------------- #
def test_metrics_on_known_confusion_matrix():
    # TP=2, FN=1, FP=1, TN=6
    y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    y_pred = np.array([1, 1, 0, 1, 0, 0, 0, 0, 0, 0])
    m = evaluate_model(y_true, y_pred)
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (2, 1, 1, 6)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["f1"] == pytest.approx(2 / 3)
    assert m["fpr"] == pytest.approx(1 / 7)
    assert m["fnr"] == pytest.approx(1 / 3)
    assert m["accuracy"] == pytest.approx(0.8)


def test_high_accuracy_can_hide_zero_recall():
    """Pourquoi l'accuracy ne suffit pas : prédire « normal » partout."""
    y_true = np.r_[np.ones(17), np.zeros(9983)].astype(int)
    m = evaluate_model(y_true, np.zeros_like(y_true))
    assert m["accuracy"] > 0.998
    assert m["recall"] == 0.0
    assert m["precision"] == 0.0  # zero_division -> 0, pas d'exception
    assert m["fnr"] == 1.0


def test_auc_is_none_when_single_class():
    m = evaluate_model([0, 0, 0], [0, 1, 0], [0.1, 0.7, 0.2])
    assert m["roc_auc"] is None and m["average_precision"] is None


def test_perfect_ranking_gives_auc_one():
    m = evaluate_model([0, 0, 1, 1], [0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert m["roc_auc"] == 1.0 and m["average_precision"] == 1.0


@pytest.mark.parametrize(
    "y_true,y_pred,y_proba",
    [
        ([0, 1], [0], None),
        ([], [], None),
        ([0, 2], [0, 1], None),
        ([0, 1], [0, 1], [0.1, 1.2]),
        ([0, 1], [0, 1], [0.1, np.nan]),
    ],
)
def test_metrics_reject_invalid_inputs(y_true, y_pred, y_proba):
    with pytest.raises(ValueError):
        evaluate_model(y_true, y_pred, y_proba)


def test_bootstrap_ci_contains_point_estimate(rng):
    y = rng.binomial(1, 0.1, 1000)
    p = np.clip(0.5 * y + rng.normal(0.25, 0.15, 1000), 0, 1)
    point = evaluate_model(y, (p >= 0.5).astype(int), p)
    ci = bootstrap_ci(y, p, 0.5, n_boot=200)
    for k in ("recall", "precision", "fpr"):
        lo, hi = ci[k]
        assert lo <= point[k] <= hi
        assert hi - lo > 0


def test_acceptance_criteria():
    good = {
        "roc_auc": 0.97,
        "average_precision": 0.8,
        "recall": 0.8,
        "precision": 0.6,
        "fpr": 0.001,
    }
    assert all(check_acceptance(good).values())
    bad = {**good, "recall": 0.5}
    result = check_acceptance(bad)
    assert not result["recall"] and result["precision"]


# --------------------------------------------------------------------------- #
# Décision de non-régression                                                  #
# --------------------------------------------------------------------------- #
BASE = {"average_precision": 0.80, "roc_auc": 0.97, "recall": 0.80, "precision": 0.70, "fpr": 0.001}


def test_identical_version_is_accepted():
    assert compare_to_baseline(BASE, BASE)["decision"] == "accepter"


def test_large_degradation_is_rejected():
    cur = {**BASE, "average_precision": 0.75}
    res = compare_to_baseline(cur, BASE)
    assert res["decision"] == "rejeter"
    assert res["details"]["average_precision"]["status"] == "rejeter"


def test_fpr_increase_is_a_degradation():
    cur = {**BASE, "fpr": 0.003}
    assert compare_to_baseline(cur, BASE)["details"]["fpr"]["status"] == "rejeter"


def test_small_degradation_triggers_investigation():
    cur = {**BASE, "recall": 0.78}  # -0.02 > tolérance/2 = 0.015
    assert compare_to_baseline(cur, BASE)["decision"] == "investiguer"


def test_suspicious_improvement_triggers_investigation():
    cur = {**BASE, "average_precision": 0.95}  # +0.15 : fuite de données ?
    assert compare_to_baseline(cur, BASE)["decision"] == "investiguer"


# --------------------------------------------------------------------------- #
# Entrées de type API (sécurité opérationnelle)                               #
# --------------------------------------------------------------------------- #
def _payload(X: pd.DataFrame, n: int = 2) -> dict:
    return {"transactions": X.iloc[:n].to_dict(orient="records")}


def test_api_valid_request(bundle, X_test):
    status, body = handle_request(json.dumps(_payload(X_test)), bundle)
    assert status == 200
    assert len(body["predictions"]) == 2
    assert all(0 <= p["fraud_probability"] <= 1 for p in body["predictions"])
    assert body["threshold"] == bundle.threshold


def test_api_accepts_bytes(bundle, X_test):
    status, _ = handle_request(json.dumps(_payload(X_test)).encode(), bundle)
    assert status == 200


@pytest.mark.parametrize(
    "mutate,message",
    [
        (lambda r: r.pop("V3"), "manquants.*V3"),
        (lambda r: r.update(iban="FR76..."), "non autorisés.*iban"),
        (lambda r: r.update(Amount="100"), "non numériques.*Amount"),
        (lambda r: r.update(Amount=True), "non numériques.*Amount"),
        (lambda r: r.update(Amount=None), "non numériques.*Amount"),
        (lambda r: r.update(Amount=[1, 2]), "non numériques.*Amount"),
    ],
    ids=["missing", "extra", "string", "bool", "null", "list"],
)
def test_api_rejects_bad_fields(bundle, X_test, mutate, message):
    payload = _payload(X_test, 1)
    mutate(payload["transactions"][0])
    status, body = handle_request(json.dumps(payload), bundle)
    assert status == 400
    assert re.search(message, body["error"])


@pytest.mark.parametrize(
    "body",
    [
        "not json",
        "[]",
        '{"transactions": []}',
        '{"transactions": [1]}',
        '{"transactions": [], "debug": true}',
        b"\xff\xfe",
    ],
)
def test_api_rejects_malformed_bodies(bundle, body):
    status, resp = handle_request(body, bundle)
    assert status == 400
    assert "error" in resp


def test_api_rejects_nan_and_infinity_literals(bundle, X_test):
    raw = json.dumps(_payload(X_test, 1)).replace(str(X_test.iloc[0]["V1"]), "NaN", 1)
    status, _ = handle_request(raw, bundle)
    assert status == 400
    status, _ = handle_request(raw.replace("NaN", "Infinity"), bundle)
    assert status == 400


def test_api_rejects_oversized_batch(bundle, X_test):
    row = X_test.iloc[0].to_dict()
    body = json.dumps({"transactions": [row] * (MAX_BATCH + 1)})
    status, resp = handle_request(body, bundle)
    assert status == 400 and "volumineux" in resp["error"]


def test_api_rejects_negative_amount(bundle, X_test):
    payload = _payload(X_test, 1)
    payload["transactions"][0]["Amount"] = -5
    assert handle_request(json.dumps(payload), bundle)[0] == 400


def test_logs_do_not_contain_transaction_values(bundle, X_test, caplog):
    payload = _payload(X_test, 3)
    marker = 987654.321
    payload["transactions"][0]["Amount"] = marker
    with caplog.at_level("DEBUG"):
        handle_request(json.dumps(payload), bundle)
        payload["transactions"][1].pop("V2")
        handle_request(json.dumps(payload), bundle)
    assert caplog.records  # des événements sont bien journalisés
    assert "987654" not in caplog.text
    for v in X_test.iloc[0][["V1", "V2", "V3"]]:
        assert f"{v:.6f}"[:8] not in caplog.text


def test_parse_payload_returns_contract_columns(X_test):
    df = parse_api_payload(json.dumps(_payload(X_test, 3)))
    assert list(df.columns) == FEATURE_COLUMNS
    assert (df.dtypes == "float64").all()
