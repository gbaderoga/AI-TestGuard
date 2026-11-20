"""Tests de robustesse et de sensibilité (T20-T25).

Pour chaque scénario, le comportement attendu est explicite :
REJET, TRAITEMENT ou ALERTE (cf. reports/03_robustesse_non_regression.md).
Les perturbations sont propres à chaque type de variable :
* V1..V28 : composantes ACP standardisables -> bruit gaussien relatif à l'écart-type ;
* Amount  : perturbation multiplicative (arrondi, frais, conversion de devise) ;
* Time    : décalage de quelques secondes (horodatage) ;
On mesure la sensibilité, on n'impose pas une invariance universelle.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.extra.pandas import column, data_frames

from app.monitoring import alert_rate_check, drift_report, psi
from app.prediction import predict_fraud
from app.preprocessing import FEATURE_COLUMNS, PCA_COLUMNS, DataValidationError


# --------------------------------------------------------------------------- #
# Scénarios obligatoires 1, 2, 6 : REJET                                      #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("cols", [["V1"], ["V4", "V14", "Amount"]])
def test_missing_values_are_rejected(bundle, X_test, cols):
    X = X_test.copy()
    X.loc[X.index[:3], cols] = np.nan
    with pytest.raises(DataValidationError, match="manquantes"):
        predict_fraud(bundle, X)


def test_infinite_values_are_rejected(bundle, X_test):
    X = X_test.copy()
    X.loc[X.index[0], "Amount"] = np.inf
    with pytest.raises(DataValidationError, match="infinies"):
        predict_fraud(bundle, X)


def test_non_numeric_values_are_rejected(bundle, X_test):
    X = X_test.astype({"V10": object})
    X.loc[X.index[0], "V10"] = "N/A"
    with pytest.raises(DataValidationError, match="non numérique"):
        predict_fraud(bundle, X)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda X: X.drop(columns=["V28"]),
        lambda X: X.rename(columns={"Amount": "amount"}),
        lambda X: pd.concat([X, X[["V1"]]], axis=1),
        lambda X: X.iloc[:0],
    ],
    ids=["missing-column", "renamed-column", "duplicated-column", "empty"],
)
def test_invalid_schema_is_rejected(bundle, X_test, mutate):
    with pytest.raises(DataValidationError):
        predict_fraud(bundle, mutate(X_test))


# --------------------------------------------------------------------------- #
# Scénario 3 : montants extrêmes -> TRAITEMENT + ALERTE                       #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("amount", [1e5, 1e7, 1e12])
def test_extreme_amount_is_processed_with_alert(bundle, X_test, amount):
    X = X_test.iloc[:20].copy()
    X["Amount"] = amount
    out = predict_fraud(bundle, X)
    assert np.isfinite(out["fraud_probability"]).all()
    assert out["out_of_domain_alert"].all()


def test_in_domain_data_raises_few_alerts(bundle, splits):
    out = predict_fraud(bundle, splits["train"][0])
    assert not out["out_of_domain_alert"].any()


# --------------------------------------------------------------------------- #
# Scénario 4 : perturbations limitées -> SENSIBILITÉ MESURÉE                  #
# --------------------------------------------------------------------------- #
def perturb(X: pd.DataFrame, rng, v_sigma=0.0, amount_rel=0.0, time_shift=0.0):
    Xp = X.copy()
    if v_sigma:
        sd = X[PCA_COLUMNS].std().to_numpy()
        Xp[PCA_COLUMNS] = X[PCA_COLUMNS] + rng.normal(0, v_sigma, (len(X), 28)) * sd
    if amount_rel:
        Xp["Amount"] = X["Amount"] * (1 + rng.uniform(-amount_rel, amount_rel, len(X)))
    if time_shift:
        Xp["Time"] = (X["Time"] + rng.uniform(-time_shift, time_shift, len(X))).clip(0)
    return Xp


@pytest.mark.parametrize(
    "kwargs,max_mean_delta",
    [
        ({"v_sigma": 0.01}, 0.01),
        ({"amount_rel": 0.01}, 0.005),
        ({"time_shift": 60}, 0.005),
    ],
    ids=["V-noise-1%sd", "Amount-1%", "Time-60s"],
)
def test_small_perturbations_have_small_effect(bundle, X_test, rng, kwargs, max_mean_delta):
    base = predict_fraud(bundle, X_test)
    pert = predict_fraud(bundle, perturb(X_test, rng, **kwargs))
    delta = (pert["fraud_probability"] - base["fraud_probability"]).abs()
    assert delta.mean() < max_mean_delta
    flips = base["is_fraud"] != pert["is_fraud"]
    assert flips.mean() < 0.01


def test_class_flips_happen_only_near_the_decision_boundary(bundle, X_test, rng):
    base = predict_fraud(bundle, X_test)
    pert = predict_fraud(bundle, perturb(X_test, rng, v_sigma=0.05))
    flips = base["is_fraud"] != pert["is_fraud"]
    margin = (base["fraud_probability"] - bundle.threshold).abs()
    if flips.any():
        assert margin[flips].max() < margin.quantile(0.5)


def test_sensitivity_grows_with_perturbation_amplitude(bundle, X_test):
    deltas = []
    for sigma in (0.01, 0.05, 0.2):
        rng = np.random.default_rng(0)
        base = predict_fraud(bundle, X_test)["fraud_probability"]
        pert = predict_fraud(bundle, perturb(X_test, rng, v_sigma=sigma))["fraud_probability"]
        deltas.append((pert - base).abs().mean())
    assert deltas[0] < deltas[1] < deltas[2]


# --------------------------------------------------------------------------- #
# Scénario 5 : dérive de distribution -> ALERTE                               #
# --------------------------------------------------------------------------- #
def test_psi_is_near_zero_for_same_distribution(rng):
    a, b = rng.normal(size=5000), rng.normal(size=5000)
    assert psi(a, b) < 0.02


def test_psi_detects_shift(rng):
    a = rng.normal(size=5000)
    assert psi(a, a + 0.3) > 0.05
    assert psi(a, a + 1.0) > 0.25


def test_psi_handles_constant_reference():
    assert psi(np.zeros(10), np.zeros(5)) == 0.0
    assert psi(np.zeros(10), np.ones(5)) == float("inf")


def test_psi_rejects_empty():
    with pytest.raises(ValueError):
        psi(np.array([]), np.array([1.0]))


def test_drift_report_flags_only_shifted_features(splits):
    ref = splits["train"][0]
    cur = splits["test"][0].copy()
    cur["Amount"] = cur["Amount"] * 3  # ex. changement de devise
    cur["V14"] = cur["V14"] + 1.0
    report = drift_report(ref, cur).set_index("feature")
    assert report.loc["Amount", "level"] == "alerte"
    assert report.loc["V14", "level"] == "alerte"
    untouched = report.drop(index=["Amount", "V14"])
    assert (untouched["level"] != "alerte").all()


def test_alert_rate_monitoring():
    assert not alert_rate_check(20, 10_000, expected_rate=0.002)["alert"]
    assert alert_rate_check(80, 10_000, expected_rate=0.002)["alert"]
    assert alert_rate_check(0, 10_000, expected_rate=0.01)["alert"]
    with pytest.raises(ValueError):
        alert_rate_check(0, 0, 0.01)


# --------------------------------------------------------------------------- #
# Génération de cas (Hypothesis) : contrat de sortie sur entrées valides       #
# --------------------------------------------------------------------------- #
finite = st.floats(min_value=-50, max_value=50, allow_nan=False, allow_infinity=False)
valid_frames = data_frames(
    [column("Time", elements=st.floats(0, 200_000, allow_nan=False))]
    + [column(c, elements=finite) for c in PCA_COLUMNS]
    + [column("Amount", elements=st.floats(0, 1e9, allow_nan=False))],
).filter(lambda df: len(df) > 0)


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(X=valid_frames)
def test_any_valid_input_yields_valid_probabilities(bundle, X):
    out = predict_fraud(bundle, X[FEATURE_COLUMNS])
    p = out["fraud_probability"].to_numpy()
    assert len(out) == len(X)
    assert np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all()
