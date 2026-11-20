"""Indicateurs de surveillance : dérive des données et taux d'alerte."""

from __future__ import annotations

import numpy as np
import pandas as pd

from app import config
from app.preprocessing import FEATURE_COLUMNS


def psi(reference: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
    """Population Stability Index sur des classes définies par quantiles de référence."""
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    if reference.size == 0 or current.size == 0:
        raise ValueError("Échantillons vides")
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, n_bins + 1)))
    if edges.size < 2:  # variable constante dans la référence
        return 0.0 if np.allclose(current, reference[0]) else float("inf")
    edges[0], edges[-1] = -np.inf, np.inf
    ref_pct = np.histogram(reference, edges)[0] / reference.size
    cur_pct = np.histogram(current, edges)[0] / current.size
    eps = 1e-6
    ref_pct, cur_pct = np.clip(ref_pct, eps, None), np.clip(cur_pct, eps, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def drift_report(reference: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    """PSI par variable et niveau d'alerte (ok / surveiller / alerte)."""
    rows = []
    for col in FEATURE_COLUMNS:
        value = psi(reference[col].to_numpy(), current[col].to_numpy())
        level = (
            "alerte"
            if value >= config.PSI_ALERT
            else "surveiller"
            if value >= config.PSI_WARNING
            else "ok"
        )
        rows.append({"feature": col, "psi": value, "level": level})
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


def alert_rate_check(n_alerts: int, n_total: int, expected_rate: float, z: float = 3.0) -> dict:
    """Contrôle du taux de transactions signalées (disponible sans label).

    Alerte si le taux observé sort de l'intervalle binomial à ±z écarts-types
    autour du taux attendu (mesuré sur la validation).
    """
    if n_total <= 0:
        raise ValueError("n_total doit être > 0")
    rate = n_alerts / n_total
    sd = np.sqrt(expected_rate * (1 - expected_rate) / n_total)
    low, high = max(0.0, expected_rate - z * sd), expected_rate + z * sd
    return {
        "rate": rate,
        "expected": expected_rate,
        "low": low,
        "high": high,
        "alert": not (low <= rate <= high),
    }
