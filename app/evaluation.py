"""Métriques, intervalles de confiance, courbes et comparaison à la baseline."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from app import config


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def evaluate_model(y_true, y_pred, y_proba=None) -> dict:
    """Calcule toutes les métriques obligatoires.

    ROC-AUC et Average Precision valent None si une seule classe est présente
    (elles ne sont pas définies) au lieu de lever une exception.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Dimensions incohérentes : {y_true.shape} vs {y_pred.shape}")
    if y_true.size == 0:
        raise ValueError("Aucune observation à évaluer")
    for name, arr in (("y_true", y_true), ("y_pred", y_pred)):
        if not np.isin(arr, [0, 1]).all():
            raise ValueError(f"{name} doit ne contenir que 0 et 1")

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    tn, fp, fn, tp = int(tn), int(fp), int(fn), int(tp)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    metrics = {
        "n": int(y_true.size),
        "n_fraud": int(y_true.sum()),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": _safe_div(tp + tn, y_true.size),
        "precision": precision,
        "recall": recall,
        "f1": _safe_div(2 * precision * recall, precision + recall),
        "fpr": _safe_div(fp, fp + tn),
        "fnr": _safe_div(fn, fn + tp),
        "alert_rate": _safe_div(tp + fp, y_true.size),
        "roc_auc": None,
        "average_precision": None,
    }
    if y_proba is not None:
        y_proba = np.asarray(y_proba, dtype=float)
        if y_proba.shape != y_true.shape:
            raise ValueError("y_proba doit avoir la même taille que y_true")
        if not (np.isfinite(y_proba).all() and (y_proba >= 0).all() and (y_proba <= 1).all()):
            raise ValueError("y_proba doit contenir des probabilités finies dans [0, 1]")
        if len(np.unique(y_true)) == 2:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
            metrics["average_precision"] = float(average_precision_score(y_true, y_proba))
    return metrics


def bootstrap_ci(
    y_true,
    y_proba,
    threshold: float,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = config.SEED,
    metrics: tuple[str, ...] = ("precision", "recall", "fpr", "f1", "average_precision"),
) -> dict[str, tuple[float, float]]:
    """Intervalles de confiance percentiles par bootstrap stratifié."""
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba, dtype=float)
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y_true == 1), np.flatnonzero(y_true == 0)
    samples: dict[str, list[float]] = {m: [] for m in metrics}
    for _ in range(n_boot):
        idx = np.concatenate(
            [rng.choice(pos, len(pos), replace=True), rng.choice(neg, len(neg), replace=True)]
        )
        m = evaluate_model(y_true[idx], (y_proba[idx] >= threshold).astype(int), y_proba[idx])
        for k in metrics:
            if m[k] is not None:
                samples[k].append(m[k])
    return {
        k: (
            float(np.quantile(v, alpha / 2)),
            float(np.quantile(v, 1 - alpha / 2)),
        )
        for k, v in samples.items()
        if v
    }


def curves(y_true, y_proba) -> dict:
    fpr, tpr, roc_thr = roc_curve(y_true, y_proba)
    prec, rec, pr_thr = precision_recall_curve(y_true, y_proba)
    return {"roc": (fpr, tpr, roc_thr), "pr": (prec, rec, pr_thr)}


def check_acceptance(metrics: dict, criteria: dict | None = None) -> dict[str, bool]:
    """Vérifie chaque critère d'acceptation et renvoie le détail."""
    c = criteria or config.ACCEPTANCE
    return {
        "roc_auc": (metrics["roc_auc"] or 0) >= c["roc_auc_min"],
        "average_precision": (metrics["average_precision"] or 0) >= c["average_precision_min"],
        "recall": metrics["recall"] >= c["recall_min"],
        "precision": metrics["precision"] >= c["precision_min"],
        "fpr": metrics["fpr"] <= c["fpr_max"],
    }


HIGHER_IS_BETTER = {"average_precision", "roc_auc", "recall", "precision"}
LOWER_IS_BETTER = {"fpr"}


def compare_to_baseline(current: dict, baseline: dict, tolerance: dict | None = None) -> dict:
    """Compare une nouvelle version à la référence figée.

    Décision :
    * « rejeter »   si au moins une métrique se dégrade au-delà de la tolérance ;
    * « investiguer » si une métrique se dégrade de plus de la moitié de la
      tolérance (signal faible) ou si elle s'améliore de plus de 3 fois la
      tolérance (amélioration suspecte : fuite de données ?) ;
    * « accepter »  sinon.
    """
    tolerance = tolerance or config.REGRESSION_TOLERANCE
    details, decision = {}, "accepter"
    rank = {"accepter": 0, "investiguer": 1, "rejeter": 2}
    for metric, tol in tolerance.items():
        b, c = baseline[metric], current[metric]
        if b is None or c is None:
            status = "investiguer"
            degradation = None
        else:
            degradation = (b - c) if metric in HIGHER_IS_BETTER else (c - b)
            if degradation > tol:
                status = "rejeter"
            elif degradation > tol / 2 or -degradation > 3 * tol:
                status = "investiguer"
            else:
                status = "accepter"
        details[metric] = {
            "baseline": b,
            "current": c,
            "degradation": degradation,
            "tolerance": tol,
            "status": status,
        }
        if rank[status] > rank[decision]:
            decision = status
    return {"decision": decision, "details": details}
