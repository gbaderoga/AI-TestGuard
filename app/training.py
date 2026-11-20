"""Entraînement, sélection de modèle et choix du seuil (validation uniquement)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from app import config
from app.preprocessing import FEATURE_COLUMNS, build_preprocessor

MODEL_NAMES = ("logreg", "tree")


def build_pipeline(
    model_name: str = "logreg", params: dict | None = None, seed: int = config.SEED
) -> Pipeline:
    """Pipeline complet : prétraitement + classifieur."""
    params = {"class_weight": "balanced", "random_state": seed, **(params or {})}
    if model_name == "logreg":
        clf = LogisticRegression(max_iter=2000, **params)
    elif model_name == "tree":
        clf = DecisionTreeClassifier(**params)
    else:
        raise ValueError(f"Modèle inconnu : {model_name!r} (attendu : {MODEL_NAMES})")
    return Pipeline([("preprocess", build_preprocessor()), ("model", clf)])


def train_model(
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    model_name: str = "logreg",
    params: dict | None = None,
    seed: int = config.SEED,
) -> Pipeline:
    """Ajuste le pipeline sur les données d'entraînement fournies."""
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X doit être un pandas.DataFrame")
    missing = [c for c in FEATURE_COLUMNS if c not in X.columns]
    if missing:
        raise ValueError(f"Colonnes absentes de X : {missing}")
    y = np.asarray(y)
    if len(X) != len(y):
        raise ValueError(f"Dimensions incohérentes : X={len(X)} lignes, y={len(y)}")
    classes = set(np.unique(y).tolist())
    if classes != {0, 1}:
        raise ValueError(f"y doit contenir les deux classes 0 et 1, reçu : {sorted(classes)}")
    pipeline = build_pipeline(model_name, params, seed)
    pipeline.fit(X[FEATURE_COLUMNS], y)
    return pipeline


def expected_cost(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
    cost_fn: float = config.COST_FN,
    cost_fp: float = config.COST_FP,
) -> float:
    y_true = np.asarray(y_true)
    pred = np.asarray(y_proba) >= threshold
    fn = int(((y_true == 1) & ~pred).sum())
    fp = int(((y_true == 0) & pred).sum())
    return cost_fn * fn + cost_fp * fp


def select_threshold(
    y_val: np.ndarray,
    proba_val: np.ndarray,
    cost_fn: float = config.COST_FN,
    cost_fp: float = config.COST_FP,
) -> float:
    """Seuil minimisant le coût attendu sur la VALIDATION.

    Candidats : les probabilités observées (et 0.5). En cas d'égalité,
    le seuil le plus élevé est retenu (moins d'alertes pour un coût égal).
    """
    y_val = np.asarray(y_val)
    proba_val = np.asarray(proba_val, dtype=float)
    if len(y_val) != len(proba_val) or len(y_val) == 0:
        raise ValueError("y_val et proba_val doivent avoir la même taille non nulle")
    candidates = np.unique(np.concatenate([proba_val, [0.5]]))
    # Tri par probabilité décroissante : coût cumulé en O(n log n).
    order = np.argsort(-proba_val, kind="mergesort")
    p_sorted, y_sorted = proba_val[order], y_val[order]
    tp_cum = np.cumsum(y_sorted == 1)
    fp_cum = np.cumsum(y_sorted == 0)
    n_pos = int((y_val == 1).sum())
    # Nombre de cas avec proba >= t pour chaque candidat t.
    k = len(p_sorted) - np.searchsorted(p_sorted[::-1], candidates, side="left")
    tp = np.where(k > 0, tp_cum[np.maximum(k - 1, 0)], 0)
    fp = np.where(k > 0, fp_cum[np.maximum(k - 1, 0)], 0)
    costs = cost_fn * (n_pos - tp) + cost_fp * fp
    best = np.flatnonzero(costs == costs.min())
    return float(candidates[best].max())


@dataclass
class SelectionResult:
    model_name: str
    params: dict
    val_average_precision: float
    pipeline: Pipeline
    history: list[dict] = field(default_factory=list)


def select_model(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    model_name: str,
    grid: list[dict] | None = None,
    seed: int = config.SEED,
) -> SelectionResult:
    """Recherche sur grille, critère : Average Precision sur la validation."""
    grid = grid if grid is not None else config.PARAM_GRIDS[model_name]
    history, best = [], None
    for params in grid:
        pipe = train_model(X_train, y_train, model_name, params, seed)
        ap = float(average_precision_score(y_val, pipe.predict_proba(X_val)[:, 1]))
        history.append({"params": params, "val_average_precision": ap})
        if best is None or ap > best.val_average_precision:
            best = SelectionResult(model_name, params, ap, pipe)
    best.history = history
    return best
