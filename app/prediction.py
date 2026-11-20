"""Prédiction, sauvegarde/rechargement et validation des entrées de type API."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import platform
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline

from app import __version__
from app.preprocessing import FEATURE_COLUMNS, DataValidationError, validate_data

logger = logging.getLogger(__name__)


class ModelLoadError(RuntimeError):
    """Le fichier de modèle est absent, corrompu ou incompatible."""


@dataclass
class ModelBundle:
    """Ce qui est déployé : pipeline + seuil + domaine d'entraînement + métadonnées."""

    pipeline: Pipeline
    threshold: float
    feature_min: dict[str, float]
    feature_max: dict[str, float]
    metadata: dict = field(default_factory=dict)

    @classmethod
    def from_training(
        cls, pipeline: Pipeline, threshold: float, X_train: pd.DataFrame, metadata: dict
    ) -> ModelBundle:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Le seuil doit être compris entre 0 et 1")
        meta = {
            "code_version": __version__,
            "sklearn_version": sklearn.__version__,
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "python_version": platform.python_version(),
            **metadata,
        }
        return cls(
            pipeline=pipeline,
            threshold=float(threshold),
            feature_min=X_train[FEATURE_COLUMNS].min().astype(float).to_dict(),
            feature_max=X_train[FEATURE_COLUMNS].max().astype(float).to_dict(),
            metadata=meta,
        )


# --------------------------------------------------------------------------- #
# Prédiction                                                                  #
# --------------------------------------------------------------------------- #
def predict_proba(model: Pipeline | ModelBundle, X: pd.DataFrame) -> np.ndarray:
    pipeline = model.pipeline if isinstance(model, ModelBundle) else model
    clean = validate_data(X, require_target=False)
    proba = pipeline.predict_proba(clean[FEATURE_COLUMNS])[:, 1]
    if not np.isfinite(proba).all():  # garde-fou du contrat de sortie
        raise RuntimeError("Le modèle a produit des probabilités non finies")
    return proba


def out_of_domain(bundle: ModelBundle, X: pd.DataFrame) -> np.ndarray:
    """Vrai si au moins une variable sort de l'intervalle observé à l'entraînement."""
    lo = pd.Series(bundle.feature_min)[FEATURE_COLUMNS].to_numpy()
    hi = pd.Series(bundle.feature_max)[FEATURE_COLUMNS].to_numpy()
    values = X[FEATURE_COLUMNS].to_numpy(dtype=float)
    return ((values < lo) | (values > hi)).any(axis=1)


def predict_fraud(
    model: Pipeline | ModelBundle, X: pd.DataFrame, threshold: float | None = None
) -> pd.DataFrame:
    """Renvoie, pour chaque transaction : probabilité, décision et alerte de domaine.

    Le seuil utilisé est celui du bundle (choisi sur la validation), sauf s'il
    est explicitement fourni. Les entrées invalides sont rejetées
    (DataValidationError) ; aucune imputation silencieuse.
    """
    if threshold is None:
        threshold = model.threshold if isinstance(model, ModelBundle) else 0.5
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Le seuil doit être compris entre 0 et 1")
    proba = predict_proba(model, X)
    result = pd.DataFrame(
        {"fraud_probability": proba, "is_fraud": (proba >= threshold).astype(int)},
        index=X.index,
    )
    if isinstance(model, ModelBundle):
        ood = out_of_domain(model, X)
        result["out_of_domain_alert"] = ood
        if ood.any():
            logger.warning("Transactions hors domaine d'entraînement : %d", int(ood.sum()))
    return result


# --------------------------------------------------------------------------- #
# Persistance                                                                 #
# --------------------------------------------------------------------------- #
def save_model(bundle: ModelBundle, path: str | Path) -> str:
    """Sauvegarde le bundle et renvoie son empreinte SHA-256."""
    if not isinstance(bundle, ModelBundle):
        raise TypeError("save_model attend un ModelBundle")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path, compress=3)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_model(path: str | Path, expected_sha256: str | None = None) -> ModelBundle:
    """Recharge un bundle. Toute anomalie produit une ModelLoadError explicite.

    Attention : joblib/pickle exécute du code au chargement. En production,
    `expected_sha256` (empreinte renvoyée par save_model et enregistrée dans le
    registre de modèles) doit être fourni : le fichier est vérifié AVANT d'être
    désérialisé.
    """
    path = Path(path)
    if not path.is_file():
        raise ModelLoadError(f"Fichier de modèle introuvable : {path}")
    if expected_sha256 is not None:
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected_sha256.lower():
            raise ModelLoadError(f"Empreinte SHA-256 invalide pour {path.name} : fichier refusé")
    try:
        bundle = joblib.load(path)
    except Exception as exc:  # fichier tronqué, format inconnu, etc.
        raise ModelLoadError(f"Fichier de modèle illisible ou corrompu : {path.name}") from exc
    if not isinstance(bundle, ModelBundle):
        raise ModelLoadError(f"Objet inattendu dans {path.name} : {type(bundle).__name__}")
    saved = bundle.metadata.get("sklearn_version")
    if saved and saved != sklearn.__version__:
        logger.warning(
            "Version scikit-learn différente (modèle : %s, environnement : %s)",
            saved,
            sklearn.__version__,
        )
    return bundle


# --------------------------------------------------------------------------- #
# Entrée de type API                                                          #
# --------------------------------------------------------------------------- #
MAX_BATCH = 1_000


def parse_api_payload(body: str | bytes) -> pd.DataFrame:
    """Valide un corps JSON {"transactions": [{...}, ...]} de manière stricte.

    Rejette : JSON invalide, structure inattendue, lot vide ou trop grand,
    champ manquant ou supplémentaire, type non numérique (y compris booléen,
    null ou chaîne de caractères), valeur non finie.
    """
    if isinstance(body, bytes):
        try:
            body = body.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise DataValidationError("Corps de requête non UTF-8") from exc
    try:
        payload = json.loads(body, parse_constant=_reject_constant)
    except json.JSONDecodeError as exc:
        raise DataValidationError("JSON invalide") from exc

    if not isinstance(payload, dict) or set(payload) != {"transactions"}:
        raise DataValidationError('Structure attendue : {"transactions": [...]}')
    rows = payload["transactions"]
    if not isinstance(rows, list) or not rows:
        raise DataValidationError("'transactions' doit être une liste non vide")
    if len(rows) > MAX_BATCH:
        raise DataValidationError(f"Lot trop volumineux (max {MAX_BATCH})")

    expected = set(FEATURE_COLUMNS)
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            raise DataValidationError(f"Transaction {i} : objet JSON attendu")
        keys = set(row)
        if keys != expected:
            raise DataValidationError(
                f"Transaction {i} : champs manquants {sorted(expected - keys)}, "
                f"champs non autorisés {sorted(keys - expected)}"
            )
        bad = [
            k
            for k, v in row.items()
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
        ]
        if bad:
            raise DataValidationError(f"Transaction {i} : valeurs non numériques pour {bad}")
    return pd.DataFrame(rows, columns=FEATURE_COLUMNS, dtype="float64")


def _reject_constant(name: str):
    raise DataValidationError(f"Constante JSON non autorisée : {name}")


def handle_request(body: str | bytes, bundle: ModelBundle) -> tuple[int, dict]:
    """Traitement complet d'une requête (sans la couche HTTP).

    Renvoie (code HTTP, réponse JSON). Les logs ne contiennent que des
    compteurs, jamais les valeurs des transactions.
    """
    try:
        X = parse_api_payload(body)
        result = predict_fraud(bundle, X)
    except DataValidationError as exc:
        logger.info("Requête rejetée : %s", exc)
        return 400, {"error": str(exc)}
    logger.info(
        "Requête traitée : %d transactions, %d alertes",
        len(result),
        int(result["is_fraud"].sum()),
    )
    return 200, {
        "predictions": [
            {
                "fraud_probability": round(float(p), 6),
                "is_fraud": bool(f),
                "out_of_domain_alert": bool(o),
            }
            for p, f, o in result[
                ["fraud_probability", "is_fraud", "out_of_domain_alert"]
            ].itertuples(index=False)
        ],
        "threshold": bundle.threshold,
        "model_version": bundle.metadata.get("code_version"),
    }
