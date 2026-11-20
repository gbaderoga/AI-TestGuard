"""Contrat de données, chargement, partition et prétraitement.

Politique de traitement des entrées invalides (documentée, cf. plan de test) :

* colonne requise absente, colonne dupliquée, DataFrame vide, type non numérique,
  valeur manquante (NaN), valeur infinie, montant ou temps négatif,
  cible hors {0, 1}                                   -> REJET (DataValidationError)
* colonne supplémentaire non prévue au contrat        -> TRAITEMENT : ignorée,
  avec un avertissement qui cite le nom de la colonne mais jamais ses valeurs
* montant très élevé (valeur finie)                   -> TRAITEMENT : log1p dans le
  pipeline ; une ALERTE « hors domaine » est levée à la prédiction
  (cf. app/prediction.py)

Le pipeline ne corrige donc jamais silencieusement une valeur manquante ou infinie :
dans un contexte de détection de fraude, imputer une valeur reviendrait à fabriquer
une transaction différente de celle qui a été reçue.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from app import config

logger = logging.getLogger(__name__)

PCA_COLUMNS: list[str] = [f"V{i}" for i in range(1, 29)]
FEATURE_COLUMNS: list[str] = ["Time", *PCA_COLUMNS, "Amount"]
TARGET_COLUMN = "Class"
REQUIRED_COLUMNS: list[str] = [*FEATURE_COLUMNS, TARGET_COLUMN]
NON_NEGATIVE_COLUMNS = ("Time", "Amount")


class DataValidationError(ValueError):
    """Entrée qui ne respecte pas le contrat de données."""


# --------------------------------------------------------------------------- #
# Validation                                                                  #
# --------------------------------------------------------------------------- #
def validate_data(df: pd.DataFrame, require_target: bool = True) -> pd.DataFrame:
    """Vérifie le contrat de données et renvoie une copie réordonnée et typée.

    Les messages d'erreur citent les colonnes et le nombre de lignes en cause,
    jamais les valeurs (pour ne pas exposer de données de transaction dans les logs).
    """
    if not isinstance(df, pd.DataFrame):
        raise DataValidationError(f"Entrée attendue : pandas.DataFrame, reçu : {type(df).__name__}")
    if df.empty or len(df.columns) == 0:
        raise DataValidationError("Le jeu de données est vide.")

    duplicated = sorted(set(df.columns[df.columns.duplicated()].astype(str)))
    if duplicated:
        raise DataValidationError(f"Colonnes dupliquées : {duplicated}")

    expected = REQUIRED_COLUMNS if require_target else FEATURE_COLUMNS
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise DataValidationError(f"Colonnes requises absentes : {missing}")

    extra = [c for c in df.columns if c not in REQUIRED_COLUMNS]
    if extra:
        logger.warning("Colonnes hors contrat ignorées : %s", extra)

    out = df[expected].copy()

    wrong_type = [
        c
        for c in expected
        if pd.api.types.is_bool_dtype(out[c]) or not pd.api.types.is_numeric_dtype(out[c])
    ]
    if wrong_type:
        raise DataValidationError(f"Type non numérique pour : {wrong_type}")

    out[FEATURE_COLUMNS] = out[FEATURE_COLUMNS].astype("float64")
    values = out[FEATURE_COLUMNS].to_numpy()

    nan_cols = [
        c for c, bad in zip(FEATURE_COLUMNS, np.isnan(values).any(axis=0), strict=True) if bad
    ]
    if nan_cols:
        n = int(np.isnan(values).any(axis=1).sum())
        raise DataValidationError(f"Valeurs manquantes ({n} lignes) dans : {nan_cols}")

    inf_cols = [
        c for c, bad in zip(FEATURE_COLUMNS, np.isinf(values).any(axis=0), strict=True) if bad
    ]
    if inf_cols:
        n = int(np.isinf(values).any(axis=1).sum())
        raise DataValidationError(f"Valeurs infinies ({n} lignes) dans : {inf_cols}")

    negative = [c for c in NON_NEGATIVE_COLUMNS if (out[c] < 0).any()]
    if negative:
        raise DataValidationError(f"Valeurs négatives interdites dans : {negative}")

    if require_target:
        target = out[TARGET_COLUMN]
        if target.isna().any():
            raise DataValidationError("La cible contient des valeurs manquantes.")
        invalid = ~target.isin([0, 1])
        if invalid.any():
            raise DataValidationError(
                f"La cible doit valoir 0 ou 1 ({int(invalid.sum())} valeurs invalides)."
            )
        out[TARGET_COLUMN] = target.astype("int64")

    return out


def load_data(path: str | Path, require_target: bool = True) -> pd.DataFrame:
    """Charge un CSV et applique le contrat. Les erreurs sont explicites."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Fichier de données introuvable : {path}")
    if not path.is_file():
        raise DataValidationError(f"Le chemin n'est pas un fichier : {path}")
    if path.stat().st_size == 0:
        raise DataValidationError(f"Fichier vide : {path}")

    # pandas renomme silencieusement les en-têtes dupliqués (Amount -> Amount.1) :
    # on lit donc l'en-tête brut pour détecter les doublons avant pandas.
    try:
        with path.open(newline="", encoding="utf-8") as fh:
            header = next(csv.reader(fh), [])
        dupes = sorted({c for c in header if header.count(c) > 1})
        if dupes:
            raise DataValidationError(f"Colonnes dupliquées dans le fichier : {dupes}")
        df = pd.read_csv(path)
    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise DataValidationError(f"Fichier CSV illisible : {path.name}") from exc
    return validate_data(df, require_target=require_target)


# --------------------------------------------------------------------------- #
# Partition                                                                   #
# --------------------------------------------------------------------------- #
def split_data(
    df: pd.DataFrame,
    val_size: float = config.VAL_SIZE,
    test_size: float = config.TEST_SIZE,
    seed: int = config.SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Partition stratifiée train / validation / test, AVANT tout ajustement.

    Les doublons exacts de transactions sont regroupés du même côté de la
    partition (on déduplique d'abord), sinon une même transaction pourrait se
    trouver à la fois en entraînement et en test (fuite de données).
    """
    if not 0 < val_size < 1 or not 0 < test_size < 1 or val_size + test_size >= 1:
        raise ValueError("val_size et test_size doivent être dans ]0,1[ et de somme < 1")
    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    if len(df) < n_before:
        logger.info("Doublons exacts supprimés avant partition : %d", n_before - len(df))

    train_val, test = train_test_split(
        df, test_size=test_size, stratify=df[TARGET_COLUMN], random_state=seed
    )
    relative_val = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val,
        test_size=relative_val,
        stratify=train_val[TARGET_COLUMN],
        random_state=seed,
    )
    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


# --------------------------------------------------------------------------- #
# Prétraitement                                                               #
# --------------------------------------------------------------------------- #
def preprocess_data(
    df: pd.DataFrame, require_target: bool = True
) -> tuple[pd.DataFrame, pd.Series | None]:
    """Valide puis sépare variables explicatives et cible.

    Aucune transformation apprise n'est faite ici : la normalisation est ajustée
    dans le Pipeline scikit-learn, uniquement sur les données d'entraînement.
    """
    clean = validate_data(df, require_target=require_target)
    X = clean[FEATURE_COLUMNS]
    y = clean[TARGET_COLUMN] if require_target else None
    return X, y


def _log1p_amount(x):
    return np.log1p(x)


def build_preprocessor() -> ColumnTransformer:
    """Transformations apprises, regroupées pour être ajustées sur le train seul.

    * Amount : log1p (montants très asymétriques) puis standardisation ;
    * Time et V1..V28 : standardisation (les V sont déjà issues d'une ACP,
      elles sont recentrées mais non réinterprétées).
    """
    amount = Pipeline(
        [
            ("log1p", FunctionTransformer(_log1p_amount, feature_names_out="one-to-one")),
            ("scale", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        [
            ("amount", amount, ["Amount"]),
            ("other", StandardScaler(), ["Time", *PCA_COLUMNS]),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
