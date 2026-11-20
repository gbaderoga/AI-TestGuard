"""Données synthétiques au schéma du jeu Kaggle, pour la CI et les tests.

Ces données ne sont PAS des transactions réelles et ne servent qu'à exercer le
code : les métriques obtenues dessus n'ont aucune valeur de qualification.
Elles permettent d'exécuter la CI sans versionner le jeu de données Kaggle.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.preprocessing import PCA_COLUMNS, REQUIRED_COLUMNS

# Composantes décalées pour la classe positive (choix arbitraire, synthétique).
_SHIFTED = {"V4": 2.0, "V10": -2.5, "V11": 1.5, "V12": -2.5, "V14": -3.0, "V17": -2.0}


def make_synthetic_data(n: int = 4000, fraud_rate: float = 0.02, seed: int = 0) -> pd.DataFrame:
    if n < 2 or not 0 < fraud_rate < 1:
        raise ValueError("n >= 2 et 0 < fraud_rate < 1 requis")
    rng = np.random.default_rng(seed)
    n_fraud = max(1, round(n * fraud_rate))
    y = np.zeros(n, dtype=int)
    y[rng.choice(n, n_fraud, replace=False)] = 1

    V = rng.normal(0.0, 1.0, size=(n, len(PCA_COLUMNS)))
    for name, shift in _SHIFTED.items():
        j = PCA_COLUMNS.index(name)
        V[y == 1, j] += shift * rng.uniform(0.3, 1.2, size=n_fraud)

    df = pd.DataFrame(V, columns=PCA_COLUMNS)
    df.insert(0, "Time", np.sort(rng.uniform(0, 172_800, size=n)))
    amount = rng.lognormal(mean=3.0, sigma=1.4, size=n)
    amount[y == 1] = rng.lognormal(mean=3.5, sigma=1.8, size=n_fraud)
    df["Amount"] = np.round(amount, 2)
    df["Class"] = y
    return df[REQUIRED_COLUMNS]
