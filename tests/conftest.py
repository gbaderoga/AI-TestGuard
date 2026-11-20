"""Fixtures partagées. Les tests rapides utilisent des données synthétiques
au schéma Kaggle : la CI ne dépend pas du jeu de données réel."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.prediction import ModelBundle
from app.preprocessing import preprocess_data, split_data
from app.synthetic import make_synthetic_data
from app.training import select_threshold, train_model


@pytest.fixture
def valid_df() -> pd.DataFrame:
    return make_synthetic_data(n=200, fraud_rate=0.05, seed=1)


@pytest.fixture(scope="session")
def dataset() -> pd.DataFrame:
    return make_synthetic_data(n=4000, fraud_rate=0.02, seed=0)


@pytest.fixture(scope="session")
def splits(dataset):
    train, val, test = split_data(dataset)
    return {
        name: preprocess_data(part)
        for name, part in (("train", train), ("val", val), ("test", test))
    }


@pytest.fixture(scope="session")
def bundle(splits) -> ModelBundle:
    X_train, y_train = splits["train"]
    X_val, y_val = splits["val"]
    pipe = train_model(X_train, y_train, "logreg", {"C": 0.1})
    threshold = select_threshold(y_val, pipe.predict_proba(X_val)[:, 1])
    return ModelBundle.from_training(pipe, threshold, X_train, {"seed": 42})


@pytest.fixture(scope="session")
def X_test(splits) -> pd.DataFrame:
    return splits["test"][0]


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(123)
