"""Tests unitaires du contrat de données et du prétraitement (T01-T03, T08-T10)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from app.preprocessing import (
    FEATURE_COLUMNS,
    REQUIRED_COLUMNS,
    TARGET_COLUMN,
    DataValidationError,
    build_preprocessor,
    load_data,
    preprocess_data,
    split_data,
    validate_data,
)


# --------------------------------------------------------------------------- #
# T01 - colonnes requises                                                     #
# --------------------------------------------------------------------------- #
def test_valid_data_passes_and_keeps_contract_order(valid_df):
    shuffled = valid_df[list(reversed(valid_df.columns))]
    out = validate_data(shuffled)
    assert list(out.columns) == REQUIRED_COLUMNS
    assert len(out) == len(valid_df)
    pd.testing.assert_frame_equal(out, valid_df.astype({c: "float64" for c in FEATURE_COLUMNS}))


@pytest.mark.parametrize("column", ["Time", "V1", "V14", "V28", "Amount", "Class"])
def test_missing_required_column_is_rejected(valid_df, column):
    with pytest.raises(DataValidationError, match=rf"absentes.*'{column}'"):
        validate_data(valid_df.drop(columns=column))


def test_missing_target_is_allowed_for_inference(valid_df):
    out = validate_data(valid_df.drop(columns=TARGET_COLUMN), require_target=False)
    assert list(out.columns) == FEATURE_COLUMNS


def test_duplicated_column_is_rejected(valid_df):
    dup = pd.concat([valid_df, valid_df[["Amount"]]], axis=1)
    with pytest.raises(DataValidationError, match=r"dupliquées.*Amount"):
        validate_data(dup)


def test_extra_column_is_dropped_with_warning_without_values(valid_df, caplog):
    df = valid_df.assign(card_number="4970-1234-5678-9012")
    with caplog.at_level("WARNING"):
        out = validate_data(df)
    assert "card_number" not in out.columns
    assert "card_number" in caplog.text
    assert "4970-1234" not in caplog.text  # le nom de colonne, jamais la valeur


# --------------------------------------------------------------------------- #
# T02 - types                                                                 #
# --------------------------------------------------------------------------- #
def test_types_are_normalised(valid_df):
    df = valid_df.astype({"V1": "float32", "Time": "int64"})
    out = validate_data(df)
    assert (out[FEATURE_COLUMNS].dtypes == "float64").all()
    assert out[TARGET_COLUMN].dtype == "int64"


@pytest.mark.parametrize(
    "bad_value", ["12.5", "abc", None], ids=["numeric-string", "text", "object-none"]
)
def test_non_numeric_type_is_rejected(valid_df, bad_value):
    df = valid_df.astype({"Amount": object})
    df.loc[0, "Amount"] = bad_value
    with pytest.raises(DataValidationError, match=r"non numérique.*Amount"):
        validate_data(df)


def test_boolean_column_is_rejected(valid_df):
    df = valid_df.assign(V3=valid_df["V3"] > 0)
    with pytest.raises(DataValidationError, match=r"non numérique.*V3"):
        validate_data(df)


def test_non_dataframe_input_is_rejected(valid_df):
    with pytest.raises(DataValidationError, match="DataFrame"):
        validate_data(valid_df.to_numpy())


# --------------------------------------------------------------------------- #
# T03 - valeurs non finies, vides, bornes                                     #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("column", ["V7", "Amount", "Time"])
def test_missing_values_are_rejected_with_count(valid_df, column):
    df = valid_df.copy()
    df.loc[[3, 10], column] = np.nan
    with pytest.raises(DataValidationError, match=rf"manquantes \(2 lignes\).*{column}"):
        validate_data(df)


@pytest.mark.parametrize("value", [np.inf, -np.inf])
def test_infinite_values_are_rejected(valid_df, value):
    df = valid_df.copy()
    df.loc[0, "V2"] = value
    with pytest.raises(DataValidationError, match=r"infinies.*V2"):
        validate_data(df)


def test_empty_dataframe_is_rejected():
    with pytest.raises(DataValidationError, match="vide"):
        validate_data(pd.DataFrame(columns=REQUIRED_COLUMNS))
    with pytest.raises(DataValidationError, match="vide"):
        validate_data(pd.DataFrame())


@pytest.mark.parametrize("column", ["Amount", "Time"])
def test_negative_amount_or_time_is_rejected(valid_df, column):
    df = valid_df.copy()
    df.loc[0, column] = -0.01
    with pytest.raises(DataValidationError, match=rf"négatives.*{column}"):
        validate_data(df)


def test_zero_amount_is_a_valid_boundary(valid_df):
    df = valid_df.copy()
    df.loc[0, "Amount"] = 0.0
    assert validate_data(df).loc[0, "Amount"] == 0.0


# --------------------------------------------------------------------------- #
# T09 - cible                                                                 #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad_target", [2, -1, 0.5])
def test_target_outside_zero_one_is_rejected(valid_df, bad_target):
    df = valid_df.astype({"Class": float})
    df.loc[0, "Class"] = bad_target
    with pytest.raises(DataValidationError, match="0 ou 1"):
        validate_data(df)


def test_target_with_nan_is_rejected(valid_df):
    df = valid_df.astype({"Class": float})
    df.loc[0, "Class"] = np.nan
    with pytest.raises(DataValidationError, match="cible"):
        validate_data(df)


def test_float_target_zero_one_is_accepted_and_cast(valid_df):
    out = validate_data(valid_df.astype({"Class": float}))
    assert out["Class"].dtype == "int64"
    assert set(out["Class"].unique()) <= {0, 1}


# --------------------------------------------------------------------------- #
# Une seule observation et dimensions                                         #
# --------------------------------------------------------------------------- #
def test_single_observation_is_processed(valid_df):
    X, y = preprocess_data(valid_df.iloc[[0]])
    assert X.shape == (1, len(FEATURE_COLUMNS))
    assert y.shape == (1,)


def test_preprocess_returns_consistent_dimensions(valid_df):
    X, y = preprocess_data(valid_df)
    assert X.shape == (len(valid_df), 30)
    assert len(y) == len(X)
    assert list(X.columns) == FEATURE_COLUMNS
    assert X.index.equals(y.index)


def test_preprocessor_output_is_finite_and_has_expected_width(valid_df):
    X, _ = preprocess_data(valid_df)
    Z = build_preprocessor().fit_transform(X)
    assert Z.shape == (len(X), 30)
    assert np.isfinite(Z).all()


def test_scaler_statistics_on_training_data(valid_df):
    """Exemple du sujet (2.2), appliqué au préprocesseur réel."""
    X, _ = preprocess_data(valid_df)
    Z = build_preprocessor().fit_transform(X)
    assert np.allclose(Z.mean(axis=0), 0.0, atol=1e-10)
    assert np.allclose(Z.std(axis=0), 1.0)


def test_amount_log_transform_tames_extreme_values(valid_df):
    X, _ = preprocess_data(valid_df)
    prep = build_preprocessor().fit(X)
    extreme = X.iloc[[0]].assign(Amount=1e12)
    z_amount = prep.transform(extreme)[0, 0]
    assert np.isfinite(z_amount)
    assert z_amount < 50  # log1p(1e12) ≈ 27.6 : reste borné après standardisation


# --------------------------------------------------------------------------- #
# Partition (T05, T10)                                                        #
# --------------------------------------------------------------------------- #
def test_split_is_stratified_disjoint_and_complete(dataset):
    train, val, test = split_data(dataset)
    assert len(train) + len(val) + len(test) == len(dataset.drop_duplicates())
    base_rate = dataset["Class"].mean()
    for part in (train, val, test):
        assert abs(part["Class"].mean() - base_rate) < 0.005
    keys = [set(map(tuple, p[FEATURE_COLUMNS].to_numpy())) for p in (train, val, test)]
    assert not keys[0] & keys[1] and not keys[0] & keys[2] and not keys[1] & keys[2]


def test_split_removes_exact_duplicates_before_partition(dataset):
    with_dupes = pd.concat([dataset, dataset.iloc[:300]], ignore_index=True)
    train, val, test = split_data(with_dupes)
    rows = pd.concat([train, val, test])
    assert not rows.duplicated().any()


def test_split_is_deterministic_for_a_given_seed(dataset):
    a = split_data(dataset, seed=7)
    b = split_data(dataset, seed=7)
    c = split_data(dataset, seed=8)
    for x, y in zip(a, b, strict=True):
        pd.testing.assert_frame_equal(x, y)
    assert not a[2].equals(c[2])


@pytest.mark.parametrize("val_size,test_size", [(0, 0.2), (0.5, 0.5), (0.2, 1.2)])
def test_split_rejects_invalid_proportions(dataset, val_size, test_size):
    with pytest.raises(ValueError):
        split_data(dataset, val_size=val_size, test_size=test_size)


# --------------------------------------------------------------------------- #
# Chargement de fichiers                                                      #
# --------------------------------------------------------------------------- #
def test_load_data_roundtrip(tmp_path, valid_df):
    path = tmp_path / "sample.csv"
    valid_df.to_csv(path, index=False)
    out = load_data(path)
    assert out.shape == valid_df.shape


def test_load_data_missing_file_is_explicit(tmp_path):
    with pytest.raises(FileNotFoundError, match="introuvable"):
        load_data(tmp_path / "absent.csv")


def test_load_data_empty_file_is_rejected(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("")
    with pytest.raises(DataValidationError, match="vide"):
        load_data(path)


def test_load_data_directory_is_rejected(tmp_path):
    with pytest.raises(DataValidationError, match="pas un fichier"):
        load_data(tmp_path)


def test_load_data_detects_duplicated_header(tmp_path, valid_df):
    """pandas renomme Amount en Amount.1 : le doublon doit être détecté avant."""
    path = tmp_path / "dup.csv"
    pd.concat([valid_df, valid_df[["Amount"]]], axis=1).to_csv(path, index=False)
    with pytest.raises(DataValidationError, match=r"dupliquées.*Amount"):
        load_data(path)


def test_load_data_unreadable_file(tmp_path):
    path = tmp_path / "binary.csv"
    path.write_bytes(b"Time,V1\n\xff\xfe\x00\x81")
    with pytest.raises(DataValidationError):
        load_data(path)


# --------------------------------------------------------------------------- #
# Propriétés (Hypothesis)                                                     #
# --------------------------------------------------------------------------- #
@settings(max_examples=50, deadline=None)
@given(
    row=st.integers(min_value=0, max_value=199),
    col=st.sampled_from(FEATURE_COLUMNS),
    bad=st.sampled_from([np.nan, np.inf, -np.inf]),
)
def test_any_single_non_finite_value_is_rejected(row, col, bad):
    from app.synthetic import make_synthetic_data

    df = make_synthetic_data(n=200, fraud_rate=0.05, seed=1)
    df.loc[row, col] = bad
    with pytest.raises(DataValidationError, match=col):
        validate_data(df)
