"""Tests de robustesse et de sensibilité sur le jeu réel (Partie V).

Usage : python scripts/run_robustness.py --data data/creditcard.csv --out reports
Prérequis : scripts/run_experiments.py (modèle dans models/model.joblib).
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from _common import INK_2, ROOT, SERIES, dump_json, load_splits, plt

from app.evaluation import evaluate_model
from app.monitoring import drift_report
from app.prediction import load_model, predict_fraud
from app.preprocessing import PCA_COLUMNS, DataValidationError, preprocess_data
from app.training import select_threshold, train_model

SEED = 2024


def score(bundle, X, y, base=None):
    out = predict_fraud(bundle, X)
    m = evaluate_model(y, out["is_fraud"], out["fraud_probability"])
    row = {
        "recall": m["recall"],
        "precision": m["precision"],
        "fpr": m["fpr"],
        "average_precision": m["average_precision"],
        "alerts": int(out["is_fraud"].sum()),
        "ood_alerts": int(out["out_of_domain_alert"].sum()),
    }
    if base is not None:
        d = (out["fraud_probability"] - base["fraud_probability"]).abs()
        flips = out["is_fraud"] != base["is_fraud"]
        row.update(
            mean_abs_dproba=float(d.mean()),
            p99_abs_dproba=float(d.quantile(0.99)),
            max_abs_dproba=float(d.max()),
            class_flips=int(flips.sum()),
            flips_fraud_to_normal=int((flips & (base["is_fraud"] == 1)).sum()),
            flips_normal_to_fraud=int((flips & (base["is_fraud"] == 0)).sum()),
        )
    return row, out


def try_reject(bundle, X):
    try:
        predict_fraud(bundle, X)
        return "accepté (ANOMALIE)", None
    except DataValidationError as exc:
        return "rejeté", str(exc)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/creditcard.csv")
    ap.add_argument("--out", default="reports")
    ap.add_argument("--model-path", default="models/model.joblib")
    args = ap.parse_args()
    out_dir = ROOT / args.out
    rng = np.random.default_rng(SEED)

    df, splits = load_splits(args.data)
    X_tr, _ = splits["train"]
    X_te, y_te = splits["test"]
    bundle = load_model(ROOT / args.model_path)
    base_row, base = score(bundle, X_te, y_te)
    rows = [{"scenario": "Référence (test non perturbé)", "attendu": "—", **base_row}]

    # ---- Entrées invalides : REJET attendu --------------------------------
    rejections = []
    idx = rng.choice(len(X_te), size=int(0.01 * len(X_te)), replace=False)

    def mutated(fn):
        X = X_te.copy()
        fn(X)
        return X

    cases = {
        "NaN dans V14 (1 % des lignes)": lambda X: X.iloc.__setitem__(
            (idx, X.columns.get_loc("V14")), np.nan
        ),
        "NaN dans Amount, V1, V2 (1 %)": lambda X: [
            X.iloc.__setitem__((idx, X.columns.get_loc(c)), np.nan) for c in ("Amount", "V1", "V2")
        ],
        "+inf dans Amount (1 %)": lambda X: X.iloc.__setitem__(
            (idx, X.columns.get_loc("Amount")), np.inf
        ),
        "-inf dans V3 (1 %)": lambda X: X.iloc.__setitem__((idx, X.columns.get_loc("V3")), -np.inf),
        "Montant négatif (1 %)": lambda X: X.iloc.__setitem__(
            (idx, X.columns.get_loc("Amount")), -10.0
        ),
    }
    for name, fn in cases.items():
        status, msg = try_reject(bundle, mutated(fn))
        rejections.append(
            {
                "scenario": name,
                "attendu": "rejet",
                "obtenu": status,
                "message": msg,
                "lignes_invalides": len(idx),
            }
        )

    text = X_te.astype({"V10": object})
    text.iloc[idx, text.columns.get_loc("V10")] = "N/A"
    status, msg = try_reject(bundle, text)
    rejections.append(
        {
            "scenario": "Texte « N/A » dans V10 (1 %)",
            "attendu": "rejet",
            "obtenu": status,
            "message": msg,
            "lignes_invalides": len(idx),
        }
    )
    for name, X in {
        "Colonne V28 absente": X_te.drop(columns="V28"),
        "Colonne renommée (Amount -> amount)": X_te.rename(columns={"Amount": "amount"}),
        "Colonne V1 dupliquée": pd.concat([X_te, X_te[["V1"]]], axis=1),
        "Lot vide": X_te.iloc[:0],
    }.items():
        status, msg = try_reject(bundle, X)
        rejections.append(
            {
                "scenario": name,
                "attendu": "rejet",
                "obtenu": status,
                "message": msg,
                "lignes_invalides": len(X_te),
            }
        )
    extra = X_te.assign(merchant_id=1)
    try:
        predict_fraud(bundle, extra)
        status = "traité (colonne ignorée + avertissement)"
    except DataValidationError as exc:
        status = f"rejeté : {exc}"
    rejections.append(
        {
            "scenario": "Colonne supplémentaire merchant_id",
            "attendu": "traitement",
            "obtenu": status,
            "message": None,
            "lignes_invalides": 0,
        }
    )

    # ---- Montants extrêmes : TRAITEMENT + ALERTE ---------------------------
    amax = float(X_tr["Amount"].max())
    for label, f in {
        "Amount × 10 (toutes les lignes)": lambda a: a * 10,
        "Amount × 100": lambda a: a * 100,
        f"Amount = 2 × max d'entraînement ({2 * amax:.0f})": lambda a: a * 0 + 2 * amax,
        "Amount = 1e9": lambda a: a * 0 + 1e9,
    }.items():
        X = X_te.assign(Amount=f(X_te["Amount"]))
        r, _ = score(bundle, X, y_te, base)
        rows.append({"scenario": label, "attendu": "traitement + alerte hors domaine", **r})

    # ---- Perturbations limitées : SENSIBILITÉ ------------------------------
    sd = X_tr[PCA_COLUMNS].std().to_numpy()
    sens = []
    for sigma in (0.01, 0.05, 0.10, 0.25):
        X = X_te.copy()
        X[PCA_COLUMNS] = X_te[PCA_COLUMNS] + rng.normal(0, sigma, (len(X_te), 28)) * sd
        r, _ = score(bundle, X, y_te, base)
        rows.append(
            {
                "scenario": f"Bruit gaussien V1..V28 : σ = {sigma:.0%} de l'écart-type",
                "attendu": "sensibilité mesurée",
                **r,
            }
        )
        sens.append((sigma, r["mean_abs_dproba"], r["class_flips"], r["recall"], r["fpr"]))
    for rel in (0.01, 0.05):
        X = X_te.assign(Amount=X_te["Amount"] * (1 + rng.uniform(-rel, rel, len(X_te))))
        r, _ = score(bundle, X, y_te, base)
        rows.append(
            {
                "scenario": f"Amount ± {rel:.0%} (arrondi, frais, change)",
                "attendu": "sensibilité mesurée",
                **r,
            }
        )
    for shift in (60, 3600):
        X = X_te.assign(Time=(X_te["Time"] + rng.uniform(-shift, shift, len(X_te))).clip(0))
        r, _ = score(bundle, X, y_te, base)
        rows.append(
            {"scenario": f"Time ± {shift} s (horodatage)", "attendu": "sensibilité mesurée", **r}
        )

    # Proximité de la frontière : où se produisent les changements de classe ?
    X = X_te.copy()
    X[PCA_COLUMNS] = X_te[PCA_COLUMNS] + rng.normal(0, 0.05, (len(X_te), 28)) * sd
    pert = predict_fraud(bundle, X)
    flips = pert["is_fraud"] != base["is_fraud"]
    logit = np.log(np.clip(base["fraud_probability"], 1e-15, 1 - 1e-15)) - np.log1p(
        -np.clip(base["fraud_probability"], 1e-15, 1 - 1e-15)
    )
    t = bundle.threshold
    margin = np.abs(logit - np.log(t / (1 - t)))
    boundary = {
        "n_flips_sigma_5pct": int(flips.sum()),
        "logit_margin_flipped_max": float(margin[flips].max()) if flips.any() else None,
        "logit_margin_flipped_median": float(margin[flips].median()) if flips.any() else None,
        "logit_margin_all_median": float(margin.median()),
        "share_within_logit_1": float((margin < 1).mean()),
    }

    # ---- Dérive de distribution : ALERTE ----------------------------------
    drift_rows = []
    for label, X in {
        "Amount × 1,1 (inflation / devise)": X_te.assign(Amount=X_te["Amount"] * 1.1),
        "Amount × 1,5": X_te.assign(Amount=X_te["Amount"] * 1.5),
        "V14 décalée de +0,5 σ": X_te.assign(V14=X_te["V14"] + 0.5 * X_tr["V14"].std()),
        "V4, V11, V12, V14 décalées de 1 σ": X_te.assign(
            **{
                c: X_te[c] + X_tr[c].std() * (1 if c in ("V4", "V11") else -1)
                for c in ("V4", "V11", "V12", "V14")
            }
        ),
    }.items():
        rep = drift_report(X_tr, X)
        r, _ = score(bundle, X, y_te, base)
        drift_rows.append(
            {
                "scenario": label,
                "psi_max": float(rep["psi"].iloc[0]),
                "feature_psi_max": rep["feature"].iloc[0],
                "n_alerte": int((rep["level"] == "alerte").sum()),
                "n_surveiller": int((rep["level"] == "surveiller").sum()),
                **{k: r[k] for k in ("recall", "precision", "fpr", "alerts", "class_flips")},
            }
        )
    ref_rep = drift_report(X_tr, X_te)
    drift_rows.insert(
        0,
        {
            "scenario": "Test non perturbé (contrôle)",
            "psi_max": float(ref_rep["psi"].iloc[0]),
            "feature_psi_max": ref_rep["feature"].iloc[0],
            "n_alerte": int((ref_rep["level"] == "alerte").sum()),
            "n_surveiller": int((ref_rep["level"] == "surveiller").sum()),
            **{k: base_row[k] for k in ("recall", "precision", "fpr", "alerts")},
            "class_flips": 0,
        },
    )

    # ---- Validation temporelle (hors échantillon, plus réaliste) ----------
    # Même configuration (modèle, hyperparamètres, règle de seuil) ré-entraînée
    # sur le passé : jour 1 -> entraînement, début du jour 2 -> seuil, fin -> évaluation.
    full = df.drop_duplicates().sort_values("Time").reset_index(drop=True)
    n = len(full)
    tr_t, va_t, te_t = (
        full.iloc[: int(0.6 * n)],
        full.iloc[int(0.6 * n) : int(0.8 * n)],
        full.iloc[int(0.8 * n) :],
    )
    (Xa, ya), (Xb, yb), (Xc, yc) = map(preprocess_data, (tr_t, va_t, te_t))
    pipe = train_model(Xa, ya, bundle.metadata["model"], bundle.metadata["params"])
    thr_t = select_threshold(yb, pipe.predict_proba(Xb)[:, 1])
    pc = pipe.predict_proba(Xc)[:, 1]
    temporal = {
        "train_time_range_h": [float(tr_t.Time.min() / 3600), float(tr_t.Time.max() / 3600)],
        "test_time_range_h": [float(te_t.Time.min() / 3600), float(te_t.Time.max() / 3600)],
        "n_test": len(yc),
        "n_test_fraud": int(yc.sum()),
        "threshold": thr_t,
        "metrics": evaluate_model(yc, (pc >= thr_t).astype(int), pc),
        "psi_train_vs_test_top": drift_report(Xa, Xc).head(5).to_dict(orient="records"),
    }

    results = {
        "base": base_row,
        "rejections": rejections,
        "scenarios": rows,
        "boundary": boundary,
        "drift": drift_rows,
        "temporal_validation": temporal,
        "seed": SEED,
    }
    dump_json(results, out_dir / "results" / "robustness.json")

    # ---- Figure : sensibilité au bruit ------------------------------------
    s = np.array(sens)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].plot(
        s[:, 0] * 100, s[:, 1], marker="o", ms=8, mec="white", mew=2, color=SERIES["logreg"]
    )
    axes[0].set(
        xlabel="Amplitude du bruit (% de l'écart-type)",
        ylabel="|Δ probabilité| moyen",
        title="Variation des probabilités",
    )
    axes[1].plot(
        s[:, 0] * 100,
        s[:, 3],
        marker="o",
        ms=8,
        mec="white",
        mew=2,
        color=SERIES["logreg"],
        label="Rappel",
    )
    axes[1].axhline(base_row["recall"], ls=":", color=INK_2, lw=1)
    axes[1].set(
        xlabel="Amplitude du bruit (% de l'écart-type)",
        ylabel="Rappel (test)",
        title="Rappel sous perturbation",
        ylim=(0, 1),
    )
    fig.savefig(out_dir / "figures" / "robustness_noise.png")
    plt.close(fig)

    print(pd.DataFrame(rejections)[["scenario", "obtenu"]].to_string(index=False))
    cols = [
        "scenario",
        "recall",
        "precision",
        "fpr",
        "ood_alerts",
        "class_flips",
        "mean_abs_dproba",
    ]
    print(pd.DataFrame(rows).reindex(columns=cols).round(5).to_string(index=False))
    print(pd.DataFrame(drift_rows).round(4).to_string(index=False))
    print(boundary)
    tm = temporal["metrics"]
    print(
        "temporel",
        {
            k: round(tm[k], 4)
            for k in ("precision", "recall", "fpr", "roc_auc", "average_precision")
        },
        tm["n_fraud"],
    )


if __name__ == "__main__":
    main()
