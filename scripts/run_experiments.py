"""Évaluation ML complète (Partie III et VI.1).

Protocole :
1. partition stratifiée 60/20/20 avant tout ajustement ;
2. grille d'hyperparamètres évaluée sur la VALIDATION (Average Precision) ;
3. seuil choisi sur la VALIDATION (coût FN=100, FP=5) ;
4. choix du modèle final sur la VALIDATION ;
5. évaluation UNIQUE sur le test, comparée aux critères fixés à l'avance.

Usage : python scripts/run_experiments.py --data data/creditcard.csv --out reports
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
from _common import INK_2, LABELS, ROOT, SERIES, dump_json, load_splits, plt

from app import config
from app.evaluation import bootstrap_ci, check_acceptance, curves, evaluate_model
from app.prediction import ModelBundle, save_model
from app.training import expected_cost, select_model, select_threshold


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/creditcard.csv")
    ap.add_argument("--out", default="reports")
    ap.add_argument("--model-path", default="models/model.joblib")
    args = ap.parse_args()
    out = ROOT / args.out
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    df, splits = load_splits(args.data)
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = splits["train"], splits["val"], splits["test"]
    data_summary = {
        "n_raw": len(df),
        "n_fraud_raw": int(df["Class"].sum()),
        "n_duplicates_removed": int(df.duplicated().sum()),
        **{
            f"{k}": {"n": len(v[1]), "n_fraud": int(v[1].sum()), "rate": float(v[1].mean())}
            for k, v in splits.items()
        },
    }

    results, selections = {}, {}
    for name in ("logreg", "tree"):
        t0 = time.perf_counter()
        sel = select_model(X_tr, y_tr, X_va, y_va, name)
        fit_s = time.perf_counter() - t0
        pipe = sel.pipeline
        p_tr = pipe.predict_proba(X_tr)[:, 1]
        p_va = pipe.predict_proba(X_va)[:, 1]
        thr = select_threshold(y_va, p_va)
        res = {
            "params": sel.params,
            "grid": sel.history,
            "threshold": thr,
            "grid_search_seconds": fit_s,
            "train": evaluate_model(y_tr, (p_tr >= thr).astype(int), p_tr),
            "val": evaluate_model(y_va, (p_va >= thr).astype(int), p_va),
            "val_at_0.5": evaluate_model(y_va, (p_va >= 0.5).astype(int), p_va),
            "val_cost": expected_cost(y_va, p_va, thr),
        }
        selections[name] = (sel, thr, p_va)
        results[name] = res

    # Choix du modèle final : Average Precision de validation (jamais le test).
    final_name = max(results, key=lambda n: results[n]["val"]["average_precision"])

    # ---- Évaluation finale sur le test (une seule fois, pour les deux modèles) ----
    test_proba = {}
    for name, (sel, thr, _) in selections.items():
        p_te = sel.pipeline.predict_proba(X_te)[:, 1]
        test_proba[name] = p_te
        m = evaluate_model(y_te, (p_te >= thr).astype(int), p_te)
        results[name]["test"] = m
        results[name]["test_ci95"] = bootstrap_ci(y_te, p_te, thr, n_boot=1000)
        results[name]["acceptance"] = check_acceptance(m)
        results[name]["accepted"] = all(results[name]["acceptance"].values())
        results[name]["overfit_gap_ap"] = (
            results[name]["train"]["average_precision"] - m["average_precision"]
        )

    dummy = evaluate_model(y_te, np.zeros_like(y_te))

    # ---- Analyse des erreurs du modèle final ----
    sel, thr, _ = selections[final_name]
    p_te = test_proba[final_name]
    pred = p_te >= thr
    y = y_te.to_numpy()
    groups = {
        "TP": (y == 1) & pred,
        "FN": (y == 1) & ~pred,
        "FP": (y == 0) & pred,
        "TN": (y == 0) & ~pred,
    }
    error_analysis = {
        g: {
            "n": int(mask.sum()),
            "proba_median": float(np.median(p_te[mask])) if mask.any() else None,
            "amount_median": float(X_te["Amount"][mask].median()) if mask.any() else None,
            "amount_mean": float(X_te["Amount"][mask].mean()) if mask.any() else None,
        }
        for g, mask in groups.items()
    }
    error_analysis["FN_amount_total"] = float(X_te["Amount"][groups["FN"]].sum())
    error_analysis["TP_amount_total"] = float(X_te["Amount"][groups["TP"]].sum())
    error_analysis["FN_list"] = (
        pd.DataFrame({"proba": p_te[groups["FN"]], "Amount": X_te["Amount"][groups["FN"]]})
        .sort_values("proba", ascending=False)
        .round(4)
        .to_dict(orient="records")
    )
    error_analysis["FP_proba_quantiles"] = (
        np.quantile(p_te[groups["FP"]], [0.1, 0.5, 0.9]).tolist() if groups["FP"].any() else []
    )

    # ---- Sensibilité au seuil (validation) ----
    _, _, p_va = selections[final_name]
    grid_thr = np.unique(np.r_[np.linspace(0.01, 0.99, 99), 1 - np.logspace(-7, -2, 60), thr])
    thr_table = []
    for t in grid_thr:
        m = evaluate_model(y_va, (p_va >= t).astype(int))
        thr_table.append(
            {
                "threshold": float(t),
                "precision": m["precision"],
                "recall": m["recall"],
                "fpr": m["fpr"],
                "fp": m["fp"],
                "fn": m["fn"],
                "cost": expected_cost(y_va, p_va, t),
            }
        )

    # ---- Bundle final ----
    bundle = ModelBundle.from_training(
        sel.pipeline,
        thr,
        X_tr,
        {
            "model": final_name,
            "params": sel.params,
            "seed": config.SEED,
            "cost_fn": config.COST_FN,
            "cost_fp": config.COST_FP,
            "train_size": len(X_tr),
        },
    )
    model_path = ROOT / args.model_path
    digest = save_model(bundle, model_path)

    summary = {
        "data": data_summary,
        "final_model": final_name,
        "final_threshold": thr,
        "model_sha256": digest,
        "model_size_bytes": model_path.stat().st_size,
        "results": results,
        "dummy_all_normal": dummy,
        "error_analysis": error_analysis,
        "threshold_sensitivity_val": thr_table,
        "acceptance_criteria": config.ACCEPTANCE,
    }
    dump_json(summary, out / "results" / "experiments.json")

    # ---- Figures ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for name, p in test_proba.items():
        c = curves(y_te, p)
        fpr, tpr, _ = c["roc"]
        prec, rec, _ = c["pr"]
        auc = results[name]["test"]["roc_auc"]
        apv = results[name]["test"]["average_precision"]
        axes[0].plot(fpr, tpr, color=SERIES[name], label=f"{LABELS[name]} (AUC {auc:.3f})")
        axes[1].plot(rec, prec, color=SERIES[name], label=f"{LABELS[name]} (AP {apv:.3f})")
        m = results[name]["test"]
        axes[1].scatter(
            m["recall"],
            m["precision"],
            s=60,
            color=SERIES[name],
            edgecolor="white",
            linewidth=2,
            zorder=3,
        )
        axes[0].scatter(
            m["fpr"],
            m["recall"],
            s=60,
            color=SERIES[name],
            edgecolor="white",
            linewidth=2,
            zorder=3,
        )
    axes[0].plot([0, 1], [0, 1], ls="--", color=INK_2, lw=1, label="Hasard")
    axes[0].set(xlabel="Taux de faux positifs", ylabel="Rappel (TPR)", title="Courbe ROC (test)")
    prevalence = float(y_te.mean())
    axes[1].axhline(prevalence, ls="--", color=INK_2, lw=1, label=f"Prévalence {prevalence:.4f}")
    axes[1].set(
        xlabel="Rappel",
        ylabel="Précision",
        title="Courbe précision-rappel (test)",
        ylim=(0, 1.02),
    )
    for a in axes:
        a.legend(loc="lower right" if a is axes[0] else "lower left", fontsize=8.5)
    fig.text(
        0.5,
        -0.02,
        "Points : seuil retenu sur la validation",
        ha="center",
        color=INK_2,
        fontsize=8.5,
    )
    fig.savefig(fig_dir / "roc_pr_test.png")
    plt.close(fig)

    # Zoom ROC sur la zone utile (FPR < 1 %)
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    for name, p in test_proba.items():
        fpr, tpr, _ = curves(y_te, p)["roc"]
        ax.plot(fpr, tpr, color=SERIES[name], label=LABELS[name])
    ax.axvline(config.ACCEPTANCE["fpr_max"], ls=":", color=INK_2, lw=1)
    ax.text(config.ACCEPTANCE["fpr_max"], 0.05, " FPR max\n accepté", color=INK_2, fontsize=8)
    ax.set(
        xlim=(0, 0.01),
        ylim=(0, 1),
        xlabel="Taux de faux positifs",
        ylabel="Rappel",
        title="ROC – zone opérationnelle (test)",
    )
    ax.legend(loc="lower right", fontsize=8.5)
    fig.savefig(fig_dir / "roc_zoom_test.png")
    plt.close(fig)

    # Matrices de confusion
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for a, name in zip(axes, ("logreg", "tree"), strict=True):
        m = results[name]["test"]
        mat = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        a.imshow(np.log10(mat + 1), cmap="Blues", vmin=0, vmax=5.5)
        a.grid(False)
        labels = [["TN", "FP"], ["FN", "TP"]]
        for i in range(2):
            for j in range(2):
                v = mat[i, j]
                a.text(
                    j,
                    i,
                    f"{labels[i][j]}\n{v:,}".replace(",", " "),
                    ha="center",
                    va="center",
                    fontsize=11,
                    color="white" if v > 3000 else "#0b0b0b",
                )
        a.set_xticks([0, 1], ["Prédit normal", "Prédit fraude"])
        a.set_yticks([0, 1], ["Réel normal", "Réel fraude"])
        a.set_title(f"{LABELS[name]}\nseuil = {results[name]['threshold']:.3f}", fontsize=10)
    fig.savefig(fig_dir / "confusion_matrices_test.png")
    plt.close(fig)

    # Sensibilité au seuil (validation)
    tt = pd.DataFrame(thr_table)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(tt.threshold, tt.precision, color=SERIES["logreg"], label="Précision")
    axes[0].plot(tt.threshold, tt.recall, color=SERIES["tree"], label="Rappel")
    axes[0].axvline(thr, ls=":", color=INK_2, lw=1)
    axes[0].set(
        xlabel="Seuil (échelle logit)",
        ylabel="Valeur",
        title=f"Précision / rappel – {LABELS[final_name]}",
        ylim=(0, 1.02),
    )
    axes[0].legend(fontsize=8.5)
    for a in axes:
        a.set_xscale("logit")
        a.set_xlim(0.01, 1 - 1e-7)
    axes[1].plot(tt.threshold, tt.cost, color=SERIES["logreg"])
    axes[1].axvline(thr, ls=":", color=INK_2, lw=1)
    axes[1].annotate(
        f"seuil retenu\n{thr:.4f}",
        (thr, tt.cost.min()),
        xytext=(-70, 40),
        textcoords="offset points",
        color=INK_2,
        fontsize=8.5,
    )
    axes[1].set(
        xlabel="Seuil (échelle logit)",
        ylabel="Coût (FN×100 + FP×5)",
        title="Coût attendu sur la validation",
        yscale="log",
    )
    fig.savefig(fig_dir / "threshold_sensitivity_val.png")
    plt.close(fig)

    print(f"Modèle final : {final_name}  seuil={thr:.4f}  sha256={digest[:12]}")
    for name in results:
        t = results[name]["test"]
        print(
            f"{name:7s} test: P={t['precision']:.3f} R={t['recall']:.3f} F1={t['f1']:.3f} "
            f"ROC={t['roc_auc']:.4f} AP={t['average_precision']:.4f} FPR={t['fpr']:.5f} "
            f"accepté={results[name]['accepted']}"
        )


if __name__ == "__main__":
    main()
