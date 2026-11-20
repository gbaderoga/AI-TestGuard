"""Paramètres centralisés du système.

Toutes les valeurs ci-dessous sont fixées *avant* l'évaluation sur le jeu de test
(cf. reports/01_plan_de_test.md, section « Critères d'acceptation »).
Les modifier après observation du jeu de test invaliderait la qualification.
"""

from __future__ import annotations

SEED = 42

# Partition stratifiée : 60 % entraînement / 20 % validation / 20 % test
VAL_SIZE = 0.20
TEST_SIZE = 0.20

# Coûts métier utilisés pour choisir le seuil (sur la validation uniquement).
# FN : perte moyenne d'une fraude non détectée + frais de litige (ordre de grandeur).
# FP : coût d'une vérification par un analyste + friction client.
COST_FN = 100.0
COST_FP = 5.0

# Grilles d'hyperparamètres explorées sur la validation.
PARAM_GRIDS: dict[str, list[dict]] = {
    "logreg": [{"C": c} for c in (0.01, 0.1, 1.0)],
    "tree": [{"max_depth": d, "min_samples_leaf": leaf} for d in (4, 6, 8) for leaf in (20, 50)],
}

# Critères d'acceptation (jeu de test, évaluation finale unique).
ACCEPTANCE = {
    "roc_auc_min": 0.95,
    "average_precision_min": 0.70,
    "recall_min": 0.75,
    "precision_min": 0.30,
    "fpr_max": 0.005,
}

# Non-régression : dégradation absolue tolérée par rapport à la baseline.
REGRESSION_TOLERANCE = {
    "average_precision": 0.02,
    "roc_auc": 0.01,
    "recall": 0.03,
    "precision": 0.05,
    "fpr": 0.001,  # hausse maximale tolérée
}

# Reproductibilité : écart maximal toléré entre deux entraînements identiques
# ou entre un modèle et sa version rechargée.
REPRO_ATOL = 1e-9

# Performance (cf. reports/05_performance.md pour la justification).
PERF = {
    "p95_single_ms_max": 20.0,
    "throughput_min_per_s": 5_000,
    "model_size_mb_max": 5.0,
    "rss_mb_max": 500.0,
}

# Surveillance : seuils d'alerte de dérive (Population Stability Index).
PSI_WARNING = 0.10
PSI_ALERT = 0.25
