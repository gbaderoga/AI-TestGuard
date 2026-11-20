# AI-TestGuard

**Validation, robustesse et fiabilité d'un modèle de détection de fraude bancaire.**

Ce projet qualifie un modèle de Machine Learning (jeu Kaggle *Credit Card Fraud Detection*) en suivant une démarche d'ingénierie QA. Il comprend un plan de test, des tests unitaires, d'intégration, de robustesse et de non-régression, des mesures de performance, une intégration continue GitHub Actions et un rapport de qualification argumenté.

> **Décision de qualification : AJOURNÉ.** Le pipeline logiciel est conforme : 156 tests réussis, couverture de 98 %. En revanche, le modèle n'atteint pas le critère de PR-AUC fixé à l'avance (0,681 pour un minimum de 0,70), et trois anomalies majeures restent ouvertes. Voir [`reports/RAPPORT_QUALIFICATION.md`](reports/RAPPORT_QUALIFICATION.md).

## Structure

```
AI-TestGuard/
├── app/
│   ├── config.py           # graines, coûts, critères d'acceptation et tolérances (figés)
│   ├── preprocessing.py    # contrat de données, chargement, partition, prétraitement
│   ├── training.py         # pipeline, sélection sur la validation, seuil par le coût
│   ├── prediction.py       # prédiction, persistance + SHA-256, validation API
│   ├── evaluation.py       # métriques, IC bootstrap, critères, comparaison à la baseline
│   ├── monitoring.py       # PSI, taux d'alerte
│   └── synthetic.py        # données synthétiques au schéma Kaggle (CI)
├── tests/
│   ├── test_preprocessing.py   test_prediction.py   test_integration.py
│   ├── test_robustness.py      test_regression.py   test_security.py
│   └── baselines/ci_baseline.json   # référence CI figée
├── scripts/
│   ├── run_experiments.py   # évaluation ML complète + figures + modèle
│   ├── run_robustness.py    # scénarios de robustesse, dérive, validation temporelle
│   ├── compare_baseline.py  # non-régression sur le jeu réel (--freeze pour figer)
│   ├── benchmark.py         # latence, débit, mémoire
│   └── make_ci_baseline.py
├── data/README.md            # jeu non versionné : comment l'obtenir et le vérifier
├── reports/                  # plan de test, rapports, figures, résultats JSON
├── .github/workflows/tests.yml
├── requirements.txt  pyproject.toml  .gitignore
```

## Démarrage

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Tests rapides (sans le jeu réel), identiques à la CI
ruff check . && ruff format --check .
pytest -m "not slow" --cov=app

# Qualification complète (data/creditcard.csv requis, cf. data/README.md)
python scripts/run_experiments.py      # ≈ 1 min, crée models/model.joblib
python scripts/run_robustness.py
python scripts/compare_baseline.py     # compare à reports/baselines/baseline_v1.0.json
python scripts/benchmark.py
pytest                                 # inclut le test lent sur le jeu réel
```

## Rapports

| Partie du sujet | Livrable |
|---|---|
| I. Plan de test | [`reports/01_plan_de_test.md`](reports/01_plan_de_test.md) : 30 exigences, risques, matrice, critères |
| II et IV. Tests unitaires et d'intégration | [`reports/08_execution_tests.md`](reports/08_execution_tests.md), `tests/` |
| III et VI.1. Évaluation ML | [`reports/02_evaluation_ml.md`](reports/02_evaluation_ml.md) |
| V et VI.2. Robustesse et non-régression | [`reports/03_robustesse_non_regression.md`](reports/03_robustesse_non_regression.md) |
| VII. Biais | [`reports/04_audit_biais.md`](reports/04_audit_biais.md) |
| VIII. Performance et sécurité | [`reports/05_performance_securite.md`](reports/05_performance_securite.md) |
| IX. CI | [`reports/07_ci_github_actions.md`](reports/07_ci_github_actions.md), `.github/workflows/tests.yml` |
| X. Surveillance | [`reports/06_surveillance.md`](reports/06_surveillance.md) |
| Rapport final | [`reports/RAPPORT_QUALIFICATION.md`](reports/RAPPORT_QUALIFICATION.md) |

## Règles méthodologiques appliquées

* Partition stratifiée **avant** tout ajustement, avec retrait des doublons exacts au préalable.
* Hyperparamètres, seuil et choix du modèle fixés sur la **validation** uniquement. Le test n'est évalué qu'une fois.
* Critères d'acceptation figés dans `app/config.py` avant l'évaluation.
* Les baselines ne sont jamais recalculées automatiquement.
* Graine 42 partout, versions des bibliothèques figées.
* Aucune donnée, aucun modèle et aucun secret ne sont versionnés (vérifié par `tests/test_security.py`).

## Échec volontaire (preuve de la CI)

*Actions → tests → Run workflow → `simulate_regression` = true*, ou bien en local :

```bash
AITG_SIMULATE_REGRESSION=1 pytest tests/test_regression.py   # 3 échecs attendus
```
