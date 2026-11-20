# 8. Rapport d'exécution des tests (Parties II et IV)

Exécution locale du 20 novembre 2026, avec Python 3.13.18 et pytest 9.1.1. La sortie brute est dans `reports/results/pytest_output.txt`.

```bash
pytest --cov=app --cov-report=term-missing      # avec data/creditcard.csv présent
pytest -m "not slow" --cov=app                   # configuration de la CI
```

## 8.1 Synthèse

| Fichier | Tests | Résultat | Portée |
|---|---:|---|---|
| `tests/test_preprocessing.py` | 48 | ✅ 48 réussis | T01 à T03, T08 à T10, T16 : contrat de données, types, NaN et inf, valeurs vides, cible, une observation, dimensions, partition, chargement de CSV, propriétés Hypothesis |
| `tests/test_prediction.py` | 57 | ✅ 57 réussis | T04, T11, T12, T17, T22, T24, T25 : contrat de sortie, seuil, entraînement, métriques, IC bootstrap, décision de non-régression, API, journaux |
| `tests/test_integration.py` | 14 | ✅ 14 réussis (dont 1 lent, sur le jeu réel) | T05, T06, T14, T15 : bout en bout, rechargement, ajustement sur l'entraînement seul, erreurs de chargement, empreinte |
| `tests/test_robustness.py` | 24 | ✅ 24 réussis | T18 à T21 : rejet, montants extrêmes, perturbations, dérive, Hypothesis (60 lots aléatoires) |
| `tests/test_regression.py` | 5 | ✅ 5 réussis | T06, T22, T23 : baseline figée, tolérances, reproductibilité numérique |
| `tests/test_security.py` | 8 | ✅ 8 réussis | T26 : secrets, données versionnées, `.gitignore` |
| **Total** | **156** | **✅ 156 réussis, 0 échec** | 11,6 s |

Dans la CI (sans le jeu réel) : 155 réussis et 1 désélectionné (`-m "not slow"`).

## 8.2 Couverture (`pytest-cov`, branches comprises)

| Module | Instructions | Couverture | Non couvert |
|---|---:|---:|---|
| `app/preprocessing.py` | 98 | 100 % | — |
| `app/training.py` | 75 | 100 % | — |
| `app/prediction.py` | 120 | 99 % | 1 ligne : garde-fou contre des probabilités non finies, inatteignable avec scikit-learn |
| `app/evaluation.py` | 72 | 91 % | `curves()` (exercée par les scripts et non par les tests), une branche de `bootstrap_ci` |
| `app/monitoring.py` | 33 | 100 % | — |
| `app/synthetic.py` | 23 | 93 % | message d'erreur sur les paramètres |
| **Total** | 435 | **98 %** | — |

## 8.3 Comment les tests vérifient un comportement (et pas seulement l'absence d'exception)

* Chaque rejet vérifie le **type** d'exception **et** le contenu du message : colonne nommée, nombre de lignes. Par exemple `match=r"manquantes \(2 lignes\).*V7"`.
* Les métriques sont vérifiées sur une **matrice de confusion connue à la main** (TP = 2, FN = 1, FP = 1, TN = 6).
* Le seuil est comparé à un **optimum par force brute**.
* L'absence de fuite est vérifiée sur les **statistiques internes** du scaler (`mean_`, `n_samples_seen_`).
* Le détecteur de secrets est **lui-même testé** sur des exemples positifs : un test toujours vert ne prouverait rien.
* Les valeurs limites sont couvertes : montant nul (accepté), −0,01 (rejeté), une seule observation, lot vide, lot de 1 001 lignes (rejeté), 1e12 (traité).

## 8.4 Anomalie détectée par les tests

| ID | Test | Constat | Correction | Statut |
|---|---|---|---|---|
| A10 | `test_load_data_unreadable_file` | Un CSV non UTF-8 provoquait une `UnicodeDecodeError` brute : la lecture de l'en-tête, ajoutée pour détecter les colonnes dupliquées, était hors du bloc `try`. | Lecture de l'en-tête déplacée dans le `try`, avec conversion en `DataValidationError` explicite. | **Corrigée**. Test vert depuis. |

## 8.5 Comportements documentés (rejet ou correction)

Conformément au § 2.1 du sujet, chaque cas invalide a un comportement **unique et testé** :

| Cas | Comportement | Test |
|---|---|---|
| Colonne absente | rejet | `test_missing_required_column_is_rejected` |
| Colonne dupliquée (DataFrame ou en-tête CSV) | rejet | `test_duplicated_column_is_rejected`, `test_load_data_detects_duplicated_header` |
| Colonne en trop | **correction documentée** : ignorée, avec avertissement | `test_extra_column_is_dropped_with_warning_without_values` |
| Types `float32` / `int64` | **correction documentée** : conversion en `float64` | `test_types_are_normalised` |
| Cible en `float` (0,0 / 1,0) | **correction documentée** : conversion en `int64` | `test_float_target_zero_one_is_accepted_and_cast` |
| NaN, ±inf, texte, booléen, cible ∉ {0, 1}, montant < 0, données vides | rejet | voir § 8.1 |
