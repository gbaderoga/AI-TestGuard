# 1. Plan de test : AI-TestGuard

| | |
|---|---|
| **Système testé** | Pipeline de détection de fraude par carte bancaire (prétraitement, modèle, seuil, prédiction, interface de type API) |
| **Version** | code 1.0.0, modèle v1.0 (régression logistique) |
| **Jeu de données** | Kaggle *Credit Card Fraud Detection* (mlg-ulb) : 284 807 transactions, 492 fraudes, SHA-256 `76274b69…51a89` |
| **Rédaction du plan** | avant toute évaluation sur le jeu de test. Les critères de la section 6 sont figés dans `app/config.py`. |

## 1.1 Périmètre

| Composant | Inclus | Fichier | Ce qui est testé |
|---|---|---|---|
| Données | oui | `app/preprocessing.py` (`validate_data`, `load_data`) | contrat de schéma, types, valeurs non finies, bornes, cible, doublons |
| Partition | oui | `split_data` | stratification, partitions disjointes, déduplication, déterminisme |
| Prétraitement | oui | `build_preprocessor`, `preprocess_data` | absence de valeurs non finies, dimensions, ajustement sur l'entraînement seul |
| Modèle | oui | `app/training.py` | entraînement, sélection sur la validation, choix du seuil par le coût |
| Prédiction | oui | `app/prediction.py` (`predict_fraud`) | contrat de sortie, seuil, alerte hors domaine, invariances |
| Persistance | oui | `save_model`, `load_model` | rechargement identique, erreurs explicites, empreinte SHA-256 |
| Intégration / API | oui (sans déploiement) | `parse_api_payload`, `handle_request` | validation stricte des requêtes, codes 200/400, journaux |
| Évaluation | oui | `app/evaluation.py` | exactitude des métriques, IC bootstrap, comparaison à la baseline |
| Surveillance | prototype | `app/monitoring.py` | PSI, taux d'alerte |
| CI | oui | `.github/workflows/tests.yml` | lint, tests, couverture, signalement des échecs |
| **Hors périmètre** | — | — | infrastructure de production, authentification de l'API, base de données des transactions, processus d'investigation des analystes, conformité réglementaire (validée par d'autres fonctions) |

## 1.2 Exigences

### Fonctionnelles

| Réf. | Exigence |
|---|---|
| EF1 | Le système refuse toute entrée qui ne respecte pas le contrat de données, avec un message qui nomme la cause et les colonnes concernées. |
| EF2 | Le système renvoie pour chaque transaction une probabilité finie dans [0, 1], une décision binaire et une alerte « hors domaine ». |
| EF3 | La décision applique le seuil choisi sur la validation et enregistré avec le modèle. |
| EF4 | Le modèle sauvegardé puis rechargé produit des prédictions identiques. |
| EF5 | Les métriques de qualité (précision, rappel, F1, ROC-AUC, PR-AUC, FPR, FNR, matrice de confusion) sont calculées correctement. |
| EF6 | Une nouvelle version est comparée à une référence figée et la décision accepter / investiguer / rejeter est automatique. |

### Non fonctionnelles

| Réf. | Exigence |
|---|---|
| ENF1 | **Qualité prédictive** : critères de la section 6 atteints sur un jeu de test indépendant. |
| ENF2 | **Reproductibilité** : même code, mêmes données et même graine donnent un écart de probabilité ≤ 1e-9. |
| ENF3 | **Robustesse** : comportement contrôlé (rejet, traitement ou alerte) sur les entrées imparfaites, et sensibilité mesurée aux perturbations. |
| ENF4 | **Performance** : P95 d'une prédiction ≤ 20 ms, débit ≥ 5 000 tx/s, modèle ≤ 5 Mo, mémoire ≤ 500 Mo. |
| ENF5 | **Sécurité opérationnelle** : aucune valeur de transaction dans les journaux, aucun secret ni jeu de données dans le dépôt, modèle vérifié avant chargement. |
| ENF6 | **Automatisation** : les tests rapides s'exécutent à chaque push et à chaque Pull Request, et un échec bloque la fusion. |
| ENF7 | **Absence de fuite** : aucune information du jeu de test n'intervient dans l'apprentissage, la sélection ou le seuil. |

## 1.3 Matrice de traçabilité (exigences, risques, tests)

Les lignes T01 à T07 reprennent la matrice du sujet. Les lignes T08 à T30 sont les exigences ajoutées.

| ID | Exigence | Réf. | Risque maîtrisé | Type de test | Résultat attendu | Implémentation | Statut |
|---|---|---|---|---|---|---|---|
| T01 | Les colonnes nécessaires sont présentes | EF1 | prédiction sur un schéma tronqué, valeurs décalées | Données | Rejet si une des 31 colonnes manque | `test_preprocessing::test_missing_required_column_is_rejected` (6 cas), `test_valid_data_passes…` | ✅ |
| T02 | Les données respectent les types attendus | EF1 | chaîne « 12,5 » ou booléen convertis silencieusement | Unitaire | Types numériques, rejet des types texte, objet et booléen | `test_types_are_normalised`, `test_non_numeric_type_is_rejected` (3), `test_boolean_column_is_rejected` | ✅ |
| T03 | Le prétraitement ne produit pas de valeurs non finies | EF1, EF2 | NaN propagé jusqu'au score, score indéfini | Unitaire | Rejet des NaN et ±inf en entrée, sortie du préprocesseur finie | `test_missing_values_are_rejected_with_count`, `test_infinite_values_are_rejected`, `test_preprocessor_output_is_finite…`, Hypothesis (50 cas) | ✅ |
| T04 | Le modèle produit une probabilité valide | EF2 | score hors [0, 1] mal interprété par l'aval | Fonctionnel ML | Probabilités finies dans [0, 1] | `test_prediction_probabilities_are_valid`, `test_any_valid_input_yields_valid_probabilities` (Hypothesis, 60 lots) | ✅ |
| T05 | Les performances sont évaluées sur des données indépendantes | ENF7 | performance surestimée par fuite | ML | Partitions disjointes, scaler ajusté sur l'entraînement, seuil issu de la validation | `test_split_is_stratified_disjoint…`, `test_scaler_is_fitted_on_training_data_only`, `test_threshold_was_chosen_on_validation` | ✅ |
| T06 | Les résultats sont reproductibles | ENF2 | impossibilité d'auditer ou de comparer des versions | Non-régression | Écart ≤ 1e-9 | `test_training_is_reproducible`, `test_predictions_match_baseline_numerically`, `test_split_is_deterministic…` | ✅ |
| T07 | Les tests s'exécutent sur GitHub | ENF6 | régression fusionnée sans contrôle | CI | Workflow vert, ou échec visible et bloquant | `.github/workflows/tests.yml`, échec volontaire `simulate_regression` | ✅ en local, preuve GitHub à fournir (cf. `07_ci_github_actions.md`) |
| **T08** | Colonnes dupliquées refusées, colonnes hors contrat ignorées avec un avertissement | EF1, ENF5 | pandas renomme un doublon (`Amount.1`) et masque l'erreur ; une colonne sensible ajoutée par erreur | Unitaire | Rejet des doublons, même dans un CSV ; colonne en trop retirée, nom journalisé sans sa valeur | `test_duplicated_column_is_rejected`, `test_load_data_detects_duplicated_header`, `test_extra_column_is_dropped_with_warning…` | ✅ |
| **T09** | La cible ne contient que 0 et 1 | EF1 | étiquette corrompue faussant l'entraînement | Unitaire | Rejet de 2, −1, 0,5 et NaN | `test_target_outside_zero_one_is_rejected` (3), `test_target_with_nan_is_rejected` | ✅ |
| **T10** | Partition stratifiée, sans doublon exact entre partitions | ENF7 | 1 081 doublons exacts dans le jeu Kaggle : une même transaction en entraînement et en test | Données / ML | Prévalence identique à ±0,05 point, aucune ligne commune | `test_split_removes_exact_duplicates…`, `test_split_is_stratified…` | ✅ |
| **T11** | Les métriques sont exactes et l'accuracy seule est insuffisante | EF5 | décision prise sur une métrique fausse ou trompeuse | Unitaire | Valeurs connues retrouvées ; « tout normal » donne 99,8 % d'accuracy pour un rappel nul | `test_metrics_on_known_confusion_matrix`, `test_high_accuracy_can_hide_zero_recall` | ✅ |
| **T12** | Le seuil minimise le coût métier sur la validation | EF3 | seuil 0,5 arbitraire : 1 479 fausses alertes sur la validation | Unitaire / ML | Seuil égal à l'optimum par force brute, qui varie avec le ratio de coûts | `test_threshold_minimises_cost_brute_force`, `test_threshold_moves_with_cost_ratio` | ✅ |
| **T13** | Les critères d'acceptation sont atteints sur le test | ENF1 | mise en production d'un modèle insuffisant | ML | 5 critères de la section 6 | `scripts/run_experiments.py` | ❌ PR-AUC 0,681 < 0,70 |
| **T14** | Les prédictions sont identiques après sauvegarde et rechargement | EF4 | divergence entre le modèle validé et le modèle servi | Intégration | Écart ≤ 1e-9, mêmes décisions, même seuil, mêmes métadonnées | `test_predictions_identical_after_reload` | ✅ |
| **T15** | Les erreurs de chargement sont explicites | EF1 | plantage opaque ou chargement d'un artefact altéré | Intégration | `ModelLoadError` ou `DataValidationError` lisibles | `test_load_missing/corrupted/wrong_object…`, `test_checksum_is_verified…`, `test_load_data_unreadable_file` | ✅ (anomalie A10 corrigée) |
| **T16** | Les dimensions restent cohérentes, y compris pour une seule observation | EF2 | décalage entre lignes et prédictions | Unitaire / Intégration | 30 variables en sortie, autant de prédictions que de lignes, index conservé | `test_single_observation_is_processed`, `test_dimensions_are_consistent`, `test_preprocess_returns_consistent_dimensions` | ✅ |
| **T17** | Les prédictions ne dépendent ni de l'ordre des lignes ni de la taille du lot | EF2 | score qui change selon le regroupement des requêtes | Fonctionnel | Écart ≤ 1e-12 | `test_prediction_is_row_order_invariant`, `test_single_row_equals_batch_prediction` | ✅ |
| **T18** | Les entrées invalides sont rejetées en robustesse | ENF3 | score produit sur une donnée corrompue | Robustesse | 10 scénarios rejetés sur 10 | `test_robustness.py`, `scripts/run_robustness.py` | ✅ (anomalie A5 : rejet du lot entier) |
| **T19** | Un montant extrême est traité et signalé | ENF3 | saturation numérique, ou montant aberrant non remarqué | Robustesse | Probabilité finie et alerte hors domaine | `test_extreme_amount_is_processed_with_alert` (3), scénarios réels | ✅ |
| **T20** | La sensibilité aux petites perturbations est mesurée et bornée | ENF3 | décision instable pour du bruit de mesure | Robustesse | Avec σ = 1 % : \|Δp\| moyen < 0,01 et < 1 % de changements de classe, uniquement près du seuil | `test_small_perturbations_have_small_effect` (3), `test_class_flips_happen_only_near…` | ✅ |
| **T21** | Une dérive de distribution est détectée | ENF3 | dégradation silencieuse en production | Robustesse / surveillance | PSI ≥ 0,25 sur les variables décalées, PSI < 0,02 sans décalage | `test_drift_report_flags_only_shifted_features`, `test_psi_*` | ✅ (anomalie A6 : PSI peu sensible aux montants) |
| **T22** | Pas de dégradation au-delà de la tolérance par rapport à la baseline | EF6 | régression introduite par une modification | Non-régression | Décision « accepter », un échec simulé est détecté | `test_no_performance_regression`, `scripts/compare_baseline.py` | ✅ |
| **T23** | La baseline est figée et contient tout le contexte de reproduction | EF6 | référence recalculée qui masque les régressions | Non-régression | Version du code, commit, paramètres, graine, seuil, versions des bibliothèques, empreinte des données ; refus de l'écrasement | `test_baseline_records_required_context`, `--freeze` refusé si le fichier existe | ✅ |
| **T24** | L'API valide strictement les requêtes | ENF5 | injection de types, champs inattendus, déni de service par gros lots | Sécurité | Code 400 pour 16 formes invalides (champ manquant ou en trop, chaîne, booléen, null, liste, NaN, Infinity, lot > 1 000, montant négatif, JSON invalide…) | `test_api_rejects_*` | ✅ |
| **T25** | Aucune donnée de transaction dans les journaux | ENF5 | fuite de données sensibles par les logs | Sécurité | Journaux limités aux compteurs et aux noms de colonnes | `test_logs_do_not_contain_transaction_values`, `test_extra_column…` | ✅ |
| **T26** | Aucun secret ni jeu de données dans le dépôt | ENF5 | fuite d'identifiants ou de données sous licence | Sécurité | Recherche de motifs vide, aucun fichier > 5 Mo, `.gitignore` conforme | `test_security.py` (8 tests, dont un qui vérifie le détecteur lui-même) | ✅ |
| **T27** | La latence, le débit, la taille et la mémoire respectent les seuils | ENF4 | délai d'autorisation de paiement dépassé | Performance | Seuils ENF4 | `scripts/benchmark.py` | ✅ |
| **T28** | Pas de surapprentissage excessif | ENF1 | modèle qui mémorise l'entraînement | ML | Écart de PR-AUC entre entraînement et test ≤ 0,10 | `scripts/run_experiments.py` | ✅ LR 0,047, ❌ arbre 0,13 (écarté) |
| **T29** | La performance tient sur une période postérieure | ENF1 | modèle dépassé dès la mise en service | ML hors échantillon | Critères atteints sur une partition temporelle | `scripts/run_robustness.py` | ❌ rappel 0,689 |
| **T30** | Pas d'écart de performance entre groupes de clients | — | discrimination | Équité | Non testable : aucun attribut de groupe dans les données | `04_audit_biais.md` | ⚠️ non évaluable |

## 1.4 Politique de traitement des entrées

| Situation | Comportement | Justification |
|---|---|---|
| Colonne absente, renommée ou dupliquée | **Rejet** | Le score n'aurait pas de sens. |
| Type non numérique, booléen ou null | **Rejet** | Aucune conversion implicite. |
| NaN, ±inf | **Rejet** | Imputer reviendrait à scorer une transaction qui n'a pas été reçue. |
| `Amount < 0` ou `Time < 0` | **Rejet** | Valeurs impossibles selon le contrat. |
| Colonne supplémentaire | **Traitement** : colonne ignorée et avertissement (nom seulement) | Tolérance aux évolutions de l'appelant, sans fuite de valeur. |
| Valeur finie hors de l'intervalle d'entraînement (montant extrême) | **Traitement et alerte** `out_of_domain_alert` | Le score reste calculable (log1p), mais sa fiabilité est inconnue. |
| Dérive de distribution d'un lot | **Alerte** de surveillance (PSI) | Signal statistique, pas une invalidité ligne par ligne. |

## 1.5 Outils et preuves conservées

| Activité | Outils | Preuve conservée |
|---|---|---|
| Tests unitaires et d'intégration | pytest 9.1, pytest-cov, Hypothesis | `reports/results/pytest_output.txt`, `junit.xml`, `coverage.xml` (artefacts de la CI) |
| Évaluation ML | scikit-learn 1.9.1, matplotlib | `reports/results/experiments.json`, `reports/figures/*.png` |
| Robustesse | NumPy, Hypothesis | `reports/results/robustness.json` |
| Non-régression | baseline JSON figée | `reports/baselines/baseline_v1.0.json`, `tests/baselines/ci_baseline.json`, `reports/results/regression_comparison.json` |
| Performance | `time.perf_counter`, psutil, `http.server` | `reports/results/benchmark.json` |
| Qualité du code | Ruff | journal de la CI |
| Traçabilité du modèle | SHA-256 du modèle et du CSV | `experiments.json`, baseline |

## 1.6 Critères d'acceptation et de rejet du modèle

Ces critères sont fixés **avant** l'évaluation finale. Ils sont figés dans `app/config.py` et ne doivent pas être ajustés après observation du jeu de test.

| Critère | Seuil | Justification |
|---|---|---|
| ROC-AUC | ≥ 0,95 | Capacité de classement minimale. Sur ce jeu, la littérature et une régression logistique simple dépassent 0,95. |
| PR-AUC (Average Precision) | ≥ 0,70 | Métrique principale pour une classe rare (prévalence 0,17 %). Une PR-AUC aléatoire vaudrait 0,0017. |
| Rappel au seuil retenu | ≥ 0,75 | Au moins 3 fraudes sur 4 détectées. En dessous, la perte évitée ne justifie pas le système. |
| Précision au seuil retenu | ≥ 0,30 | Au plus 7 alertes sur 10 à tort : charge supportable pour les analystes et friction client limitée. |
| FPR | ≤ 0,5 % | Au plus 1 client légitime sur 200 bloqué, soit 280 blocages à tort pour 56 000 transactions. |

**Choix du seuil.** Le seuil minimise sur la validation le coût `100 × FN + 5 × FP`. Le coût d'un FN (100 €) correspond à l'ordre de grandeur du montant moyen d'une fraude (122 € dans le jeu) majoré des frais de litige. Le coût d'un FP (5 €) correspond à une vérification par un analyste et à la friction pour le client. Ces coûts sont des **hypothèses à valider par le métier**.

**Règle de décision.**

* **Accepté pour la suite** : les 5 critères sont atteints, toutes les exigences T01 à T29 sont vertes et aucune anomalie majeure n'est ouverte.
* **Ajourné** : au moins un critère ML n'est pas atteint, ou une anomalie majeure est ouverte, mais le pipeline logiciel est conforme.
* **Rejeté** : une défaillance logicielle bloquante (fuite de données, prédictions non reproductibles, entrées invalides acceptées).

La réussite des tests techniques **n'autorise pas** le déploiement. Celui-ci exige en plus une validation métier, opérationnelle, sécuritaire et réglementaire.
