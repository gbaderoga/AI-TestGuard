# Rapport final de qualification : AI-TestGuard

| | |
|---|---|
| **Objet** | Qualification du pipeline de détection de fraude par carte bancaire, version 1.0.0 |
| **Modèle** | Régression logistique (C = 0,01, `class_weight=balanced`), seuil 0,9994, SHA-256 `951e69d6bbc8…` |
| **Données** | Kaggle *Credit Card Fraud Detection* : 284 807 transactions, 492 fraudes (0,17 %), 2 jours de septembre 2013 |
| **Date** | 20 novembre 2026 |
| **Décision** | **AJOURNÉ : pipeline logiciel conforme, modèle non apte à la phase pilote en l'état** (voir § 10) |

---

## 1. Contexte et périmètre

Une banque veut savoir si un modèle de détection de fraude peut passer en phase pilote. La qualification porte sur le **système complet** : contrat de données, prétraitement, modèle, choix du seuil, prédiction, persistance, interface de type API et automatisation. Le score seul ne suffit pas.

Hors périmètre : infrastructure de production, authentification, processus des analystes et conformité réglementaire. Ces points relèvent d'autres validations (§ 10.3).

## 2. Plan de test et traçabilité

Le plan complet est dans `01_plan_de_test.md`. Il couvre 6 exigences fonctionnelles et 7 non fonctionnelles, et 30 cas de test : T01 à T07 imposés par le sujet, **T08 à T30 ajoutés**, chacun relié au risque qu'il maîtrise.

| Résultat | Nombre | Identifiants |
|---|---:|---|
| ✅ Conforme | 26 | T01 à T12, T14 à T27 |
| ❌ Non conforme | 2 | **T13** (critères ML sur le test), **T29** (validation temporelle) |
| ✅ / ❌ partiel | 1 | T28 : la régression logistique est conforme, l'arbre (écarté) ne l'est pas |
| ⚠️ Non évaluable | 1 | T30 : équité, faute d'attributs de groupe |

T07 est conforme en local. La preuve sur GitHub reste à produire par l'étudiant au moment du dépôt (`07_ci_github_actions.md`).

## 3. Architecture du pipeline et choix méthodologiques

```
CSV ──► load_data ──► validate_data ──► split_data ──────────────────────┐
        (en-tête      (schéma, types,    (déduplication, puis             │
         dupliqué)     NaN/inf, bornes)   stratification 60/20/20)        │
                                                                          ▼
           ┌──────────── Pipeline scikit-learn (ajusté sur l'ENTRAÎNEMENT) ──────────┐
           │ ColumnTransformer : Amount → log1p → StandardScaler                     │
           │                     Time, V1..V28 → StandardScaler                     │
           │ Classifieur : LogisticRegression(class_weight="balanced")               │
           └─────────────────────────────────────────────────────────────────────────┘
                     │ hyperparamètres et seuil choisis sur la VALIDATION
                     ▼
     ModelBundle {pipeline, seuil, domaine min/max, métadonnées} ──► joblib + SHA-256
                     │
                     ▼
 predict_fraud / handle_request (API) ──► {probabilité, décision, alerte hors domaine}
```

| Choix | Raison |
|---|---|
| Déduplication avant la partition | 1 081 doublons exacts : sans elle, des transactions identiques se retrouveraient en entraînement et en test (fuite). |
| Transformations dans un `Pipeline` | Garantit qu'elles ne sont ajustées que sur l'entraînement. C'est vérifié par un test sur `scaler.mean_`. |
| `log1p(Amount)` | Montants très asymétriques (médiane 22, maximum 25 691). Les montants extrêmes restent finis. |
| Rejet des NaN et inf plutôt qu'imputation | Imputer reviendrait à noter une transaction différente de celle qui a été reçue. |
| Seuil par minimisation de `100·FN + 5·FP` | Le seuil 0,5 donnerait 1 479 fausses alertes sur la validation. |
| Sélection par Average Precision | C'est la métrique adaptée à une classe rare (prévalence 0,17 %). |
| Critères fixés **avant** le test | Évite d'ajuster les critères après coup. |

## 4. Résultats des tests unitaires et d'intégration

**156 tests, 156 réussis, couverture de 98 %** (`08_execution_tests.md`), Ruff sans avertissement.

* Contrat de données : 48 tests (colonnes absentes ou dupliquées, types, NaN, inf, valeurs vides, cible hors {0, 1}, une seule observation, dimensions), plus des propriétés Hypothesis.
* Intégration : chaîne complète CSV → rapport, prédictions identiques après rechargement (écart ≤ 1e-9), scaler ajusté sur l'entraînement seul, seuil issu de la validation, entraînement reproductible, erreurs de chargement explicites, empreinte vérifiée.
* Une anomalie a été trouvée par les tests et corrigée (A10).

## 5. Métriques de performance et matrices de confusion

Détails dans `02_evaluation_ml.md`. Valeurs sur le test (56 746 transactions, 95 fraudes), avec IC bootstrap à 95 %.

| | Régression logistique (retenue) | Arbre de décision | Critère |
|---|---:|---:|---:|
| Précision | **0,771** [0,70 – 0,85] | 0,782 | ≥ 0,30 ✅ |
| Rappel | **0,779** [0,70 – 0,86] | 0,716 | ≥ 0,75 ✅ |
| F1 | 0,775 | 0,747 | — |
| ROC-AUC | **0,961** | 0,893 | ≥ 0,95 ✅ |
| **PR-AUC** | **0,681** [0,57 – 0,79] | 0,628 | **≥ 0,70 ❌** |
| FPR | 0,039 % | 0,034 % | ≤ 0,5 % ✅ |
| Matrice de confusion (TN / FP / FN / TP) | 56 629 / 22 / 21 / 74 | 56 632 / 19 / 27 / 68 | — |

Un modèle trivial « tout normal » obtiendrait 99,83 % d'accuracy pour un rappel nul : l'accuracy n'est pas un indicateur de qualité ici.

## 6. Robustesse et non-régression

Détails dans `03_robustesse_non_regression.md`.

* **Entrées invalides** : 10 scénarios sur 10 rejetés avec un message explicite. La colonne en trop est ignorée avec un avertissement.
* **Montants extrêmes** (jusqu'à 1e9) : probabilités finies, alerte hors domaine levée. Le modèle est peu sensible au montant.
* **Bruit** sur V1 à V28 (σ de 1 % à 25 % de l'écart-type), sur `Amount` (±5 %) et sur `Time` (±1 h) : aucun changement de métrique jusqu'à σ = 10 %, 3 changements de classe à σ = 25 %. Les décisions sont stables parce que très peu de transactions sont proches du seuil.
* **Dérive** : le PSI détecte les décalages des composantes V. Il **ne détecte pas** une hausse de 10 % des montants (A6).
* **Validation temporelle** : rappel de **0,689** (< 0,75) sur les dernières heures. PSI de 12,4 sur `Time` (A2, A3).
* **Non-régression** : baseline v1.0 figée. 4 versions candidates comparées : 2 acceptées (dont le contrôle de déterminisme) et 2 rejetées. L'échec volontaire est détecté.

## 7. Latence et ressources

Détails dans `05_performance_securite.md`.

| Indicateur | Mesure | Critère |
|---|---:|---:|
| P95 d'une prédiction (validation + modèle) | 9,7 ms | ≤ 20 ms ✅ |
| P95 d'une requête HTTP locale | 13,4 ms | ≤ 20 ms ✅ |
| Débit (lots de 1 000) | 114 000 tx/s | ≥ 5 000 ✅ |
| Taille du modèle | 3,5 ko | ≤ 5 Mo ✅ |
| Mémoire maximale | 275 Mo | ≤ 500 Mo ✅ |

## 8. Anomalies

| ID | Description | Gravité | Statut | Action recommandée |
|---|---|---|---|---|
| **A1** | PR-AUC de test de 0,681, sous le critère de 0,70. Écart important entre validation et test (0,791 → 0,681). | **Majeure** | Ouverte | Évaluer d'autres modèles (gradient boosting, calibration) par validation croisée répétée, puis sur des **données nouvelles** |
| **A2** | `Time` = secondes depuis le début de la collecte : variable non stationnaire, hors domaine en production (PSI 12,4) | **Majeure** | Ouverte | Supprimer `Time`, ou le remplacer par l'heure de la journée, puis requalifier |
| **A3** | Validation temporelle : rappel de 0,689 < 0,75 | **Majeure** | Ouverte | Liée à A1 et A2. Adopter la partition temporelle comme protocole de référence. |
| A4 | Scores non calibrés (seuil à 0,9994) : une « probabilité » trompeuse pour les analystes | Mineure | Ouverte | Calibrer (`CalibratedClassifierCV` sur la validation) ou présenter un score et non une probabilité |
| A5 | Une ligne invalide fait rejeter tout le lot | Mineure | Ouverte | Rejet ligne par ligne en mode lot |
| A6 | Le PSI ne détecte pas une hausse de 10 % des montants | Mineure | Ouverte | Suivre aussi la médiane et le P95 de `Amount` |
| A7 | Alerte hors domaine fondée sur le min/max : règle grossière | Mineure | Ouverte | Bornes par quantiles (0,1 % – 99,9 %) |
| A8 | Arbre de décision en surapprentissage (écart d'AP de 0,13) | Info | Close (modèle écarté) | — |
| A9 | Validation pandas ≈ 65 % de la latence unitaire | Mineure | Ouverte | Validation vectorisée sans pandas pour l'API |
| A10 | `UnicodeDecodeError` brute sur un CSV non UTF-8 | Mineure | **Corrigée** | Test de non-régression ajouté |
| A11 | Désérialisation joblib non vérifiée (risque d'exécution de code) | Majeure | **Corrigée** | Vérification SHA-256 avant chargement ; registre signé à prévoir |

## 9. Limites de l'évaluation et risques résiduels

1. **Faible nombre de fraudes** : 95 en test. Les IC à 95 % du rappel (± 8 points) et de la PR-AUC (± 11 points) sont plus larges que les écarts que l'on cherche à mesurer. Le critère de PR-AUC est manqué par une estimation ponctuelle dont l'intervalle contient 0,70 : **l'échec n'est pas statistiquement certain, mais la réussite ne l'est pas davantage**.
2. **Données de 2013, sur 2 jours, d'un seul émetteur européen** : elles ne représentent ni les fraudes actuelles, ni les cycles hebdomadaires ou saisonniers.
3. **Variables anonymisées par ACP** : on ne peut ni interpréter les erreurs métier par métier, ni vérifier que le prétraitement en production reproduira la même ACP. Cette transformation est hors de notre contrôle et constitue un risque d'intégration majeur.
4. **Équité non évaluable** (`04_audit_biais.md`).
5. **Coûts FN et FP supposés** (100 € et 5 €) : ils doivent être validés par le métier, car le seuil en dépend.
6. **Robustesse adverse non testée** : un fraudeur qui s'adapte n'est pas simulé.
7. **Performance mesurée** sur un conteneur partagé, sans test de charge concurrent.
8. **Jeu de test consommé** : tout modèle corrigé doit être évalué sur des données nouvelles.

## 10. Décision

### 10.1 Verdict technique

| Volet | Verdict |
|---|---|
| Pipeline logiciel : contrat de données, prévention des fuites, reproductibilité, persistance, API, sécurité, CI | ✅ **Conforme** |
| Performance d'exécution | ✅ **Conforme** |
| Qualité prédictive par rapport aux critères fixés à l'avance | ❌ **Non conforme** (PR-AUC 0,681 < 0,70 ; rappel temporel 0,689 < 0,75) |
| Anomalies majeures ouvertes | ❌ 3 (A1, A2, A3) |

### 10.2 Décision : **AJOURNÉ**

Selon la règle définie dans le plan de test (§ 1.6), le modèle v1.0 **n'est pas apte à poursuivre vers une phase pilote en l'état**. Ce n'est pas un rejet définitif. Le pipeline est sain et réutilisable, et l'écart au critère est faible et incertain. Mais déployer un modèle qui manque un critère fixé à l'avance, et dont une variable (`Time`) n'a pas de sens en production, ne serait pas défendable.

**Conditions de reprise** :

1. Corriger A2 (retrait ou transformation de `Time`) et A4 (calibration).
2. Sélectionner le modèle par validation croisée temporelle sur les données d'entraînement et de validation existantes.
3. Évaluer **une seule fois** sur un jeu jamais observé : données plus récentes ou période d'observation en mode « ombre » (*shadow mode*), où le modèle note les transactions sans bloquer.
4. Rejouer la totalité du plan de test. Atteindre les 5 critères ML et le rappel ≥ 0,75 en validation temporelle.
5. Figer une nouvelle baseline v2.0 par décision explicite.

Un fonctionnement **en mode ombre**, sans aucune décision automatique sur les clients, peut commencer dès maintenant pour collecter ces données. Il ne constitue pas un déploiement.

### 10.3 Réussir les tests techniques n'est pas une autorisation de déploiement

Même après une requalification réussie, le déploiement exigera :

* **une validation métier** : coûts FN et FP, capacité des analystes (environ 0,17 % des transactions signalées), parcours client en cas de blocage ;
* **une validation opérationnelle** : intégration avec le système d'autorisation, reproduction de la transformation ACP en amont, astreinte, plan de retour arrière ;
* **une validation sécurité** : authentification de l'API, TLS, registre de modèles signé, gestion des secrets, tests d'intrusion ;
* **une validation réglementaire et conformité** : RGPD (AIPD, décision automatisée au sens de l'article 22), exigences prudentielles sur les modèles, documentation du règlement européen sur l'IA si le système y est soumis, audit d'équité sur des données adaptées.

---

**Annexes** : `01_plan_de_test.md` · `02_evaluation_ml.md` · `03_robustesse_non_regression.md` · `04_audit_biais.md` · `05_performance_securite.md` · `06_surveillance.md` · `07_ci_github_actions.md` · `08_execution_tests.md` · `results/*.json` · `figures/*.png`
