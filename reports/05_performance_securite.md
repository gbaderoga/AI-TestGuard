# 5. Rapport de performance et de sécurité opérationnelle (Partie VIII)

Reproduction : `python scripts/benchmark.py` → `reports/results/benchmark.json`.

## 5.1 Protocole de mesure

| Élément | Valeur |
|---|---|
| Matériel | Intel Xeon @ 2,10 GHz, 2 cœurs logiques, 7,8 Go de RAM, conteneur Linux partagé |
| Logiciels | Python 3.13.18, scikit-learn 1.9.1, numpy 2.5.3, pandas 3.0.5 |
| Horloge | `time.perf_counter` |
| Préchauffage | 200 appels non mesurés (50 pour HTTP) |
| Latence unitaire | 2 000 répétitions, 1 transaction par appel (1 000 pour HTTP) |
| Débit | lots de 100, 1 000 et 10 000 transactions, 20 répétitions, médiane retenue |
| Mémoire | RSS du processus (psutil) avant et après chargement, pic échantillonné toutes les 2 ms pendant 5 lots de 56 746 lignes |
| Données | jeu synthétique au schéma réel : seule la forme des données influe sur le temps de calcul d'une régression logistique |

**Trois niveaux de latence sont distingués :**

1. **pipeline seul** : `predict_proba` de scikit-learn ;
2. **prédiction applicative** : `predict_fraud`, c'est-à-dire validation du contrat, pipeline et alerte hors domaine ;
3. **requête API** : serveur HTTP local (`http.server`), avec sérialisation JSON, transport HTTP, validation stricte de la requête, prédiction et réponse JSON. Le réseau réel, TLS, l'authentification et un serveur de production ne sont **pas** inclus.

## 5.2 Résultats

| Niveau (1 transaction) | P50 | **P95** | P99 | Max |
|---|---:|---:|---:|---:|
| Pipeline seul | 2,67 ms | 3,10 ms | 3,52 ms | 4,8 ms |
| `predict_fraud` (validation + modèle) | 7,82 ms | **9,66 ms** | 12,1 ms | 24,3 ms |
| Traitement de la requête, sans HTTP | 8,27 ms | 10,4 ms | 13,4 ms | 36,3 ms |
| **Requête HTTP locale** | 10,9 ms | **13,4 ms** | 16,1 ms | 37,3 ms |

| Taille du lot | Temps médian | Débit |
|---:|---:|---:|
| 100 | 7,6 ms | 13 079 tx/s |
| 1 000 | 8,8 ms | **114 196 tx/s** |
| 10 000 | 12,0 ms | 832 356 tx/s |

| Ressource | Mesure |
|---|---|
| Taille du modèle sauvegardé (joblib, compression 3) | **3,5 ko** |
| RSS après chargement | 183,9 Mo (dont 0,2 Mo dus au modèle ; le reste est l'environnement Python, numpy, pandas et scikit-learn) |
| RSS maximal pendant un lot de 56 746 transactions | **275 Mo** |

## 5.3 Seuils d'acceptation justifiés

| Critère | Seuil | Justification | Résultat |
|---|---|---|---|
| P95 d'une prédiction | ≤ 20 ms | Une autorisation carte doit aboutir en quelques centaines de ms de bout en bout, et le scoring de fraude n'est qu'une étape parmi d'autres (réseau, émetteur, 3-D Secure). Un budget de 20 ms laisse de la marge aux autres composants. | ✅ 9,7 ms (applicatif), 13,4 ms (HTTP local) |
| Débit | ≥ 5 000 tx/s par instance | Le jeu contient 284 807 transactions en 48 h, soit 1,6 tx/s en moyenne. Une banque de taille moyenne connaît des pointes de quelques milliers de tx/s : on vise une instance capable d'absorber une pointe seule. | ✅ 114 000 tx/s (lots de 1 000) ; ⚠️ en unitaire, ≈ 100 tx/s par fil d'exécution, voir ci-dessous |
| Taille du modèle | ≤ 5 Mo | Déploiement dans un conteneur léger et rechargement rapide | ✅ 3,5 ko |
| Mémoire | ≤ 500 Mo | Conteneur standard de 512 Mo à 1 Go | ✅ 275 Mo |

**Lecture.** La latence est dominée par la **validation pandas** (≈ 5 ms sur 7,8 ms), pas par le modèle (2,7 ms). Si les transactions arrivent une par une, une instance mono-fil traite environ 100 tx/s. Pour atteindre la cible en unitaire, il faut soit regrouper les requêtes en micro-lots de quelques millisecondes, soit plusieurs processus, soit une validation sans pandas (anomalie A9, mineure).

## 5.4 Tests de sécurité opérationnelle

| Contrôle | Test | Résultat |
|---|---|---|
| Champ manquant | `test_api_rejects_bad_fields[missing]` | 400 ✅ |
| Champ supplémentaire (ex. `iban`) | `[extra]` | 400 ✅ (l'API est stricte ; le pipeline par lots tolère et ignore) |
| Types incorrects : chaîne, booléen, null, liste | `[string]`, `[bool]`, `[null]`, `[list]` | 400 ✅ (le booléen n'est pas accepté comme 0/1) |
| `NaN` et `Infinity` en JSON (acceptés par défaut par le module `json`) | `test_api_rejects_nan_and_infinity_literals` | 400 ✅ |
| JSON invalide, structure inattendue, lot vide, octets non UTF-8 | `test_api_rejects_malformed_bodies` (6 cas) | 400 ✅ |
| Lot trop gros (> 1 000) et corps > 2 Mo | `test_api_rejects_oversized_batch`, serveur du benchmark | 400 / 413 ✅ |
| Montant négatif | `test_api_rejects_negative_amount` | 400 ✅ |
| Requête invalide via HTTP réel | `benchmark.py` | 400 ✅ |
| Aucune valeur de transaction dans les journaux | `test_logs_do_not_contain_transaction_values`, `test_extra_column…` | ✅ (seuls des compteurs et des noms de colonnes sont journalisés) |
| Pas de secrets dans le dépôt | `test_no_secrets_in_repository` (AWS, GitHub, clés privées, Kaggle, mots de passe) et test du détecteur lui-même | ✅ |
| Pas de données ni de fichiers > 5 Mo versionnés | `test_no_dataset_or_large_file_committed`, `.gitignore` | ✅ |
| Identifiants Kaggle de la CI | secrets GitHub `KAGGLE_USERNAME` et `KAGGLE_KEY`, jamais dans le dépôt | ✅ |
| Intégrité du jeu téléchargé en CI | `sha256sum -c` | ✅ |
| Intégrité du modèle | SHA-256 vérifié **avant** `joblib.load` (`load_model(expected_sha256=…)`) | ✅ (A11 corrigée) |
| Permissions de la CI | `permissions: contents: read` | ✅ |

## 5.5 Anomalies et limites

| ID | Constat | Gravité | Statut |
|---|---|---|---|
| A9 | La validation pandas représente environ 65 % de la latence unitaire | Mineure | Ouverte (optimisation possible) |
| A11 | `joblib` exécute du code au chargement : un modèle piégé pourrait compromettre le serveur | Majeure | **Corrigée** : vérification d'empreinte avant chargement. Reste à faire : stocker l'empreinte dans un registre de modèles signé. |
| — | Mesures faites sur un conteneur partagé à 2 cœurs. Il faut les refaire sur le matériel cible, avec un test de charge concurrent (Locust) une fois l'API déployée. | Limite | — |
| — | L'API de test n'a ni authentification, ni limitation de débit, ni TLS : ces éléments relèvent du déploiement | Hors périmètre | — |
