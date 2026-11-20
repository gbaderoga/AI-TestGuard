# 6. Stratégie de surveillance après déploiement (Partie X)

Le prototype fournit les briques de calcul (`app/monitoring.py`), testées par `tests/test_robustness.py`. Prometheus et Grafana ne sont pas déployés : les alertes sont simulées dans les tests et dans `results/robustness.json`.

## 6.1 Dérive des données ou dérive de concept

| | Dérive des données (*data drift*) | Dérive de concept (*concept drift*) |
|---|---|---|
| Définition | La distribution des entrées P(X) change | La relation P(fraude \| X) change : les mêmes variables ne signifient plus la même chose |
| Exemple | Inflation, nouvelle clientèle, saison, nouveau canal de paiement | Les fraudeurs changent de mode opératoire (nouveaux schémas de test de carte) |
| Détectable sans labels ? | **Oui**, immédiatement (PSI, statistiques des entrées) | **Non**. Il faut les labels (contestations, enquêtes), disponibles avec des semaines de retard. |
| Effet | Peut dégrader le modèle, ou rester sans effet | Dégrade le modèle, même si les entrées paraissent stables |

## 6.2 Indicateurs, fréquence et seuils

### Disponibles immédiatement (sans label)

| Indicateur | Calcul | Fréquence | Seuil d'alerte | Justification et origine |
|---|---|---|---|---|
| PSI par variable | `drift_report(reference=train, current=fenêtre)` | quotidien, fenêtre glissante de 7 jours | ≥ 0,10 à surveiller, **≥ 0,25 en alerte** | Seuils usuels. Priorité aux variables V4, V11, V12 et V14, dont le décalage fait chuter la précision de 0,77 à 0,56 (§ 3.2). |
| Médiane et P95 de `Amount` | comparaison à la référence | quotidien | variation > 5 % | Compense l'insensibilité du PSI aux changements d'échelle (A6) |
| **Taux de transactions signalées** | `alert_rate_check(n_alertes, n_total, taux_attendu)` | horaire | hors de ±3 σ binomiaux autour de 0,18 % | Taux attendu mesuré sur la validation : 103 alertes sur 56 745. C'est l'indicateur le plus rapide d'une dérive de score. |
| Distribution des scores | PSI sur les scores et part de scores > 0,99 | quotidien | PSI ≥ 0,25 | Détecte une dérive qui se répercute sur le modèle |
| Part d'alertes « hors domaine » | `out_of_domain_alert` | horaire | > 1 % des transactions | Référence : 8 sur 56 746 en test (0,014 %) |
| Taux de rejet des entrées | nombre de réponses 400 / total | temps réel | > 0,1 % | Erreur d'intégration chez l'appelant |
| Disponibilité de l'API | sondes de santé | 1 min | < 99,9 % sur 30 jours | — |
| Latence P95 de l'API | histogramme | temps réel | > 20 ms pendant 5 min | Critère ENF4 |

### Nécessitant les labels (validation différée)

| Indicateur | Source des labels | Délai typique | Seuil |
|---|---|---|---|
| Rappel et FNR | contestations des porteurs, enquêtes | 30 à 90 jours | rappel < 0,75 sur une fenêtre de 30 jours |
| Précision et FPR | retour des analystes sur les alertes | 1 à 3 jours (précision) ; FPR plus lent | précision < 0,30 ; FPR > 0,5 % |
| PR-AUC | échantillon labellisé complet | 30 à 90 jours | baisse > 0,02 par rapport à la baseline |
| Montant de fraude non détectée | comptabilité des litiges | mensuel | tendance |

La **précision** se mesure vite, car chaque alerte est examinée par un analyste. Le **rappel** est lent et biaisé : une fraude manquée n'est connue que si le porteur la conteste. Pour estimer le rappel sans attendre, on peut faire examiner chaque semaine un **petit échantillon aléatoire de transactions non signalées**.

## 6.3 Procédure de réponse

| Niveau | Déclencheur | Action |
|---|---|---|
| Information | PSI ≥ 0,10 sur une variable | Note dans le tableau de bord |
| Avertissement | PSI ≥ 0,25, ou taux d'alerte hors intervalle pendant 2 h | Analyse par l'équipe ML sous 48 h, comparaison avec le calendrier (fêtes, soldes) |
| Critique | Rappel ou précision labellisés sous le critère, ou taux d'alerte ×3 | Revue d'incident. Retour à la version précédente ou aux règles métier. Ré-entraînement **suivi d'une requalification complète** (même plan de test, nouvelle baseline uniquement après décision explicite). |

Un ré-entraînement n'est jamais déclenché automatiquement en production : il passe par la CI (tests rapides), la qualification complète et la comparaison à la baseline.

## 6.4 Simulation sur le prototype

| Situation simulée | Indicateur | Résultat |
|---|---|---|
| Données de test saines | PSI max 0,0005 ; 96 alertes (0,169 %, dans l'intervalle) | aucune alerte ✅ |
| V4, V11, V12 et V14 décalées de 1 σ | PSI 1,58 sur 4 variables ; 134 alertes (0,236 %) | alerte PSI ✅ ; taux d'alerte **juste** hors de l'intervalle ±3 σ [0,128 % ; 0,235 %] ✅. Sur une seule journée, le taux d'alerte seul serait un signal faible : il faut le croiser avec le PSI. |
| Montants × 1,5 | PSI de 0,15 sur `Amount` | à surveiller ✅ |
| Montants × 1,1 | PSI de 0,005 | **non détecté par le PSI**, détecté par le suivi de la médiane (+10 %) |
| Période postérieure (validation temporelle) | PSI de 12,4 sur `Time` | alerte permanente, ce qui confirme qu'il faut retirer `Time` (A2) |
