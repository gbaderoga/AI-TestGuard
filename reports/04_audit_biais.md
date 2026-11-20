# 4. Note d'audit des biais (Partie VII)

## 4.1 Conclusion

**Le jeu de données ne permet pas d'évaluer l'équité du système.** Aucune conclusion du type « le modèle est équitable » ne peut être tirée de cette qualification. L'exigence T30 est classée **non évaluable** : elle n'est ni réussie ni échouée.

## 4.2 Pourquoi l'analyse est impossible ici

| Variable | Nature | Utilisable pour un audit d'équité ? |
|---|---|---|
| `V1` à `V28` | Composantes d'une ACP sur des variables d'origine inconnues, publiées ainsi pour des raisons de confidentialité | **Non**. Une composante peut mélanger âge, pays, type de commerçant, etc. Elle ne représente aucun groupe interprétable. |
| `Time` | Secondes écoulées depuis la première transaction du jeu (2 jours, septembre 2013) | **Non**. Ce n'est pas un attribut de la personne. |
| `Amount` | Montant de la transaction | **Non**. C'est un attribut de la transaction, pas un groupe protégé. On peut au mieux l'utiliser comme indicateur indirect de pouvoir d'achat, ce qui n'est pas justifiable sans données. |
| `Class` | Étiquette de fraude | — |

Le jeu ne contient **ni âge, ni sexe, ni nationalité, ni localisation, ni identifiant client**. On ne peut donc pas calculer un rappel, un FPR ou une précision par groupe.

**Ce qui a été volontairement évité** : reconstituer des « groupes » à partir des composantes V (par exemple par clustering), ou inventer des attributs démographiques. Ce serait une analyse factice, sans fondement, qui pourrait donner une fausse assurance d'équité.

## 4.3 Ce qu'une analyse de sous-populations permet quand même (et ses limites)

On peut comparer les performances par **tranche de montant**, une dimension opérationnelle et non démographique. L'analyse des erreurs (`02_evaluation_ml.md` § 2.4) montre une disparité :

* les fraudes de **micro-montant** (< 4) représentent 13 des 21 fraudes manquées ;
* les **faux positifs** concernent surtout des achats **élevés** (montant moyen de 1 541 contre 88).

Si les clients à faible revenu ou les clients étrangers effectuent plus souvent des transactions atypiques, les blocages à tort pourraient les toucher de façon disproportionnée. C'est une **hypothèse**, invérifiable avec ces données, qui justifie d'en collecter de complémentaires.

Avec 95 fraudes en test, toute ventilation en sous-groupes produirait d'ailleurs des effectifs de quelques unités et des intervalles de confiance inexploitables.

## 4.4 Données complémentaires nécessaires

| Donnée | Usage | Précaution juridique |
|---|---|---|
| Tranche d'âge du titulaire | Rappel et FPR par tranche | Base légale (RGPD art. 6), minimisation : tranches et non date de naissance |
| Pays de résidence et pays du commerçant | Écarts entre clients domestiques et étrangers | Risque de discrimination indirecte selon l'origine |
| Segment client (particulier, professionnel, premium) et type de carte | Disparités liées au profil économique | — |
| Catégorie de commerçant (MCC) | FP concentrés sur certains secteurs | — |
| Canal (présentiel, e-commerce, sans contact) | Performance par canal | — |
| Résultat des contestations | Vraie étiquette des FP bloqués | — |

Le sexe et l'origine sont des données sensibles ou protégées. Leur usage, même à des fins d'audit, exige une base légale, une analyse d'impact (AIPD) et l'avis du DPO. Ces attributs ne doivent **jamais** devenir des variables du modèle.

## 4.5 Protocole recommandé lorsque les données seront disponibles

1. Sur un jeu de test indépendant, avec les attributs de groupe stockés **à part** du modèle, calculer pour chaque groupe : rappel, FPR, précision et taux d'alerte (`fairlearn.metrics.MetricFrame`).
2. Accompagner chaque écart d'un **intervalle de confiance bootstrap**. N'interpréter que les groupes ayant au moins 30 fraudes.
3. Critères proposés : écart de FPR (*equalized odds*) ≤ 0,1 point, et ratio de rappel entre groupes ≥ 0,8.
4. Si un écart est confirmé : analyser les causes (données, variables indirectes) avant toute correction (repondération, seuils par groupe, ce dernier point étant soumis à validation juridique).
5. Ajouter le suivi par groupe au tableau de surveillance.

## 4.6 Recommandations

* Ne pas présenter ce modèle comme « audité pour les biais ».
* Inscrire la collecte des données de la § 4.4 comme **prérequis à la phase pilote**.
* Pendant un pilote, journaliser (hors modèle) le segment client et le pays des transactions bloquées pour pouvoir mesurer les écarts de FPR.
