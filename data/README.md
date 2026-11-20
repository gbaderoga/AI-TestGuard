# Données

Le jeu de données **n'est pas versionné** : il pèse environ 150 Mo et est soumis à la licence de sa source. Le `.gitignore` exclut `data/*.csv`.

## Obtenir le jeu

* Source : Kaggle, *Credit Card Fraud Detection* (`mlg-ulb/creditcardfraud`), transactions de porteurs européens sur 2 jours de septembre 2013, publiées par le Machine Learning Group de l'ULB et Worldline.
* Téléchargement manuel : télécharger `creditcard.csv` et le placer ici, soit `data/creditcard.csv`.
* En ligne de commande, avec un jeton Kaggle dans `~/.kaggle/kaggle.json`, jamais dans le dépôt :

```bash
kaggle datasets download -d mlg-ulb/creditcardfraud -p data --unzip
```

## Vérification d'intégrité

```bash
sha256sum data/creditcard.csv
# 76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89
```

## Caractéristiques attendues (contrôlées par `test_real_dataset_contract`)

| Élément | Valeur |
|---|---|
| Lignes | 284 807 |
| Fraudes (`Class = 1`) | 492 (0,173 %) |
| Colonnes | `Time`, `V1` à `V28`, `Amount`, `Class` (31 colonnes) |
| Valeurs manquantes | 0 |
| Doublons exacts | 1 081, retirés avant la partition |
| `Time` | de 0 à 172 792 s (48 h) : secondes depuis la première transaction |
| `Amount` | de 0 à 25 691,16, médiane 22 |

`V1` à `V28` sont des composantes principales anonymisées : elles **ne doivent pas** être interprétées comme des caractéristiques métier.

## Données de test de la CI

La CI n'utilise pas ce fichier. Elle utilise des données **synthétiques** au même schéma, générées par `app/synthetic.py`, qui n'ont aucune valeur de qualification. Seul le job hebdomadaire `full-qualification` télécharge le vrai jeu, grâce aux secrets GitHub.
