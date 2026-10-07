# TP_02 : rendu

Renommez ce fichier en `TP02_NOM_Prenom.md` (ou exportez-le en `TP02_NOM_Prenom.pdf`) avant de le déposer sur e-campus. Pour les sorties de terminal, collez le texte dans les blocs de code ou ajoutez une capture d'écran.

## Identité

- **Nom :** NOM Prénom
- **Environnement :** OS, version de Docker, version de scikit-learn utilisée au TP_01
- **SHA-256 de `TP_01_model.skops` :** `...`

## 1. Lancer l'API

Commandes `docker build` et `docker run` utilisées (avec `MODEL_SHA256` défini) :

```bash

```

## 2. Appels à l'API

Sortie de `curl http://localhost:8000/health` :

```

```

Sortie de la requête `/predict` avec `request_example.json` :

```

```

## 3. Tests

Sortie de `pytest`, lancé depuis `TP_02/` avec votre modèle dans `app/models/` :

```

```

## 4. Exercices

### Exercice 1 : MODEL_SHA256 incorrect

Ce qui se passe (commande et dernière ligne des logs) :

Pourquoi il vaut mieux refuser de démarrer :

### Exercice 2 : `households = 0`

A (notebook TP_01), type d'erreur et message :

B (API Docker), code HTTP et message :

Résultat de `total_rooms / households` et étape du pipeline qui échoue :

Partie de `app/` qui arrête la requête, et pourquoi c'est le bon endroit :

### Exercice 3 : utilisateur du conteneur

Sortie de `docker exec <container> id` et de la tentative de modification de `/app/main.py` :

```

```

Pourquoi c'est important :

### Exercice 4 : avant d'exposer l'API sur Internet

Ce qui manque encore :

## 5. Code modifié (facultatif)

Si vous avez modifié le code, lien vers votre fork ou nom du zip joint (`TP02_NOM_Prenom_code.zip`). N'y mettez pas le fichier du modèle.
