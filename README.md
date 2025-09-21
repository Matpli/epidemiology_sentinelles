
# 🦠 Epidemiology Sentinelles – French Epidemiological Dashboard

Ce projet fournit un **pipeline complet de traitement de données épidémiologiques et météorologiques** pour la France. Il inclut l’ingestion des données, leur transformation, l’entraînement d’un modèle LightGBM, la prédiction des incidences, et une visualisation interactive via Streamlit.

---

## Construcutin de l'environement 


## 🐳 Docker Compose

Permet de lancer un environnement complet local :

- Airflow (Webserver, Scheduler, Workers)  
- MLflow (tracking & modèles)  
- PostgreSQL (backend Airflow & MLflow)  


Démarrage :

```bash
docker-compose up --build -d
```
---

## Construction de l'environement 

## 📂 Contenu du dépôt

| Fichier / Dossier                          | Rôle principal                                                                 | DAG(s) défini(s) |
|--------------------------------------------|-------------------------------------------------------------------------------|------------------|
| `dags/processing.py`                       | Fonctions de traitement : ingestion Sentiweb/météo, transformation, entraînement, prédictions, gestion S3/MLflow | — |
| `dags/orchestration.py`                    | Orchestration Airflow du pipeline complet                                     | `global_pipeline_complete_s3_auto` |
| `tests/test_function_integration.py`       | Tests d’intégration des fonctions clés de `processing.py`                     | — |
| `streamlit/streamlit_epi.py`               | Tableau de bord interactif Streamlit pour visualiser les incidences et prédictions | — |
| `docker-compose.yaml`                      | Déploiement de l’environnement local avec Airflow, MLflow, MinIO, PostgreSQL et Redis | — |
| `requirements.txt`                         | Dépendances Python nécessaires au projet                                      | — |

---

## ⚙️ Pipeline principal – `global_pipeline_complete_s3_auto`

Le DAG Airflow orchestre le pipeline complet :

1. **Ingestion Sentiweb** → téléchargement API Sentiweb → S3  
2. **Ingestion Météo (Meteostat)** → calcul des moyennes hebdomadaires → S3  
3. **Transformation** → fusion Sentiweb + météo + création de lags → S3  
4. **Entraînement LightGBM** → suivi via MLflow (MAE, RMSE, R², modèles)  
5. **Prédiction** → génération des prédictions → S3  

```r
# Schéma des dépendances
# Sentiweb branch ─┐  
#                  ├─> Transformation ─> Training ─> Prediction  
# Météo branch  ───┘  
```
🔧 Configuration

Définir les variables Airflow :

S3BucketName

AWS_ACCESS_KEY_ID

AWS_SECRET_ACCESS_KEY

Configurer l’URI MLflow : MLFLOW_URI

Vérifier les identifiants AWS pour boto3

🧪 Tests

Fichier : tests/test_function_integration.py

Vérifie ingestion, transformation, entraînement et prédiction
Lancer les tests :

```bash
pytest tests/
```
---
🌐 Streamlit Dashboard – streamlit_epi.py

Ce script Streamlit fournit un tableau de bord interactif pour visualiser les incidences épidémiologiques hebdomadaires en France, ainsi que les prédictions générées par le modèle LightGBM.

## 🔗 dashboard

Le dashboard est accessible publiquement ici :  
[https://epidemiology-sentinelles.streamlit.app/](https://epidemiology-sentinelles.streamlit.app/)

Fonctionnalités principales
1. Sélection de maladies

Acute Diarrhea

Chickenpox

Acute Respiratory Infection

2. Affichage des prédictions par semaine

Affiche les valeurs réelles et prévues (inc100 et predicted_inc100)

Message d’alerte si une région dépasse un seuil critique

3. Carte interactive des régions

Utilisation de Folium et GeoPandas pour visualiser les données par région

Couleurs dégradées selon l’incidence

Infobulles affichant les valeurs actuelles et les lags

4. Graphique des tendances

Visualisation de l’évolution temporelle des incidences

Comparaison entre valeurs réelles et prédictions

Marquage du point prédictif pour la semaine suivante

Source des données

Données prédites et historiques stockées sur AWS S3 (predictions.csv)

Shapefile des régions récupéré depuis GitHub

Lancement du dashboard :
```r
streamlit run streamlit/streamlit_epi.py
```

---
