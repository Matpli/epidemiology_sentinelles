import os
import requests
import pandas as pd
import numpy as np
import mlflow
import mlflow.lightgbm
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from meteostat import Daily
from io import StringIO
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.models import Variable
from datetime import datetime
from utils_functions import reformat_sentiweb_data, fetch_meteo_data, transform_meteo_data, transform_data, train_lightgbm, predict_df

# ----------------------------
# Paramètres globaux
# ----------------------------
AWS_CONN_ID = "aws_conn_id"
S3_BUCKET_NAME = Variable.get("S3BucketName")
MLFLOW_URI = "http://192.168.1.186:5000"
sentiweb_indicators = [6, 7, 25]

regions_stations = {
    "Île-de-France": ["07149", "07150"],
    "Hauts-de-France": ["07037", "07005"],
    "Normandie": ["07027", "07015"],
    "Grand Est": ["07190", "07145"],
    "Bretagne": ["07110", "07130"],
    "Pays de la Loire": ["07222", "07240", "07315"],
    "Centre-Val de Loire": ["07240", "07280"],
    "Bourgogne-Franche-Comté": ["07280", "07190"],
    "Nouvelle-Aquitaine": ["07510", "07315", "07610", "07335"],
    "Occitanie": ["07630", "07761", "07747", "07650", "07622"],
    "Auvergne-Rhône-Alpes": ["07481", "07486"],
    "Provence-Alpes-Côte d’Azur": ["07690", "07790", "07591"],
    "Corse": ["07761", "07785"],
}

region_geo_insee = {
    "Île-de-France": "11",
    "Hauts-de-France": "32",
    "Normandie": "28",
    "Grand Est": "44",
    "Bretagne": "53",
    "Pays de la Loire": "52",
    "Centre-Val de Loire": "24",
    "Bourgogne-Franche-Comté": "27",
    "Nouvelle-Aquitaine": "75",
    "Occitanie": "76",
    "Auvergne-Rhône-Alpes": "84",
    "Provence-Alpes-Côte d’Azur": "93",
    "Corse": "94",
}

# ----------------------------
# Config AWS
# ----------------------------
os.environ["AWS_ACCESS_KEY_ID"] = Variable.get("AWS_ACCESS_KEY_ID")
os.environ["AWS_SECRET_ACCESS_KEY"] = Variable.get("AWS_SECRET_ACCESS_KEY")

# ----------------------------
# Fonctions utilitaires S3
# ----------------------------
def upload_df_to_s3(df: pd.DataFrame, s3_key: str):
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    s3 = S3Hook(aws_conn_id=AWS_CONN_ID)
    s3.load_string(csv_buffer.getvalue(), key=s3_key, bucket_name=S3_BUCKET_NAME, replace=True)

def read_s3_csv(s3_key: str) -> pd.DataFrame:
    s3 = S3Hook(aws_conn_id=AWS_CONN_ID)
    content = s3.read_key(key=s3_key, bucket_name=S3_BUCKET_NAME)
    return pd.read_csv(StringIO(content), encoding='utf-8')

# ----------------------------
# Branche Sentiweb
# ----------------------------



def fetch_and_upload_sentiweb_data():
    df_list = []
    for indicator in sentiweb_indicators:
        url = f"https://www.sentiweb.fr/api/v1/datasets/rest/incidence?indicator={indicator}&geo=RDD&span=all&$format=json"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        df = reformat_sentiweb_data(data, indicator)
        df_list.append(df[["year", "week", "geo_insee", "geo_name", "inc100", "indicator"]])
    df_final = pd.concat(df_list, ignore_index=True)
    upload_df_to_s3(df_final, "donnees_sentinelles_brutes.csv")

# ----------------------------
# Branche Meteo
# ----------------------------
def fetch_and_upload_meteo_data():
    start = datetime(1990, 1, 1)
    end = datetime.today()
    all_regions = []
    df_final=transform_meteo_data(start, end, regions_stations, region_geo_insee)
    upload_df_to_s3(df_final, "donnees_meteo.csv")

# ----------------------------
# Branche Transformation
# ----------------------------
def transform_and_upload_data():
    df_senti = read_s3_csv("donnees_sentinelles_brutes.csv")
    df_meteo = read_s3_csv("donnees_meteo.csv")
    df_expanded=transform_data(df_senti,df_meteo)
    upload_df_to_s3(df_expanded, "donnees_lag_meteo.csv")

# ----------------------------
# Branche Training
# ----------------------------
def train_lightgbm_and_log():		
    df = read_s3_csv("donnees_lag_meteo.csv")
 

    params = {'objective':'regression','metric':'rmse','boosting_type':'gbdt',
              'learning_rate':0.1,'num_leaves':40,'max_depth':10,'feature_fraction':1}

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("light_gbm_sentinelles_Airflow")
    
    X_train, X_test, y_train, y_test,model_lgbm, y_pred_lgbm = train_lightgbm(df,params)

    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.lightgbm.autolog()
        model = model_lgbm
        y_pred = y_pred_lgbm
        mlflow.log_metric("MAE", mean_absolute_error(y_test, y_pred))
        mlflow.log_metric("RMSE", mean_squared_error(y_test, y_pred) ** 0.5)
        mlflow.log_metric("R2", r2_score(y_test, y_pred))
        mlflow.lightgbm.log_model(model, "model")

# ----------------------------
# Branche Prediction
# ----------------------------
def predict_and_upload():
    df_data = read_s3_csv("donnees_lag_meteo.csv")
    mlflow.set_tracking_uri(MLFLOW_URI)
    exp = mlflow.get_experiment_by_name("light_gbm_sentinelles_Airflow")
    runs = mlflow.search_runs([exp.experiment_id])
    best_run = runs.loc[runs["metrics.RMSE"].idxmin()]
    model_uri = f"runs:/{best_run.run_id}/model"
    model = mlflow.lightgbm.load_model(model_uri)
    df=predict_df(df_data,model)
    upload_df_to_s3(df,"predictions.csv")

