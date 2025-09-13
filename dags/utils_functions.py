import os
import requests
import pandas as pd
import numpy as np
import lightgbm as lgb
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from meteostat import Daily


def reformat_sentiweb_data(data, indicator):
    df = pd.DataFrame(data["data"])
    df["year"] = df["week"] // 100
    df["week"] = df["week"] % 100
    df["indicator"] = indicator
    return df

# ----------------------------
# Branche Meteo
# ----------------------------

def fetch_meteo_data(start, end, stations):
    """
    Récupère les données météo pour une liste de stations.
    Ignore les stations pour lesquelles aucune donnée n'est disponible ou le cache est corrompu.
    """
    dfs = []

    for station in stations:
        try:
            df = Daily(station, start, end).fetch()
            if not df.empty:
                df["station"] = station
                dfs.append(df)
            else:
                logging.warning(f"Aucune donnée météo pour la station {station} entre {start} et {end}.")
        except EOFError:
            logging.warning(f"Cache corrompu ou fichier vide pour la station {station}. Ignoré.")
        except Exception as e:
            logging.error(f"Erreur inattendue pour la station {station}: {e}")

    if dfs:
        return pd.concat(dfs)
    else:
        logging.warning("Aucune donnée météo disponible pour toutes les stations.")
        return pd.DataFrame()

def transform_meteo_data(start, end, regions_stations, region_geo_insee):
    """
    Traite les données météo récupérées pour toutes les régions et fait la moyenne par semaine de la température max/min, et des precipitations
    """
    all_regions = []

    for region, stations in regions_stations.items():
        df_region = fetch_meteo_data(start, end, stations)
        
        df_region["year_week"] = df_region.index.strftime("%Y-%U")
        df_weekly = (
            df_region.groupby("year_week")
            .agg({"tmin": "mean", "tmax": "mean", "prcp": "sum"})
            .reset_index()
        )
        df_weekly["region"] = region
        df_weekly["geo_insee"] = region_geo_insee[region]
        all_regions.append(df_weekly)

    df_final = pd.concat(all_regions, ignore_index=True)
    df_final["year"] = df_final["year_week"].str.split('-').str[0]
    df_final["week"] = df_final["year_week"].str.split('-').str[1].astype(int)

    return df_final
    
# Fonction qui crée des lag (lag 4)


def transform_data(df_senti,df_meteo):    
    for i in range(1,5):
        
        df_senti[f"lag_{i}"] = df_senti.groupby(["geo_insee","indicator"])["inc100"].shift(i)
    
    df_senti.dropna(inplace=True)
    df_merged = pd.merge(df_senti, df_meteo, on=["geo_insee","year","week"], how="outer")
    
    indicators = df_merged["indicator"].dropna().unique()
    def expand(row):
        if pd.isna(row["indicator"]):
            return pd.DataFrame([{**row.to_dict(), 'indicator': i} for i in indicators])
        return pd.DataFrame([row])
    df_expanded = pd.concat([expand(row) for _, row in df_merged.iterrows()], ignore_index=True)
    print(df_expanded)
    for feat in ["tmin","tmax","prcp"]:
        df_expanded = df_expanded.sort_values(by=["geo_insee","indicator","year","week"])
        df_expanded[f"next_week_{feat}"] = df_expanded.groupby(["geo_insee","indicator"])[feat].shift(-1)
	
    df_expanded.dropna(subset=["inc100"], inplace=True)
    
    return df_expanded
    
    
# ----------------------------
# Branche Training
# Entrainement avec LightGBM
# ----------------------------
def train_lightgbm(df,params):
		
    X = df.drop(columns=["inc100","region","year_week"], errors='ignore')
    y = df["inc100"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)
    cat_cols = ["geo_insee","week","indicator","geo_name"]
    for c in cat_cols:
        if c in X_train.columns:
            X_train[c] = X_train[c].astype("category")
            X_test[c] = X_test[c].astype("category")

    train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_cols)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
    model_lgbm= lgb.train(params, train_data, valid_sets=[test_data], num_boost_round=60)
    y_pred_lgbm = model_lgbm.predict(X_test)

    return X_train, X_test, y_train, y_test,model_lgbm, y_pred_lgbm
    
    
   
#Branche predict


def predict_df(df,model):
   X = df.drop(columns=["inc100","region","year_week"], errors='ignore')
   cat_cols = ["geo_insee","week","indicator","geo_name"]
   for c in cat_cols:
      if c in X.columns:
            X[c] = X[c].astype("category")

   df["predicted_inc100"] = model.predict(X)
   return df
