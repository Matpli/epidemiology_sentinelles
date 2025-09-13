import pandas as pd
from dags.utils_functions import reformat_sentiweb_data, fetch_meteo_data, transform_meteo_data, transform_data,train_lightgbm, predict_df
import pytest
import lightgbm as lgb
from sklearn.model_selection import train_test_split


def test_reformat_sentiweb_data():
    # Données simulées
    mock_data = {
        "data": [
            {"week": 202501, "geo_insee": "12345", "geo_name": "TestVille", "inc100": 10},
            {"week": 202502, "geo_insee": "12346", "geo_name": "TestVille2", "inc100": 20},
        ]
    }
    indicator = 6

    # Appel de la fonction
    df = reformat_sentiweb_data(mock_data, indicator)

    # Vérifier les colonnes
    expected_columns = {"week", "year", "geo_insee", "geo_name", "inc100", "indicator"}
    assert set(df.columns) == expected_columns

    # Vérifier le calcul de year et week pour la première ligne
    first_row = df.iloc[0]
    assert first_row["year"] == 2025
    assert first_row["week"] == 1
    assert first_row["geo_insee"] == "12345"
    assert first_row["indicator"] == indicator

    # Vérifier la seconde ligne
    second_row = df.iloc[1]
    assert second_row["year"] == 2025
    assert second_row["week"] == 2
    assert second_row["geo_insee"] == "12346"
    assert second_row["indicator"] == indicator
    



# --- Fake DataFrame pour simuler Daily().fetch ---
def fake_fetch(station, start, end):
    dates = pd.date_range(start, end, freq="D")
    return pd.DataFrame({
        "tmin": [10 + i % 5 for i in range(len(dates))],
        "tmax": [20 + i % 5 for i in range(len(dates))],
        "prcp": [i % 3 for i in range(len(dates))]
    }, index=dates)


# Test meteo 

def test_fetch_meteo_data_unit(monkeypatch):
    """Test mocké : vérifier que fetch_meteo_data concatène bien les données."""
    class FakeDaily:
        def __init__(self, station, start, end):
            self.station = station
            self.start = start
            self.end = end
        def fetch(self):
            return fake_fetch(self.station, self.start, self.end)

    monkeypatch.setattr("dags.utils_functions.Daily", FakeDaily)

    stations = ["station1", "station2"]
    df = fetch_meteo_data("2023-01-01", "2023-01-07", stations)

    assert isinstance(df, pd.DataFrame)
    assert "station" in df.columns
    assert set(df["station"].unique()) == set(stations)



def test_transform_data_no_error():
    # -------------------------------
    # Créer df_senti factice avec 10 semaines pour chaque geo_insee / indicator
    # -------------------------------
    data_senti = []
    for geo in ["001", "002"]:
        for ind in ["A", "B"]:
            for week in range(1, 11):  # 10 semaines
                data_senti.append({
                    "geo_insee": geo,
                    "indicator": ind,
                    "year": 2023,
                    "week": week,
                    "inc100": week + ord(ind) % 5  # juste des valeurs factices
                })
    df_senti = pd.DataFrame(data_senti)

    # -------------------------------
    # df_meteo factice avec les mêmes semaines et geo_insee
    # -------------------------------
    data_meteo = []
    for geo in ["001", "002"]:
        for week in range(1, 11):  # mêmes semaines
            data_meteo.append({
                "geo_insee": geo,
                "year": 2023,
                "week": week,
                "tmin": week,
                "tmax": week + 10,
                "prcp": week % 3
            })
    df_meteo = pd.DataFrame(data_meteo)

    # -------------------------------
    # Appel de la fonction
    # -------------------------------
    df_transformed = transform_data(df_senti.copy(), df_meteo.copy())

    # Vérification minimale : pas de KeyError, output non vide
    assert not df_transformed.empty
    assert "geo_insee" in df_transformed.columns
    assert "indicator" in df_transformed.columns
    assert "inc100" in df_transformed.columns
  # -------------------------------
    # Vérification lag_4
    # -------------------------------
    # Exemple : geo_insee="001", indicator="A", semaine 5 → lag_4 = semaine 1
    row = df_transformed[
        (df_transformed["geo_insee"] == "001") &
        (df_transformed["indicator"] == "A") &
        (df_transformed["week"] == 5)
    ]
    # inc100 semaine 1 = 1 + ord("A")%5 = 1 + 0 = 1
    assert row["lag_4"].values[0] == 1


def test_train_lightgbm_with_lags():
    # -------------------------------
    # DataFrame factice
    # -------------------------------
    data = []
    for geo in ["001", "002"]:
        for ind in ["A", "B"]:
            for week in range(1, 11):  # 10 semaines
                data.append({
                    "geo_insee": geo,
                    "week": week,
                    "indicator": ind,
                    "geo_name": f"Region_{geo}",
                    "inc100": week + ord(ind) % 5,
                    "feature1": week*2,
                    "feature2": week*3
                })
    df = pd.DataFrame(data)

    # -------------------------------
    # Création des lags inc100
    # -------------------------------
    df = df.sort_values(by=["geo_insee", "indicator", "week"])
    for i in range(1, 5):
        df[f"lag_{i}"] = df.groupby(["geo_insee", "indicator"])["inc100"].shift(i)
    df.dropna(subset=[f"lag_{i}" for i in range(1,5)], inplace=True)

    # -------------------------------
    # Paramètres LightGBM factices
    # -------------------------------
    params = {
        "objective": "regression",
        "metric": "rmse",
        "verbose": -1
    }

    # -------------------------------
    # Appel de la fonction
    # -------------------------------
    X_train, X_test, y_train, y_test, model_lgbm, y_pred_lgbm = train_lightgbm(df, params)

    # -------------------------------
    # Vérifications
    # -------------------------------
    assert not X_train.empty
    assert not X_test.empty
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)
    assert len(y_pred_lgbm) == len(y_test)
    assert isinstance(model_lgbm, lgb.Booster)

    # Vérifier que les lags sont présents dans X_train
    for i in range(1,5):
        assert f"lag_{i}" in X_train.columns
        assert f"lag_{i}" in X_test.columns




class DummyModel:
    """Un modèle factice pour les tests"""
    def predict(self, X):
        # On renvoie simplement une liste de valeurs en fonction du nombre de lignes
        return [42] * len(X)

def test_predict_df():
    # DataFrame de test
    df_input = pd.DataFrame({
        "geo_insee": ["12345", "12346"],
        "week": [1, 2],
        "indicator": [6, 7],
        "geo_name": ["TestVille", "TestVille2"],
        "inc100": [10, 20],
        "region": ["Occitanie", "Occitanie"],
        "year_week": [202501, 202502],
    })

    model = DummyModel()

    # Appel de la fonction
    df_result = predict_df(df_input.copy(), model)

    # Vérifier que la colonne 'predicted_inc100' a été ajoutée
    assert "predicted_inc100" in df_result.columns

    # Vérifier les valeurs prédites
    assert all(df_result["predicted_inc100"] == 42)

    # Vérifier que les colonnes originales sont encore présentes
    assert "inc100" in df_result.columns
    assert "region" in df_result.columns
    assert "year_week" in df_result.columns

    

  

