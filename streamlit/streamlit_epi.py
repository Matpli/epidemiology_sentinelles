import streamlit as st
import pandas as pd
import boto3
from io import StringIO
import geopandas as gpd
import folium
from streamlit_folium import folium_static
from botocore.exceptions import ClientError
import branca.colormap as cm
import matplotlib.pyplot as plt

# Dictionnaire associant les numéros d'indicateurs aux maladies
indicator_mapping = {
    6: "Acute Diarrhea",
    7: "Chickenpox",
    25: "Acute Respiratory Infection"
}

# 📌 Streamlit App Title
st.title("📊 French Epidemiological Incidence Rates")

# 📌 Introduction
st.markdown("""
### Welcome to the **Epidemiological Monitoring Dashboard** 🦠
This app monitores the  **weekly incidence rates** for three key diseases in France:
- **Acute Diarrhea** 
- **Chickenpox** 
- **Acute Respiratory Infection** (including influenza, Covid-19, VRS) 

Prediction are given for next week
The app is update every wenesday
""")

@st.cache_data 
def load_data():
    # Création d'un client S3 avec boto3
    s3_client = boto3.client('s3')
    bucket = 'sentinelles'
    #key = 'predicted_incidence.csv'
    key = 'predictions.csv'
     #  nom correspond exactement à l'objet dans S3
    
    # Récupération de l'objet depuis S3
    response = s3_client.get_object(Bucket=bucket, Key=key)
    # Lecture du contenu et décodage en texte
    data = response['Body'].read().decode('utf-8')
    
    # read dataframe
    df = pd.read_csv(StringIO(data))

    return df
# --- Chargement du shapefile des régions ---
@st.cache_data
def load_geojson():
    url = "https://raw.githubusercontent.com/gregoiredavid/france-geojson/master/regions.geojson"
    gdf = gpd.read_file(url)
    return gdf

# Load data
df_ = load_data()
df_["indicator"] = df_["indicator"].map(indicator_mapping)
df_predict = df_[["week", "year", "geo_insee", "indicator", "predicted_inc100"]]
last_year = df_["year"].max()
df_predict = df_predict[df_predict["year"] == last_year]
last_week = df_predict["week"].max()
df_predict = df_predict[df_predict["week"] == last_week]
df_predict.rename(columns={'predicted_inc100': 'inc100'}, inplace=True)
df_predict["week"] = last_week + 1
# Dernières lignes
last_rows = df_.tail(n)

# Affichage dans Streamlit
st.write(f"Les {n} dernières lignes du dataframe :")
st.dataframe(last_rows)

df = pd.concat([df_, df_predict])

geojson = load_geojson()

#display card for current year
df = df[df["year"] == 2025]

if not df.empty:
    # Utilisation de la barre latérale pour le menu
    with st.sidebar:
        # Menu déroulant pour sélectionner un indicateur
        selected_indicator = st.selectbox("Select disease :", df["indicator"].unique())

    # Filtrer sur l'indicateur sélectionné
    df_indicator = df[df["indicator"] == selected_indicator]

    # Créer dictionnaire week -> "YYYY-MM-DD - YYYY-MM-DD"
    week_dict = {}
    for week, year in zip(df_indicator['week'], df_indicator['year']):
        # 1er janvier de l'année
        jan1 = pd.Timestamp(year=int(year), month=1, day=1)
        # Lundi de la semaine contenant le 1er janvier (peut être dans l'année précédente)
        first_monday = jan1 - pd.Timedelta(days=jan1.weekday())
        # Début de la semaine sélectionnée
        start_date = first_monday + pd.Timedelta(days=(week-1)*7)
        # Fin de la semaine
        end_date = start_date + pd.Timedelta(days=6)
        week_dict[week] = f"{start_date.date()} - {end_date.date()}"

    # Créer liste déroulante avec les dates
    week_options = [week_dict[w] for w in sorted(week_dict.keys())]
    selected_week_str = st.selectbox("Select week (2025):", week_options, index=len(week_options)-1)

    # Retrouver la semaine correspondante à partir de la sélection
    selected_week = [w for w, d in week_dict.items() if d == selected_week_str][0]

    # Message si c'est la dernière semaine (prédiction)
    if selected_week == df_indicator['week'].max():
        st.markdown(f""" 
        <h4 style="text-align: center; font-weight: bold; color: #FFA500;">
            Predicted incidence for next week
        </h4>
        <p style="text-align: center; font-size: 16px; color: #888888;">
            This analysis was performed using <strong>LightGBM</strong>
        </p>""", 
        unsafe_allow_html=True
    )

    ########################### MAP ###############################

    # Filtrer les données pour la semaine sélectionnée
    df_week = df_indicator[df_indicator['week'] == selected_week]
    

    for region in df_week["geo_name"].unique():
        inc = df_week.loc[df_week["geo_name"] == region, "inc100"].values
        inc = inc[0] if len(inc) > 0 else 0
        if inc > 800:
            st.markdown(f"🚨 **Alert for {region}** with **{int(inc)}** cases")
    
    df_week["geo_insee"] = df_week["geo_insee"].astype(int)
    
    # Jointure avec le geojson (il faut s'assurer que les ID correspondent)
    geojson = geojson.rename(columns={"code": "geo_insee"})  # Adapter si besoin
    geojson["geo_insee"] = geojson["geo_insee"].astype(int)
    merged = geojson.merge(df_week, on="geo_insee", how="left")
    # Affichage de la légende personnalisée au-dessus de la carte
    st.markdown(
        f"""
        <h3 style="text-align: center; font-weight: bold; color: #4CAF50;">
            Number of cases for 100k inhabitants for <span style="color: #FF6347;">{selected_indicator}</span>
        </h3>
        """, 
        unsafe_allow_html=True
    )
     # --- Personalized scale ---

    colormap = cm.LinearColormap(
        colors=['yellow','red'],  
        vmin=0, vmax=1000
    )
    # --- Création de la carte ---
    m = folium.Map(location=[46.603354, 1.888334], zoom_start=5.6)

    folium.Choropleth(
        geo_data=merged,
        name="Choropleth",
        data=merged,
        columns=["geo_insee", "inc100"],
        key_on="feature.properties.geo_insee",
        fill_color="YlOrRd",
        fill_opacity=0.8,
        line_opacity=0.5,
        legend_name="Incidence (inc100)",
    ).add_to(m)

     # hoover
    folium.GeoJson(
        merged,
        name="Regions",
        style_function=lambda feature: {
            "fillColor": colormap(feature["properties"]["inc100"]) if feature["properties"]["inc100"] else "gray",
            "color": "black",
            "weight": 1,
            "fillOpacity": 0.8,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=["nom", "inc100", "lag_1"],  # Assurez-vous que "nom" est bien le nom de la région
            aliases=["Région :", "Incidence :", "Incidence week before:"],
            localize=True,
            sticky=False,
            labels=True,
            style="background-color: white; color: black; font-size: 12px; padding: 5px;",
        ),
    ).add_to(m)

    folium_static(m)  # Afficher la carte dans Streamlit
else:
    st.write("Données non disponibles")

################### Graphique ############################
st.markdown(
    """
    ## Welcome to the Incidence Trend Analysis!
    Explore the evolution of incidence for 100k inhabitants, comparing real values with predictions for the selected region.
    """
)



selected_region = st.selectbox("🌍 Select region :", df_["geo_name"].unique())

# Filtrage du DataFrame en fonction des sélections
df_pred = df_[["predicted_inc100", "year", "week", "inc100", "geo_name", "indicator"]]
df_pred = df_pred.loc[(df_pred["geo_name"] == selected_region) & (df_pred["indicator"] == selected_indicator)]
df_pred = df_pred.drop(["geo_name", "indicator"], axis=1)

# Création de la colonne 'date' à partir de 'year' et 'week'
df_pred['date'] = pd.to_datetime(
    df_pred['year'].astype(str) + df_pred['week'].astype(str).str.zfill(2) + '0',
    format='%Y%U%w'
)

df_pred = df_pred.drop(["year", "week"], axis=1)

# Trier par date et réinitialiser l'index
df_pred = df_pred.sort_values(by='date').reset_index(drop=True)

# S'assurer que les colonnes de valeurs sont en float
df_pred['inc100'] = df_pred['inc100'].astype(float)
df_pred['predicted_inc100'] = df_pred['predicted_inc100'].astype(float)

# Calcul de la nouvelle date (la dernière date + 7 jours) pour la prédiction de la semaine suivante
new_date = df_pred['date'].max() + pd.Timedelta(days=7)
new_row = pd.DataFrame({'date': [new_date], 'predicted_inc100': [None], 'inc100': [None]})
df_pred = pd.concat([df_pred, new_row], ignore_index=True)
df_pred['predicted_inc100'] = df_pred['predicted_inc100'].shift(1)

# Conversion des dates en objets date pour le slider
start_date = df_pred['date'].min().date()
end_date = df_pred['date'].max().date()

# Slider pour sélectionner une plage de dates
selected_dates = st.slider(
    "Select date range:",
    min_value=start_date,
    max_value=end_date,
    value=(start_date, end_date),
    format="YYYY-MM-DD"
)

# Filtrer le DataFrame selon la plage de dates sélectionnée
df_pred = df_pred[(df_pred['date'].dt.date >= selected_dates[0]) &
                  (df_pred['date'].dt.date <= selected_dates[1])]

# Création du graphique
plt.figure(figsize=(10, 6))
plt.plot(df_pred['date'], df_pred['inc100'], label='inc100')
# On trace la courbe prédite sans le dernier point (correspondant à la semaine suivante)
plt.plot(df_pred['date'][:-1], df_pred['predicted_inc100'][:-1], label='predicted_incidence_100')
plt.xlabel('Date')
plt.ylabel('Incidence per 100k inhabitants')
plt.title(f'Evolution incidence for 100k inhabitants in {selected_region} for indicator {selected_indicator} (predicted vs real)')
plt.xticks(rotation=45)
plt.grid(True)

# Vérifier que le DataFrame n'est pas vide et récupérer le dernier point
if not df_pred.empty:
    last_date = df_pred['date'].iloc[-1]
    last_predicted = df_pred['predicted_inc100'].iloc[-1]
    # On vérifie que la valeur prédite existe (n'est pas None ou NaN)
    if pd.notna(last_predicted):
        plt.scatter(last_date, last_predicted, color='red', marker='x', s=100, label='Next week predicted point')

plt.legend()
plt.tight_layout()
st.pyplot(plt)