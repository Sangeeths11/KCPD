import streamlit as st
import geopandas as gpd
import pandas as pd
import folium
from streamlit_folium import st_folium
from folium.plugins import MousePosition, MarkerCluster
from shapely import wkt

@st.cache_data
def load_district_data():
    district_df = pd.read_csv("../data/city_council_districts/City_Council_Districts_Shapefile_-_Effective_2023_20241004.csv", low_memory=False)
    return district_df

@st.cache_data
def load_offenses():
    offenses = pd.read_csv("../data/mergedData/merged_df.csv", usecols=["Offense_Description"], low_memory=False)
    offenses = offenses["Offense_Description"].unique()
    return offenses

@st.cache_data
def load_crime_data(start_date=None, end_date=None, offense=None):
    crime_df = pd.read_csv("../data/mergedData/merged_df.csv", usecols=["Location", "Offense_Description", "Address", "Reported_Date", "dist_id"], low_memory=False)
    crime_df['Reported_Date'] = pd.to_datetime(crime_df['Reported_Date'], format='%m/%d/%Y')

    if start_date is not None and end_date is not None and offense is not None:
        start_date = pd.to_datetime(start_date, format='%m/%d/%Y')
        end_date = pd.to_datetime(end_date, format='%m/%d/%Y')
        crime_df = crime_df[(crime_df['Reported_Date'] >= start_date) & (crime_df['Reported_Date'] <= end_date)]
        crime_df = crime_df[(crime_df["Offense_Description"] == offense)]
    
    return crime_df

st.set_page_config(page_title="KCPD", page_icon="🌍", layout="wide")

offenses = load_offenses()

start_date = st.sidebar.date_input("Start date", value=pd.to_datetime("01/01/2024"))
end_date = st.sidebar.date_input("End date", value=pd.to_datetime("01/31/2024"))

offense = st.sidebar.selectbox("Select offense", options=offenses, index=0)

district_df = load_district_data()
crime_df = load_crime_data(start_date=start_date, end_date=end_date, offense=offense)

gdf = gpd.GeoDataFrame(district_df, geometry=gpd.GeoSeries.from_wkt(district_df["the_geom"]))
gdf['geometry'] = gdf['geometry'].simplify(tolerance=0.001)
gdf.set_crs(epsg=4326, inplace=True)

m = folium.Map(location=[39.075, -94.53], zoom_start=10)

colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF']

for i, row in gdf.iterrows():
    folium.GeoJson(row["geometry"],
                   name=row["DISTRICT_2"],
                   style_function=lambda x, color=colors[i % len(colors)]: {
                       'fillColor': color,
                       'color': 'grey',
                       'weight': 2,
                       'fillOpacity': 0.25
                   }
                   ).add_to(m)

crime_df['geometry'] = crime_df['Location'].apply(wkt.loads)
crime_gdf = gpd.GeoDataFrame(crime_df, geometry='geometry')
crime_gdf.set_crs(epsg=4326, inplace=True)

marker_cluster = MarkerCluster().add_to(m)

for idx, crime in crime_gdf.iterrows():
    lat = crime.geometry.y
    lon = crime.geometry.x
    description = f"<div style='width: 300px;'><b>{crime["Offense_Description"]}</b> at {crime['Address']} in district {crime['dist_id']}</div>"
    folium.Marker([lat, lon], popup=folium.Popup(description), icon=folium.Icon(color="red", icon="info-sign")).add_to(marker_cluster)

mouse_position = MousePosition(
    position="topright",
    separator=" | Long: ",
    prefix="Lat: ",
    num_digits=5,
)
m.add_child(mouse_position)

col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    st.title("KCPD - Time Series Data")
    st.subheader("District Map with Crime Data")
    st_data = st_folium(m, width=1100, height=700)
