import streamlit as st
import geopandas as gpd
import pandas as pd
import folium
from streamlit_folium import st_folium
from folium.plugins import MousePosition

@st.cache_data
def load_district_data():
    district_df = pd.read_csv("../data/city_council_districts/City_Council_Districts_Shapefile_-_Effective_2023_20241004.csv")
    return district_df

@st.cache_data
def load_crime_data():
    district_df = pd.read_csv("../data/mergedData/merged_df.csv")
    return district_df

# Set Streamlit page layout
st.set_page_config(page_title="KCPD", page_icon="🌍", layout="wide")

# Data preparation
district_df = load_district_data()
gdf = gpd.GeoDataFrame(district_df, geometry=gpd.GeoSeries.from_wkt(district_df["the_geom"]))  # Create a GeoDataFrame
gdf.set_crs(epsg=4326, inplace=True)  # Set the correct Coordinate Reference System (CRS)

m = folium.Map(location=[39.075, -94.53], zoom_start=10)  # Create a Folium map centered around the district's geometry

# List of colors for districts
colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF']

# Add district polygons to the map
for i, row in gdf.iterrows():
    folium.GeoJson(row["geometry"],
                   name=row["DISTRICT_2"],
                   style_function=lambda x,
                   color=colors[i % len(colors)]: {
                    'fillColor': color,  # Set the fill color for the district
                    'color': 'grey',  # Border color
                    'weight': 2,
                    'fillOpacity': 0.25  # Transparency level
                   }
                   ).add_to(m)

# Add Mouse Position Plugin to display coordinates on hover
mouse_position = MousePosition(
    position="topright",  # Position of the coordinates display on the map
    separator=" | Long: ",
    prefix="Lat: ",
    num_digits=5,  # Number of decimal places for the coordinates
)
m.add_child(mouse_position)

# Render the map in Streamlit
col1, col2, col3 = st.columns([1, 3, 1])  # Adjust ratios for centering
with col2:
    st.title("KCPD - Time Series Data")
    st.subheader("District Map")
    st_data = st_folium(m, width=1100, height=700)
