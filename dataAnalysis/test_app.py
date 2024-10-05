import streamlit as st
import geopandas as gpd
import pandas as pd
import folium
from streamlit_folium import st_folium
from folium.plugins import MousePosition
from shapely import wkt

@st.cache_data
def load_district_data():
    district_df = pd.read_csv("../data/city_council_districts/City_Council_Districts_Shapefile_-_Effective_2023_20241004.csv", low_memory=False)
    return district_df

@st.cache_data
def load_crime_data(start_date=None, end_date=None):
    crime_df = pd.read_csv("../data/mergedData/merged_df.csv", usecols=["Location", "Offense", "Description", "Address", "Reported_Date"], low_memory=False)
    # Convert the Reported_Date column to datetime format
    crime_df['Reported_Date'] = pd.to_datetime(crime_df['Reported_Date'], format='%m/%d/%Y')

    # Filter the DataFrame based on the provided date range
    if start_date is not None and end_date is not None:
        start_date = pd.to_datetime(start_date, format='%m/%d/%Y')
        end_date = pd.to_datetime(end_date, format='%m/%d/%Y')
        crime_df = crime_df[(crime_df['Reported_Date'] >= start_date) & (crime_df['Reported_Date'] <= end_date)]
    
    return crime_df

# Set Streamlit page layout
st.set_page_config(page_title="KCPD", page_icon="🌍", layout="wide")

start_date = st.date_input("Start date", value=pd.to_datetime("01/01/2024"))
end_date = st.date_input("End date", value=pd.to_datetime("01/31/2024"))

# Load the data
district_df = load_district_data()
crime_df = load_crime_data(start_date=start_date, end_date=end_date)

# Convert district data to GeoDataFrame
gdf = gpd.GeoDataFrame(district_df, geometry=gpd.GeoSeries.from_wkt(district_df["the_geom"]))  # Create a GeoDataFrame
gdf.set_crs(epsg=4326, inplace=True)  # Set the correct Coordinate Reference System (CRS)

# Create a map centered around the area
m = folium.Map(location=[39.075, -94.53], zoom_start=10)

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

# Extract crime coordinates and add markers to the map
crime_df['geometry'] = crime_df['Location'].apply(wkt.loads)  # Convert POINT data to Shapely geometry
crime_gdf = gpd.GeoDataFrame(crime_df, geometry='geometry')  # Create GeoDataFrame for crime data
crime_gdf.set_crs(epsg=4326, inplace=True)  # Ensure the CRS matches the map

# Add crime markers
for idx, crime in crime_gdf.iterrows():
    lat = crime.geometry.y
    lon = crime.geometry.x
    description = f"{crime['Offense']} - {crime['Description']} at {crime['Address']}"
    folium.Marker([lat, lon], popup=description, icon=folium.Icon(color="red", icon="info-sign")).add_to(m)

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
    st.subheader("District Map with Crime Data")
    st_data = st_folium(m, width=1100, height=700)
