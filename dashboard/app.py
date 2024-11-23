import streamlit as st
import geopandas as gpd
import pandas as pd
import folium
from streamlit_folium import st_folium
from folium.plugins import MousePosition
from shapely import wkt
import os
import datetime
from shapely.geometry import Point


st.set_page_config(page_title="KCPD", page_icon="🌍", layout="wide")

# Cache functions
@st.cache_data
def load_district_data():
    district_df = pd.read_csv("../data/city_council_districts/City_Council_Districts_Shapefile_-_Effective_2023_20241004.csv", low_memory=False)
    return district_df

@st.cache_data
def load_offenses():
    offenses = os.listdir("../data/top_10_crimes/1.0")
    return offenses

# Load data
offenses = load_offenses()
district_df = load_district_data()

# Create GeoDataFrame
gdf = gpd.GeoDataFrame(district_df, geometry=gpd.GeoSeries.from_wkt(district_df["the_geom"]))
gdf['geometry'] = gdf['geometry'].simplify(tolerance=0.001)
gdf.set_crs(epsg=4326, inplace=True)

# Map setup
m = folium.Map(location=[39.075, -94.53], zoom_start=10)

colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF', '#FFFFFF']

# Add GeoJson layers and capture click events
for i, row in gdf.iterrows():
    folium.GeoJson(
        row["geometry"],
        name=row["DISTRICT_2"],
        style_function=lambda x, color=colors[i % len(colors)]: {
            'fillColor': color,
            'color': 'grey',
            'weight': 2,
            'fillOpacity': 0.25
        },
        tooltip=folium.Tooltip(row["DISTRICT_2"]),
        highlight_function=lambda x: {'weight': 3, 'color': 'blue'}
    ).add_to(m)

# Add mouse position plugin
mouse_position = MousePosition(
    position="topright",
    separator=" | Long: ",
    prefix="Lat: ",
    num_digits=5,
)
m.add_child(mouse_position)

# Streamlit Layout
st.title("KCPD - Time Series Data")
st.markdown("---")

subcol1, subcol2 = st.columns(2)

# Initialize session state for selected district
if "selected_district" not in st.session_state:
    st.session_state["selected_district"] = ""

with subcol1:
    st.subheader("Select the district:")
    # Render the map and capture map click data
    map_data = st_folium(m, width=1000, height=650)
    
    # Check for clicked coordinates
    if map_data and map_data.get("last_object_clicked"):
        clicked_coords = map_data["last_object_clicked"]
        clicked_point = Point(clicked_coords["lng"], clicked_coords["lat"])
        
        # Find the district containing the clicked point
        clicked_district = gdf[gdf.contains(clicked_point)]
        
        if not clicked_district.empty:
            district_name = clicked_district.iloc[0]["DISTRICT_2"]
            st.session_state["selected_district"] = district_name

with subcol2:
    st.subheader("Select the crime:")
    offense = st.selectbox("Select offense", options=offenses, index=0)
    st.markdown("---")

    st.subheader("Select the prediction period:")
    today = datetime.datetime.now()
    jan_1_23 = datetime.date(2023, 1, 1)
    dec_31_23 = datetime.date(2023, 12, 31)
    dec_31_25 = datetime.date(2025, 12, 31)

    date_range = st.date_input(
        "Select the time period",
        (jan_1_23, dec_31_23),
        jan_1_23,
        dec_31_25,
        format="DD.MM.YYYY",
    )

    st.markdown("---")
    district = st.session_state['selected_district']
    if st.session_state['selected_district']:
        # Display selected district dynamically
        st.subheader("Selected District:")
        st.text(f"{district} district")
    else:
        st.info("Click on the map to select the district!")

    st.markdown("---")
    st.subheader("Select the prediction model:")
    model = st.selectbox("Select the prediction model:", ["ARIMA", "LSTM"])

    st.markdown("---")
    if offense and model and date_range and district:
        st.subheader("Run the prediction:")
        st.button("Run prediction..", use_container_width=True)
    else:
        st.info("Please set all parameters first!")
