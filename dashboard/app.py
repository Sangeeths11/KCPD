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
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import Loss
import tensorflow.keras.backend as K
from sklearn.preprocessing import MinMaxScaler
import numpy as np


class WeightedHuberLoss(Loss):
    """Custom Huber Loss to penalize peaks more."""
    def __init__(self, delta=1.0, peak_weight=5.0, **kwargs):
        super(WeightedHuberLoss, self).__init__(**kwargs)
        self.delta = delta
        self.peak_weight = peak_weight

    def call(self, y_true, y_pred):
        error = y_true - y_pred
        abs_error = K.abs(error)
        quadratic = K.minimum(abs_error, self.delta)
        linear = abs_error - quadratic
        loss = 0.5 * quadratic**2 + self.delta * linear

        # Apply higher weight for peaks
        peak_mask = K.cast(K.greater(y_true, K.mean(y_true) + 2 * K.std(y_true)), K.floatx())
        loss = loss + peak_mask * self.peak_weight * loss

        return K.mean(loss)

    def get_config(self):
        config = super(WeightedHuberLoss, self).get_config()
        config.update({
            "delta": self.delta,
            "peak_weight": self.peak_weight
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


st.set_page_config(page_title="KCPD", page_icon="🌍", layout="wide")

# Inject custom CSS for spinner or other elements
st.markdown("""
    <style>
        .st-emotion-cache-ocqkz7.e1f1d6gn5 {
            height: auto !important;
            min-height: 50px !important; /* Set a minimum height */
        }
    </style>
""", unsafe_allow_html=True)

ROUND_PREDICTIONS = st.sidebar.selectbox("Round predictions:", [True, False], 1)
if st.sidebar.button("Rerun.."):
    st.session_state["selected_district"] = ""
    st.rerun()

# Initialize session state for selected district
if "selected_district" not in st.session_state:
    st.session_state["selected_district"] = ""

# Cache functions
@st.cache_data
def load_district_data():
    district_df = pd.read_csv("../data/city_council_districts/City_Council_Districts_Shapefile_-_Effective_2023_20241004.csv", low_memory=False)
    return district_df

@st.cache_data
def load_offenses():
    offenses = os.listdir("../data/top_10_crimes/1.0")
    return offenses

@st.cache_data
def load_crime_data(dist_id, offense, subset="test"):
    data_path = f"../data/top_10_crimes/{dist_id}/{offense}/{subset}.csv"
    crime_df = pd.read_csv(data_path)
    return crime_df

def arima_predictions(dist_id, offense, n_periods):
    model_store_path = f"../dataAnalysis/Arima_models/{dist_id}/{offense}"
    
    if os.path.exists(model_store_path):
        with open(os.path.join(model_store_path, 'arima.pkl'), 'rb') as pkl:
            pickle_preds = pickle.load(pkl).predict(n_periods=n_periods)
        if ROUND_PREDICTIONS:
            pickle_preds = pickle_preds.round()
        return pickle_preds
    
def sf_predictions(dist_id, offense, n_periods):
    model_store_path = f"../dataAnalysis/sf_models_statsforecast/{dist_id}/{offense}"
    
    if os.path.exists(model_store_path):
        with open(os.path.join(model_store_path, 'sf_model.pkl'), 'rb') as pkl:
            sf_model = pickle.load(pkl)
        
        test_df = load_crime_data(dist_id, offense, subset="test")
        # Rename columns
        test_df = test_df.rename(columns={"Reported_Date": "ds", "Crime_Count": "y"})
        # Add unique_id column
        test_df['unique_id'] = dist_id
        # Ensure the date column is in datetime format
        test_df['ds'] = pd.to_datetime(test_df['ds'])

        n_periods = min(n_periods, len(test_df))
        n_periods -= 1

        # Ensure n_periods is positive
        if n_periods <= 0:
            raise ValueError("n_periods must be greater than 0 after adjustment.")

        # Generate predictions by doing cross validation
        forecast = sf_model.cross_validation(
            df=test_df.head(n_periods+1),
            h=1, # Horizon is 1, since we forecast the next day
            step_size=1, # Move the window by 1 time step
            n_windows=n_periods
        )

        return forecast

def create_sequences(data, time_steps=1):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps)])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)

def preprocess_data(data, time_steps=1):
    """Preprocessing for LSTM"""
    data = np.maximum(data, 0)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.reshape(-1, 1))
    X, y = create_sequences(scaled_data, time_steps)
    return X, y, scaler

def lstm_predictions(dist_id, offense, n_periods):
    model_store_path = f"../dataAnalysis/LSTM_models/{dist_id}/{offense}"
    
    if os.path.exists(model_store_path):
        model_store_path = os.path.join(model_store_path, 'saved_model.keras')
        # Load the saved model
        model = load_model(
            model_store_path,
            custom_objects={'WeightedHuberLoss': WeightedHuberLoss}
        )

        # Time step size
        time_steps = 12

        # Preprocessing
        train_df = load_crime_data(dist_id=dist_id, offense=offense, subset="train")
        train_df['Reported_Date'] = pd.to_datetime(train_df['Reported_Date'])
        train_df = train_df.sort_values('Reported_Date').reset_index(drop=True)

        # Prepare the training data
        data_series = train_df['Crime_Count'].values
        X_train, y_train, scaler = preprocess_data(data_series, time_steps)

        # Start with the last `time_steps` from the training set
        input_sequence = data_series[-time_steps:].reshape(1, time_steps, 1)

        # Store predictions
        predictions = []

        for _ in range(n_periods):
            # Predict the next value
            y_pred = model.predict(input_sequence)
            y_pred_inv = scaler.inverse_transform(y_pred)

            # Ensure non-negative predictions
            y_pred_inv = np.maximum(y_pred_inv, 0)
            next_value = y_pred_inv.flatten()[0]

            if ROUND_PREDICTIONS:
                next_value = round(next_value)

            predictions.append(next_value)

            # Update the input sequence for the next prediction
            input_sequence = np.append(input_sequence[:, 1:, :], [[[next_value]]], axis=1)

        return predictions

# Function to create and display the plot
def display_predictions_lstm(dist_id, offense, date_range, n_periods, predictions):
    st.subheader(f"Prediction Results for {n_periods} next days:")
    
    # Generate a date range for the predictions
    start_date = date_range[0]
    prediction_dates = pd.date_range(start=start_date, periods=n_periods, freq='d')

    # Create a DataFrame for plotting
    predictions_df = pd.DataFrame({
        'Reported_Date': prediction_dates,
        'Crime_Count': predictions
    })
    
    test_df = load_crime_data(dist_id=dist_id, offense=offense, subset="test")
    test_df['Reported_Date'] = pd.to_datetime(test_df['Reported_Date'])
    predictions_df['Reported_Date'] = pd.to_datetime(predictions_df['Reported_Date'])

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot test data
    ax.plot(test_df['Reported_Date'], test_df['Crime_Count'], label="Test Data", color='grey', linewidth=2)

    # Plot predictions
    ax.plot(predictions_df['Reported_Date'], predictions_df['Crime_Count'], label="Predictions", color='blue')

    # Title and labels
    ax.set_title(f"Predictions vs Test Data for {offense} in District {dist_id}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Crime Count")
    
    # Add legend
    ax.legend()

    # Add grid
    ax.grid()

    # Adjust layout and display the plot
    plt.tight_layout()
    st.pyplot(fig)

def display_predictions_sf_models(dist_id, offense, date_range, n_periods, predictions_df):
    st.subheader(f"Prediction Results for {n_periods} next days using Multiple Models:")
    
    # Load test data
    test_df = load_crime_data(dist_id=dist_id, offense=offense, subset="test")
    test_df['Reported_Date'] = pd.to_datetime(test_df['Reported_Date'])
    
    # Plotting
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot predictions for each model
    models = ['AutoARIMA', 'CrostonOptimized', 'ADIDA', 'IMAPA', 'TSB', 'SESOpt']
    for model in models:
        ax.plot(predictions_df['ds'], predictions_df[model], label=f"{model} Predictions")
    
    # Plot test data
    ax.plot(test_df['Reported_Date'], test_df['Crime_Count'], linestyle="dashed", label="Test Data", color='grey', linewidth=1)
    
    # Title and labels
    ax.set_title(f"Predictions vs Test Data for {offense} in District {dist_id}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Crime Count")
    
    # Add legend
    ax.legend()
    
    # Add grid
    ax.grid()
    
    # Adjust layout and display the plot
    plt.tight_layout()
    st.pyplot(fig)


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
    jan_1_24 = datetime.date(2024, 1, 1)
    dec_31_24 = datetime.date(2024, 12, 31)
    dec_31_25 = datetime.date(2025, 12, 31)

    # Display the fixed start date
    st.write(f"Prediction start date: {jan_1_24.strftime('%d.%m.%Y')} (fixed)")

    # Let the user select the end date only
    end_date = st.date_input(
        "Select the end date:",
        value=dec_31_24,  # Default end date
        min_value=jan_1_24,  # Ensure the end date cannot be earlier than the start date
        max_value=dec_31_24,  # Maximum allowed end date
        format="DD.MM.YYYY",
    )

    date_range = (jan_1_24, end_date)

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
    model = st.selectbox("Select the prediction model:", ["Statistical Models", "LSTM"])

st.markdown("---")

st.subheader(f"Run the {model} prediction:")
if st.button("Run prediction..", use_container_width=True):
    if not offense or not model or not date_range or not district:
        st.error("Please set all parameters first!")
    else:
        dist_id = f"{district[0]}.0"
        n_periods = (date_range[1] - date_range[0]).days
        
        plot_cols1, plot_cols2, plot_cols3 = st.columns([1,6,1])
        # Add spinner while running predictions
        with st.spinner(f"Running {model} predictions..."):
            if model == "Statistical Models":
                predictions = sf_predictions(dist_id, offense, n_periods)
                with plot_cols2:
                    # Display the predictions for sf models
                    display_predictions_sf_models(dist_id, offense, date_range, n_periods, predictions)
            else:
                predictions = lstm_predictions(dist_id, offense, n_periods)
                with plot_cols2:
                    # Display the predictions for lstm
                    display_predictions_lstm(dist_id, offense, date_range, n_periods, predictions)