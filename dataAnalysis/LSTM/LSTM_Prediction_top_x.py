import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input, Bidirectional, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Loss
import tensorflow.keras.backend as K

import tqdm
import logging

logging.getLogger("tensorflow").setLevel(logging.ERROR)

basepath = 'top_10_crimes'

def create_sequences(data, time_steps=1):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps)])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)

def plot_crime_forecast(train_data, test_data, forecast_data, path, crime_type="Unknown Crime", district_id="Unknown District", plot_train_aswell=False):
    plt.figure(figsize=(20, 6))
    if plot_train_aswell:
        plt.plot(train_data['ds'], train_data['y'], label="Training Data", color="blue")
    plt.plot(test_data['ds'], test_data['y'], label="Test Data (Actual)", color="green")
    plt.plot(forecast_data['ds'], forecast_data['yhat'], label="Predictions", color="red", linestyle="--")
    plt.title(f"Crime Forecast for {crime_type} in District {district_id}")
    plt.xlabel("Date")
    plt.ylabel("Number of Crimes")
    plt.legend(loc="upper left")
    plt.grid(True)
    if os.path.exists(path):
        plt.savefig(os.path.join(path, "LSTM_forecast.png"))
    else:
        os.makedirs(path)
        plt.savefig(os.path.join(path, "LSTM_forecast.png"))
    plt.close()

def preprocess_data(data, time_steps=1):
    data = np.maximum(data, 0)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.reshape(-1, 1))
    X, y = create_sequences(scaled_data, time_steps)
    return X, y, scaler

def plot_rmse_heatmap(crime_dist_dict):
    rmse_data = []
    for dist_id, crimes in crime_dist_dict.items():
        for crime_name, crime_info in crimes.items():
            if crime_info.get("rmse") is not None:
                rmse_data.append({
                    "District": dist_id,
                    "Crime": crime_name,
                    "RMSE": crime_info["rmse"]
                })
    rmse_df = pd.DataFrame(rmse_data)
    rmse_pivot = rmse_df.pivot(index="Crime", columns="District", values="RMSE")
    plt.figure(figsize=(12, 8))
    sns.heatmap(rmse_pivot, annot=True, fmt=".2f", cmap="coolwarm", cbar_kws={'label': 'RMSE'})
    plt.title("RMSE of Crime Predictions by District and Crime Category")
    plt.xlabel("District")
    plt.ylabel("Crime Category")
    plt.tight_layout()
    plt.show()

rmse_dict = {}
time_steps = 12

class WeightedHuberLoss(Loss):
    """Custom Huber Loss to penalize peaks more."""
    def __init__(self, delta=1.0, peak_weight=5.0):
        super(WeightedHuberLoss, self).__init__()
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

def build_improved_lstm_model_with_custom_loss(time_steps, learning_rate=0.001):
    model = Sequential([
        Input(shape=(time_steps, 1)),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.2),
        Bidirectional(LSTM(64)),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    custom_loss = WeightedHuberLoss(delta=1.0, peak_weight=5.0)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss=custom_loss)
    return model

# Training und Prediction Loop
for dist_id in tqdm.tqdm(os.listdir(basepath), desc="LSTM Training and Prediction with Weighted Loss", leave=False):
    crimes = os.listdir(os.path.join(basepath, str(dist_id)))
    rmse_dict[str(dist_id)] = {}
    for crime in crimes:
        rmse_dict[str(dist_id)][crime] = {}
        train_df = pd.read_csv(os.path.join(basepath, dist_id, crime, "train.csv"))
        test_df = pd.read_csv(os.path.join(basepath, dist_id, crime, "test.csv"))
        train_df['Reported_Date'] = pd.to_datetime(train_df['Reported_Date'])
        test_df['Reported_Date'] = pd.to_datetime(test_df['Reported_Date'])
        full_df = pd.concat([train_df, test_df], ignore_index=True).sort_values('Reported_Date').reset_index(drop=True)
        data_series = full_df['Crime_Count'].values
        data_series = np.maximum(data_series, 0)
        X_train, y_train, scaler = preprocess_data(data_series[:len(train_df)], time_steps)
        X_test, y_test, _ = preprocess_data(data_series[len(train_df) - time_steps:], time_steps)
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        
        model = build_improved_lstm_model_with_custom_loss(time_steps=time_steps, learning_rate=0.001)

        model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0)
        
        y_pred = model.predict(X_test)
        y_pred_inv = scaler.inverse_transform(y_pred)
        y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
        y_pred_inv = np.maximum(y_pred_inv, 0)
        min_length = min(len(y_test_inv), len(y_pred_inv))
        y_test_trimmed = y_test_inv[:min_length]
        y_pred_trimmed = y_pred_inv[:min_length]
        rmse = np.sqrt(mean_squared_error(y_test_trimmed, y_pred_trimmed))
        rmse_dict[str(dist_id)][crime]["rmse"] = rmse
        min_length = min(len(test_df['Reported_Date'].iloc[time_steps:]), len(y_pred_trimmed))

        forecast_df = pd.DataFrame({
            'ds': test_df['Reported_Date'].iloc[time_steps:time_steps + min_length].reset_index(drop=True),
            'yhat': y_pred_trimmed.flatten()[:min_length]
        })

        test_plot_df = pd.DataFrame({
            'ds': test_df['Reported_Date'].iloc[time_steps:time_steps + min_length].reset_index(drop=True),
            'y': y_test_trimmed.flatten()[:min_length]
        })
        plot_directory = os.path.join("dataAnalysis/LSTM/lstm_plots", str(dist_id), crime)
        os.makedirs(plot_directory, exist_ok=True)
        plot_crime_forecast(
            train_data=train_df.rename(columns={'Reported_Date': 'ds', 'Crime_Count': 'y'}),
            test_data=test_plot_df,
            forecast_data=forecast_df,
            path=plot_directory,
            crime_type=crime,
            district_id=dist_id,
            plot_train_aswell=False
        )

plot_rmse_heatmap(rmse_dict)
