import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
import tqdm
import logging

# Disable unnecessary logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# Define your base path
basepath = 'data\\top_10_crimes'

# Get list of districts and crimes
districts = os.listdir(basepath)
print(f"Districts: {districts}")

# Function to create sequences
def create_sequences(data, time_steps=1):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps)])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)

# Plotting function for forecasts
def plot_crime_forecast(train_data, test_data, forecast_data, path, crime_type="Unknown Crime", district_id="Unknown District", plot_train_aswell=False):
    plt.figure(figsize=(20, 6))

    if plot_train_aswell:
        # Plot training data
        plt.plot(train_data['ds'], train_data['y'], label="Training Data", color="blue")

    # Plot test data (actual values)
    plt.plot(test_data['ds'], test_data['y'], label="Test Data (Actual)", color="green")

    # Plot forecasted data (predictions)
    plt.plot(forecast_data['ds'], forecast_data['yhat'], label="Predictions", color="red", linestyle="--")

    # Formatting the plot
    plt.title(f"Crime Forecast for {crime_type} in District {district_id}")
    plt.xlabel("Date")
    plt.ylabel("Number of Crimes")
    plt.legend(loc="upper left")
    plt.grid(True)

    # Save the plot
    if os.path.exists(path):
        plt.savefig(os.path.join(path, "LSTM_forecast.png"))
    else:
        os.makedirs(path)
        plt.savefig(os.path.join(path, "LSTM_forecast.png"))
    plt.close()

# Plotting function for RMSE heatmap
def plot_rmse_heatmap(crime_dist_dict):
    rmse_data = []

    # Collect RMSE values for each crime in each district
    for dist_id, crimes in crime_dist_dict.items():
        for crime_name, crime_info in crimes.items():
            if crime_info.get("rmse") is not None:
                rmse_data.append({
                    "District": dist_id,
                    "Crime": crime_name,
                    "RMSE": crime_info["rmse"]
                })

    # Create a DataFrame from the collected RMSE data
    rmse_df = pd.DataFrame(rmse_data)

    # Pivot the DataFrame to have districts as columns and crimes as rows
    rmse_pivot = rmse_df.pivot(index="Crime", columns="District", values="RMSE")

    # Plot the heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(rmse_pivot, annot=True, fmt=".2f", cmap="coolwarm", cbar_kws={'label': 'RMSE'})

    # Formatting the plot
    plt.title("RMSE of Crime Predictions by District and Crime Category")
    plt.xlabel("District")
    plt.ylabel("Crime Category")
    plt.tight_layout()

    # Show the plot
    plt.show()

# Initialize RMSE dictionary
rmse_dict = {}

# LSTM parameters
time_steps = 12  # Adjust as needed

for dist_id in tqdm.tqdm(districts, desc="LSTM Training and Prediction", leave=False):
    crimes = os.listdir(os.path.join(basepath, str(dist_id)))
    rmse_dict[str(dist_id)] = {}
    for crime in crimes:
        rmse_dict[str(dist_id)][crime] = {}

        # Load data
        train_df = pd.read_csv(os.path.join(basepath, dist_id, crime, "train.csv"))
        test_df = pd.read_csv(os.path.join(basepath, dist_id, crime, "test.csv"))

        # Prepare data
        train_df['Reported_Date'] = pd.to_datetime(train_df['Reported_Date'])
        test_df['Reported_Date'] = pd.to_datetime(test_df['Reported_Date'])

        full_df = pd.concat([train_df, test_df], ignore_index=True)
        full_df = full_df.sort_values('Reported_Date').reset_index(drop=True)

        # Extract the 'Crime_Count' as the time series
        data_series = full_df['Crime_Count'].values.reshape(-1, 1)

        # Normalize the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data_series)

        # Split back into train and test sets
        train_size = len(train_df)
        test_size = len(test_df)
        train_data = scaled_data[:train_size]
        test_data = scaled_data[train_size - time_steps:]  # Include some data from the end of train set

        # Create sequences
        X_train, y_train = create_sequences(train_data, time_steps)
        X_test, y_test = create_sequences(test_data, time_steps)

        # Reshape input to [samples, time_steps, features]
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

        # Build LSTM model
        model = Sequential()
        model.add(LSTM(64, input_shape=(time_steps, 1)))
        model.add(Dense(1))
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

        # Train the model
        model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0)

        # Make predictions
        y_pred = model.predict(X_test)

        # Inverse transform the predictions and actual values
        y_pred_inv = scaler.inverse_transform(y_pred)
        y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

        # Ensure lengths match
        num_predictions = len(y_pred_inv)

        # Extract corresponding dates from the test set
        test_dates = test_df['Reported_Date'].iloc[time_steps:].reset_index(drop=True)

        # Adjust lengths if necessary
        min_length = min(len(test_dates), num_predictions)
        test_dates = test_dates[:min_length]
        y_pred_inv = y_pred_inv[:min_length]
        y_test_inv = y_test_inv[:min_length]

        # Compute RMSE
        rmse = root_mean_squared_error(y_test_inv, y_pred_inv)
        rmse_dict[str(dist_id)][crime]["rmse"] = rmse

        # Prepare data for plotting
        forecast_df = pd.DataFrame({
            'ds': test_dates,
            'yhat': y_pred_inv.flatten(),
            'yhat_lower': y_pred_inv.flatten(),  # For simplicity
            'yhat_upper': y_pred_inv.flatten(),
            'trend': y_pred_inv.flatten()
        })

        # Adjust test data for plotting
        test_plot_df = pd.DataFrame({
            'ds': test_dates,
            'y': y_test_inv.flatten()
        })

        # Create plot directory
        plot_directory = os.path.join("lstm_plots", str(dist_id), crime)
        os.makedirs(plot_directory, exist_ok=True)

        # Plot forecast
        plot_crime_forecast(
            train_data=train_df.rename(columns={'Reported_Date': 'ds', 'Crime_Count': 'y'}),
            test_data=test_plot_df,
            forecast_data=forecast_df,
            path=plot_directory,
            crime_type=crime,
            district_id=dist_id,
            plot_train_aswell=False
        )

# Print RMSE dictionary
print(rmse_dict)

# Plot RMSE heatmap
plot_rmse_heatmap(rmse_dict)
