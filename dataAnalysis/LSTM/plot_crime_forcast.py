import matplotlib.pyplot as plt
import os

def plot_crime_forecast(train_data, test_data, forecast_data, path, crime_type="Unknown Crime", district_id="Unknown District", plot_train_aswell=False):
    """
    Plots the actual and forecasted crime counts over time.

    Parameters:
    - train_data: DataFrame containing training data with columns ['ds', 'y'].
    - test_data: DataFrame containing test data with columns ['ds', 'y'].
    - forecast_data: DataFrame containing forecasted data with columns ['ds', 'yhat'].
    - path: Directory path where the plot image will be saved.
    - crime_type: String indicating the type of crime.
    - district_id: String indicating the district ID.
    - plot_train_aswell: Boolean indicating whether to plot the training data.
    """
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
