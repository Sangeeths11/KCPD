import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_rmse_heatmap(crime_dist_dict):
    """
    Creates a heatmap of RMSE values for different crimes across districts.

    Parameters:
    - crime_dist_dict: Nested dictionary with RMSE values structured as {district: {crime: {"rmse": value}}}.
    """
    # Initialize an empty list to hold RMSE data
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
