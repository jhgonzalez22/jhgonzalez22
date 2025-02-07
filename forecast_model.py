import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import warnings

warnings.filterwarnings("ignore")

# Load the dataset
file_path = "C:/Users/jhgonzalez/OneDrive - CoreLogic Solutions, LLC/Desktop/testdata.xlsx"
df = pd.read_excel(file_path)

# Convert date column to datetime format
df['date'] = pd.to_datetime(df['date'])

# Extract seasonality features
df['month'] = df['date'].dt.month
df['quarter'] = df['date'].dt.quarter
df['year'] = df['date'].dt.year  # Add year to capture yearly patterns

# Sort data by Client ID and Date
df = df.sort_values(by=['Client ID', 'date'])

# Filter from 2023 onwards
df_filtered = df[df['date'] >= "2023-01-01"].copy()

# Forecast horizon (months into the future)
forecast_horizon = 15

# Maximum lags (expanded to 12 months)
max_lags = 12

# Dictionary to store client-specific forecasts
forecast_results = []

# Iterate over each Client ID to train individual models
for client_id in df_filtered['Client ID'].unique():
    client_data = df_filtered[df_filtered['Client ID'] == client_id].groupby('date')['CallsOffered'].sum()

    # Ensure enough data points exist for training
    if len(client_data) < 18:  # Require at least 18 months of data
        continue

    # Create lag features (using max_lags)
    xgb_data = pd.DataFrame(client_data)
    for lag in range(1, max_lags + 1):
        xgb_data[f'lag_{lag}'] = xgb_data['CallsOffered'].shift(lag)

    # Add seasonality features
    xgb_data = xgb_data.merge(df_filtered[['date', 'month', 'quarter', 'year']], on='date', how='left')

    # 🚀 **Fix: Drop datetime column before training**
    xgb_data.drop(columns=['date'], inplace=True)

    xgb_data.dropna(inplace=True)

    # Train-test split (last 6 months as test)
    X = xgb_data.drop(columns=['CallsOffered'])
    y = xgb_data['CallsOffered']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=6, shuffle=False)

    # Train XGBoost model
    xgb_model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=50,  # Increased iterations
        learning_rate=0.05,
        max_depth=4
    )
    xgb_model.fit(X_train, y_train)

    # Forecast future values (one model per horizon)
    future_dates = pd.date_range(client_data.index[-1], periods=forecast_horizon+1, freq='MS')[1:]
    xgb_forecast = []
    last_known = X.iloc[-1].values.reshape(1, -1)

    for _ in range(forecast_horizon):
        pred = xgb_model.predict(last_known)[0]
        xgb_forecast.append(pred)
        last_known = np.roll(last_known, -1)
        last_known[0, -1] = pred  # Shift new prediction in

    # Store results per client
    for date, pred in zip(future_dates, xgb_forecast):
        forecast_results.append({
            "Client ID": client_id,
            "month_start": date.strftime('%Y-%m-%d'),
            "Predicted_Vol": round(pred),
            "Forecast_Status": "N"
        })

# Convert forecasts to DataFrame and save to CSV
forecast_df = pd.DataFrame(forecast_results)
forecast_file_path = "C:/Users/jhgonzalez/OneDrive - CoreLogic Solutions, LLC/Desktop/XGBoost_Forecast_Fixed.csv"
forecast_df.to_csv(forecast_file_path, index=False)

print(f"Forecasts saved to: {forecast_file_path}")
