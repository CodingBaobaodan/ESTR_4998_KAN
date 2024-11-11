# import yfinance as yf
# import numpy as np
# import pandas as pd

# # Define ticker symbol and date range for the entire period
# ticker_symbol = 'MSFT'
# start_date = '2002-01-01'
# end_date = '2005-12-31'

# # Download the entire data range
# msft = yf.Ticker(ticker_symbol)
# full_data = msft.history(
#     start=start_date,
#     end=end_date,
#     interval="1d",
#     auto_adjust=False
# )[['Open', 'High', 'Low', 'Close', 'Volume']]

# # Remove timezone from dates if any, to avoid warnings
# full_data.index = full_data.index.tz_localize(None)

# # Extract dates from the index for computing time markers
# dates = full_data.index

# # Compute time markers based on trading days and hours

# # Time of Day (normalized over 6.5 trading hours, assuming the market opens at 9:30 and closes at 4:00)
# # Since data is daily, we can set tod to 0.5 (midpoint of the trading day)
# tod = np.full(len(dates), 0.5)  # Shape: (L,)

# # Day of Week (normalized over 5 trading days)
# dow = dates.weekday / 4.0  # Normalize Monday (0) to Friday (4) to [0, 1]

# # Compute trading days in month and year
# trading_days_in_month = dates.to_series().groupby(dates.to_period("M")).transform("count")
# dom = (dates.day - 1) / (trading_days_in_month - 1)  # Normalize day within trading days in the month

# trading_days_in_year = dates.to_series().groupby(dates.to_period("Y")).transform("count")
# doy = (dates.dayofyear - 1) / (trading_days_in_year - 1)  # Normalize day within trading days in the year

# # Stack these markers into a single array with shape (L, 4)
# norm_time_marker = np.stack([tod, dow, dom, doy], axis=1)  # Shape: (L, 4)

# # Convert full data to norm_var with shape (L, num_features)
# norm_var = full_data.values  # Shape: (L, 5)

# # Calculate mean and std across the first dimension (time dimension) for each feature
# mean = np.mean(norm_var, axis=0)  # Shape: (5,)
# std = np.std(norm_var, axis=0)    # Shape: (5,)

# # Save the scaling information
# np.savez('dataset/MSFT/var_scaler_info.npz', mean=mean, std=std)

# # Save both norm_var and norm_time_marker into feature.npz
# np.savez('dataset/MSFT/feature.npz', norm_var=norm_var, norm_time_marker=norm_time_marker)

# # Check shapes to verify compatibility
# print("norm_var shape:", norm_var.shape)                  # Should be (L, 5)
# print("norm_time_marker shape:", norm_time_marker.shape)  # Should be (L, 4)
# print("var_scaler_info.npz created with mean and std for each feature.")

import yfinance as yf
import numpy as np
import pandas as pd
import os

# Ensure the output directory exists
output_dir = 'dataset/MSFT'
os.makedirs(output_dir, exist_ok=True)

# Define ticker symbol and date range for the entire period
ticker_symbol = 'MSFT'
start_date = '2002-01-01'
end_date = '2005-12-31'

# Download the entire data range
msft = yf.Ticker(ticker_symbol)
full_data = msft.history(
    start=start_date,
    end=end_date,
    interval="1d",
    auto_adjust=False
)[['Open', 'High', 'Low', 'Close']]

# Remove timezone from dates if any, to avoid warnings
full_data.index = full_data.index.tz_localize(None)

# Extract dates from the index for computing time markers
dates = full_data.index

# Compute time markers based on trading days and hours

# Time of Day (normalized over 6.5 trading hours)
# Since data is daily, we can set tod to 0.5
tod = np.full(len(dates), 0.5)  # Shape: (L,)

# Day of Week (normalized over 5 trading days)
dow = dates.weekday / 4.0  # Normalize Monday (0) to Friday (4)

# Compute trading days in month and year
trading_days_in_month = dates.to_series().groupby(dates.to_period("M")).transform("count")
dom = (dates.day - 1) / (trading_days_in_month - 1)

trading_days_in_year = dates.to_series().groupby(dates.to_period("Y")).transform("count")
doy = (dates.dayofyear - 1) / (trading_days_in_year - 1)

# Stack time markers into a single array with shape (L, 4)
norm_time_marker = np.stack([tod, dow, dom, doy], axis=1)  # Shape: (L, 4)

# Convert full data to norm_var with shape (L, num_features)
norm_var = full_data.values  # Shape: (L, 5)

# Calculate mean and std across the first dimension (time dimension) for each feature
mean = np.mean(norm_var, axis=0)  # Shape: (5,)
std = np.std(norm_var, axis=0)    # Shape: (5,)

# Save the scaling information
np.savez(os.path.join(output_dir, 'var_scaler_info.npz'), mean=mean, std=std)

# Normalize the data
#norm_var = (norm_var - mean) / std

# Print the first 5 rows before normalization
print("First 5 rows of norm_var before normalization:")
print(norm_var[:5])

# Save both norm_var and norm_time_marker into feature.npz
np.savez(os.path.join(output_dir, 'feature.npz'), norm_var=norm_var, norm_time_marker=norm_time_marker)

# Check shapes to verify compatibility
print("norm_var shape:", norm_var.shape)                  # Should be (L, 5)
print("norm_time_marker shape:", norm_time_marker.shape)  # Should be (L, 4)
print("var_scaler_info.npz created with mean and std for each feature.")
