from fredapi import Fred
import yfinance as yf
import numpy as np
import pandas as pd
import os
import pandas_ta as ta  # Import Pandas TA for technical indicators

# Ensure the output directory exists
output_dir = 'dataset/MSFT'
os.makedirs(output_dir, exist_ok=True)

# # Define FRED API key
API_KEY = 'ec9c3a532618d0109bc583602f15dc83'
fred = Fred(api_key=API_KEY)

# # Define macroeconomic series and fetch data
macro_series = {
    'M1': 'M1SL',
    'M2': 'M2SL',
    '15_Year_Mortgage_Rate': 'MORTGAGE15US',
    '30_Year_Mortgage_Rate': 'MORTGAGE30US',
    'CPI': 'CPIAUCSL'
}

start_date = '2000-01-01'
end_date = '2005-12-31'

# Fetch OHLCV data from Yahoo Finance
ticker_symbol = 'MSFT'
msft = yf.Ticker(ticker_symbol)
stock_data = msft.history(
    start=start_date,
    end=end_date,
    interval="1d",
    auto_adjust=False
)[['Open', 'High', 'Low', 'Close', 'Volume']]

stock_data.index = stock_data.index.tz_localize(None)  # Remove timezone

# Fetch macroeconomic data
macro_data = {}
for name, series_id in macro_series.items():
    macro_data[name] = fred.get_series(series_id, start_date, end_date)

# Convert macroeconomic data to a single DataFrame and align with OHLCV
macro_df = pd.DataFrame(macro_data)
macro_df.index = pd.to_datetime(macro_df.index)
macro_df = macro_df.resample('D').ffill()  # Convert to daily frequency using forward fill

# Align macroeconomic data with OHLCV
aligned_macro = macro_df.reindex(stock_data.index).ffill().bfill()

# Calculate new technical indicators
# # 5-Day Weighted Close Price (Weighted Close: (High + Low + 2*Close)/4)
combined_data = pd.concat([stock_data, aligned_macro], axis=1)

# Calculate Technical Indicators
combined_data['5WCLPRICE'] = (combined_data['High'] + combined_data['Low'] + 2 * combined_data['Close']) / 4
combined_data['5WCLPRICE'] = combined_data['5WCLPRICE'].rolling(window=5).mean()
combined_data['5MedPrice'] = combined_data['Close'].rolling(window=5).median()
combined_data['5AvgPrice'] = combined_data['Close'].rolling(window=5).mean()

# Drop rows with NaN values resulting from rolling calculations
combined_data.dropna(inplace=True)

# Calculate mean and std across all columns for normalization
mean = combined_data.mean()
std = combined_data.std()

# Save scaling information
np.savez(os.path.join(output_dir, 'var_scaler_info.npz'), mean=mean.values, std=std.values)

dates = combined_data.index
norm_time_marker = np.stack([
    np.full(len(dates), 0.5),  # Time of day (fixed for daily data)
    dates.weekday / 4.0,        # Day of week (normalized)
    (dates.day - 1) / (dates.to_series().groupby(dates.to_period("M")).transform("count") - 1),  # Day of month
    (dates.dayofyear - 1) / (dates.to_series().groupby(dates.to_period("Y")).transform("count") - 1)  # Day of year
], axis=1)

# Save the final combined data and normalized time markers
np.savez(os.path.join(output_dir, 'feature.npz'), norm_var=combined_data.values, norm_time_marker=norm_time_marker)

# Verify shapes
print("Final combined data shape:", combined_data.shape)  # Should have OHLCV + macro + new indicators columns
print("norm_time_marker shape:", norm_time_marker.shape)
print(f"mean shape: {mean.shape} and std shape {std.shape}")
print(f"mean value: \n {mean}")
print(f"std value: \n {std}")
print("First 10 rows of the downloaded data:")
print(combined_data.head(10))
print("Last 10 rows of the downloaded data:")
print(combined_data.tail(10))
print("Data successfully saved.")
