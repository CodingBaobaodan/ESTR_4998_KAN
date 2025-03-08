from fredapi import Fred
import yfinance as yf
import numpy as np
import pandas as pd
import os
import pandas_ta as ta 
import matplotlib.pyplot as plt
import mplfinance as mpf
from scipy.signal import hilbert
from pre_selection import *
import gc
gc.collect()

start_date, end_date = '2010-01-01','2022-12-31'
window = 30

# 20 stock symbols
ticker_symbols = ['AAPL', 'MSFT', 'ORCL', 'AMD', 'CSCO', 'ADBE', 
                  'IBM', 'TXN', 'AMAT', 'MU', 'ADI', 'INTC', 
                  'LRCX', 'KLAC', 'MSI', 'GLW', 'HPQ', 'TYL', 
                  'PTC', 'WDC']

# 20 stock data
stock_data = {}

output_dir = 'plots'
os.makedirs(output_dir, exist_ok=True)

# 20 stocks, OHLCV
for ticker_symbol in ticker_symbols:
    ticker = yf.Ticker(ticker_symbol)
    stock_series = ticker.history(
        start=start_date,
        end=end_date,
        interval="1d",
        auto_adjust=False
    )[['Open', 'High', 'Low', 'Close', 'Volume']]
    
    stock_series.index = stock_series.index.tz_localize(None)  # Remove timezone

    # Rename columns to the desired format
    stock_data[ticker_symbol] = stock_series  # Store the smoothed data in the dictionary

    '''
    mpf.plot(stock_series,
             type='line',
             volume=True,
             title=f'{ticker_symbol} OHLCV Candlestick Chart',
             style='charles',  # You can choose a style you prefer
             savefig=f"{output_dir}/{ticker_symbol}_plot.png",  # Save as PNG
             figsize=(10, 6))  # Set figure size

    print(f"Plot saved in the directory: {output_dir}/{ticker_symbol}_plot.png")
    '''
    

stock_df = pd.concat(stock_data, axis=1)
stock_df.index = pd.to_datetime(stock_df.index)
stock_df = stock_df.iloc[window:]




# 20 stocks' 31 indicators (without OHLCV) data
stock_indicators_data = {}

for ticker_symbol in ticker_symbols:
    ticker = yf.Ticker(ticker_symbol)
    stock_series = ticker.history(
        start=start_date,
        end=end_date,
        interval="1d",
        auto_adjust=False
    )[['Open', 'High', 'Low', 'Close', 'Volume']]

    # Category 1: Overlap Indicators
    ma20 = stock_series['Close'].rolling(window=20).mean()
    stock_series['Bollinger_Bands_Upper'] = ma20 + 2 * stock_series['Close'].rolling(window=20).std()
    stock_series['Bollinger_Bands_Middle'] = ma20
    stock_series['Bollinger_Bands_Lower'] = ma20 - 2 * stock_series['Close'].rolling(window=20).std()

    stock_series['DEMA'] = ta.overlap.dema(stock_series['Close'], window=20)
    stock_series['Midpoint'] = ta.overlap.midpoint(stock_series['Close'], window=20)
    stock_series['Midpoint_Price'] = (stock_series['High'] + stock_series['Low']) / 2
    stock_series['T3_Moving_Average'] = ta.overlap.t3(stock_series['Close'], window=20)

    # Category 2: Momentum Indicators
    stock_series['ADX'] = ta.trend.adx(stock_series['High'], stock_series['Low'], stock_series['Close']).iloc[:, 0]
    stock_series['Absolute_Price_Oscillator'] = ta.momentum.apo(stock_series['Close'])
    aroon = ta.trend.aroon(stock_series['High'], stock_series['Low'], window=14)
    stock_series['Aroon_Up'] = aroon.iloc[:, 0]
    stock_series['Aroon_Down'] = aroon.iloc[:, 1]
    stock_series['Aroon_Oscillator'] = aroon.iloc[:, 2]
    stock_series['Balance_of_Power'] = ta.momentum.bop(stock_series['Open'], stock_series['High'], stock_series['Low'], stock_series['Close'])
    stock_series['CCI'] = ta.momentum.cci(stock_series['High'], stock_series['Low'], stock_series['Close'], window=20)
    stock_series['Chande_Momentum_Oscillator'] = ta.momentum.cmo(stock_series['Close'], window=14)    
    
    # Calculate 26-day EMA and 12-day EMA
    exp26 = stock_series['Close'].ewm(span=26, adjust=False).mean()
    exp12 = stock_series['Close'].ewm(span=12, adjust=False).mean()
    stock_series['MACD'] = exp12 - exp26
    stock_series['MACD_Signal'] = stock_series['MACD'].ewm(span=9, adjust=False).mean()
    stock_series['MACD_Histogram'] = stock_series['MACD'] - stock_series['MACD_Signal']
    stock_series['Money_Flow_Index'] = ta.volume.mfi(stock_series['High'], stock_series['Low'], stock_series['Close'], stock_series['Volume'], window=14)

    # Category 3: Volatility Indicators
    def wwma(values, n):
        """
        J. Welles Wilder's EMA 
        """
        return values.ewm(alpha=1/n, adjust=False).mean()

    def atr(df, n=14):
        data = df.copy()
        high = data['High']
        low = data['Low']
        close = data['Close']
        data['tr0'] = abs(high - low)
        data['tr1'] = abs(high - close.shift())
        data['tr2'] = abs(low - close.shift())
        tr = data[['tr0', 'tr1', 'tr2']].max(axis=1)
        atr = wwma(tr, n)
        return atr

    stock_series['Normalized_Average_True_Range'] = atr(stock_series)


    # Category 4: Volume Indicators
    stock_series['Chaikin_A/D_Line'] = ta.volume.ad(stock_series['High'], stock_series['Low'], stock_series['Close'], stock_series['Volume'])
    stock_series['Chaikin_A/D_Oscillator'] = ta.volume.adosc(stock_series['High'], stock_series['Low'], stock_series['Close'], stock_series['Volume'])

    # Category 5: Price Transform
    stock_series['Median_Price'] = ta.statistics.median(stock_series['Close'])
    stock_series['Typical_Price'] = ta.overlap.hlc3(stock_series['High'], stock_series['Low'], stock_series['Close'])
    stock_series['Weighted_Closing_Price'] = ta.overlap.wcp(stock_series['High'], stock_series['Low'], stock_series['Close'])
    
    
    # Category 6: Hilbert Transform indicators
    def compute_dominant_cycle_phase(prices):
        analytic_signal = hilbert(prices)
        phase = np.angle(analytic_signal)
        return phase

    def compute_phasor_components(prices):
        analytic_signal = hilbert(prices)
        inphase = np.real(analytic_signal)
        quadrature = np.imag(analytic_signal)
        return inphase, quadrature
    
    def compute_sine_wave(prices):
        phase = compute_dominant_cycle_phase(prices)
        sine_wave = np.sin(phase)
        lead_sine_wave = np.sin(phase + np.pi / 4)  # Lead by 45 degrees
        return sine_wave, lead_sine_wave

    def compute_trend_vs_cycle_mode(prices):
        phase = compute_dominant_cycle_phase(prices)
        trend_mode = (np.diff(phase) > 0).astype(int)  
        trend_mode = np.append(trend_mode, 0)  
        return trend_mode
    
    stock_series['Hilbert_Dominant_Cycle_Phase'] = compute_dominant_cycle_phase(stock_series['Close'])

    inphase, quadrature = compute_phasor_components(stock_series['Close'])
    stock_series['Hilbert_Phasor_Components_Inphase'] = inphase
    stock_series['Hilbert_Phasor_Components_Quadrature'] = quadrature

    sine_wave, lead_sine_wave = compute_sine_wave(stock_series['Close'])
    stock_series['Hilbert_SineWave'] = sine_wave
    stock_series['Hilbert_LeadSineWave'] = lead_sine_wave

    stock_series['Hilbert_Trend_vs_Cycle_Mode'] = compute_trend_vs_cycle_mode(stock_series['Close'])

    # Just need indicators, but not OHLCV
    stock_indicators_data[ticker_symbol] = stock_series.drop(columns=['Open', 'High', 'Low', 'Close', 'Volume'])


stock_indicators_df = pd.concat(stock_indicators_data, axis=1)
stock_indicators_df.index = pd.to_datetime(stock_indicators_df.index)


# # Define FRED API key
API_KEY = 'ec9c3a532618d0109bc583602f15dc83'
fred = Fred(api_key=API_KEY)

# 18 macro indicators (NOT commodities) from fred 
macro_series_fred = {
    'Total Vehicle Sales': 'TOTALSA', # 1976 to Dec 2024
    'Domestic Auto Production': 'DAUPSA', # Jan 1993 to Oct 2024
    '15-Year Fixed Rate Mortgage Average in the United States': 'MORTGAGE15US', # 1991 to 2025
    '30-Year Fixed Rate Mortgage Average in the United States': 'MORTGAGE30US', # 1977 to 2025
    'Employment Level': 'CE16OV', # 1948 to Dec 2024
    'Unemployment Rate': 'UNRATE', # 1948 to Dec 2024
    'Inflation, consumer prices for the United States': 'FPCPITOTLZGUSA',  # 1960 to Sep 2023
    'Federal Funds Effective Rate': 'FEDFUNDS', # 1954 to 2025
    'Trade Balance: Goods and Services, Balance of Payments Basis': 'BOPGSTB', # 1992 to Dec 2024
    'Consumer Price Index for All Urban Consumers: All Items in U.S. City Average': 'CPIAUCNS', # 1913 to Dec 2024
    'M1': 'M1SL', # 1959 to Dec 2024
    'M2': 'M2SL', # 1959 to Dec 2024
    'Industrial Production: Total Index': 'INDPRO', # 1919 to Dec 2024
    'US_Manufacturing_PMI': 'AMTMNO', # to Dec 2024
    'New One Family Houses Sold': 'HSN1F', # to Dec 2024
    'S&P CoreLogic Case-Shiller U.S. National Home Price Index': 'CSUSHPISA', # 1987 to Nov 2024
    'All-Transactions House Price Index for the United States': 'USSTHPI', # to Q3 2024
    'New Privately-Owned Housing Units Started: Total Units': 'HOUST' # 1959 to Dec 2024
}

macro_data_fred = {}
for name, series_id in macro_series_fred.items():
    macro_data_fred[name] = fred.get_series(series_id, start_date, end_date)

macro_df_fred = pd.DataFrame(macro_data_fred)
macro_df_fred.index = pd.to_datetime(macro_df_fred.index)
macro_df_fred = macro_df_fred.resample('D').ffill().ffill().bfill() # Convert to daily frequency using forward fill



# 5 macro indicators (Commodities)
macro_series_commodities = {
    'Gold': 'GC=F',  # Gold Futures
    'Crude Oil': 'CL=F',  # Crude Oil Futures
    'Brent Oil': 'BZ=F',  # Brent Oil Futures
    'Natural Gas': 'NG=F',  # Natural Gas Futures
    'Reformulated Blendstock Oil': 'RB=F'  # RBOB Gasoline Futures
}

macro_data_commidities = []
for name, ticker in macro_series_commodities.items():
    macro_data_commidities.append(yf.download(ticker, start=start_date, end=end_date)["Close"])

macro_df_commidities = pd.concat(macro_data_commidities, axis=1)
macro_df_commidities = macro_df_commidities.resample('D').ffill().ffill().bfill() # Convert to daily frequency using forward fill



ticker_symbols = ['^GSPC'] #, '^SP500-45']  # S&P 500 Index and S&P 500 IT Sector Index

index_data = {}

# Fetch OHLCV data for each ticker
for ticker_symbol in ticker_symbols:
    ticker = yf.Ticker(ticker_symbol)
    index_series = ticker.history(
        start=start_date,
        end=end_date,
        interval="1d",
        auto_adjust=False
    )[['Open', 'High', 'Low', 'Close', 'Volume']]
    
    index_series.index = index_series.index.tz_localize(None)  # Remove timezone

    # Rename columns to the desired format
    index_data[ticker_symbol] = index_series  # Store the smoothed data in the dictionary


index_df = pd.concat(index_data, axis=1)
index_df.index = pd.to_datetime(index_df.index)
index_df = index_df.iloc[window:]



stock_indicators_df = stock_indicators_df.ffill().ffill().bfill()
stock_indicators_df.index = stock_indicators_df.index.strftime('%d/%m/%Y')
stock_indicators_df = stock_indicators_df.iloc[window:]

macro_df_fred = macro_df_fred.reindex(stock_df.index).ffill().bfill()
macro_df_fred.index = macro_df_fred.index.strftime('%d/%m/%Y')

macro_df_commidities = macro_df_commidities.reindex(stock_df.index).ffill().bfill()
macro_df_commidities.index = macro_df_commidities.index.strftime('%d/%m/%Y')

index_df = index_df.reindex(stock_df.index).ffill().bfill()
index_df.index = index_df.index.strftime('%d/%m/%Y')

stock_df.index = stock_df.index.strftime('%d/%m/%Y')


macro_df = pd.concat([macro_df_fred, macro_df_commidities], axis=1)

output_dir = 'csv'
os.makedirs(output_dir, exist_ok=True)
stock_df.to_csv(f"{output_dir}/stock_df.csv", index=True)
stock_indicators_df.to_csv(f"{output_dir}/stock_indicators_df.csv", index=True)
macro_df.to_csv(f"{output_dir}/macro_df.csv", index=True)
index_df.to_csv(f"{output_dir}/index_df.csv", index=True)


cap_weighted_composite_index_df = cap_weighted_composite_index(stock_df)
top_k_correlations = cap_weighted_correlation_plots(cap_weighted_composite_index_df, macro_df, k=10) # Check!



ticker_symbols = ['AAPL', 'MSFT', 'ORCL', 'AMD', 'CSCO', 'ADBE', 
                  'IBM', 'TXN', 'AMAT', 'MU', 'ADI', 'INTC', 
                  'LRCX', 'KLAC', 'MSI', 'GLW', 'HPQ', 'TYL', 
                  'PTC', 'WDC']


for stock in ticker_symbols:
    
    stock_indicators = stock_indicators_df.xs(stock, level=0, axis=1)
    stock_data = stock_df.xs(stock, level=0, axis=1)

    combined_data = pd.concat([
        index_df,
        stock_indicators,
        stock_data,
        macro_df[top_k_correlations.index[1:]]  
    ], axis=1)

    # Calculate min and max across all columns for min-max normalization
    min_val = combined_data.min()
    max_val = combined_data.max()

    # Ensure the output directory exists
    output_dir = f"dataset/{stock}"
    os.makedirs(output_dir, exist_ok=True)

    # Save scaling information using min and max values
    np.savez(os.path.join(output_dir, f'var_scaler_info.npz'), min=min_val.values, max=max_val.values)

    combined_data.index = pd.to_datetime(combined_data.index, format='%d/%m/%Y')
    dates = combined_data.index

    norm_time_marker = np.stack([
        np.full(len(dates), 0.5),  # Time of day (fixed for daily data)
        dates.weekday / 4.0,        # Day of week (normalized)
        (dates.day - 1) / (dates.to_series().groupby(dates.to_period("M")).transform("count") - 1),  # Day of month
        (dates.dayofyear - 1) / (dates.to_series().groupby(dates.to_period("Y")).transform("count") - 1)  # Day of year
    ], axis=1)

    # Save the final combined data and normalized time markers
    np.savez(os.path.join(output_dir, f'feature.npz'), norm_var=combined_data.values, norm_time_marker=norm_time_marker)

    combined_data.to_csv(f"{output_dir}/all_data.csv", index=True)
