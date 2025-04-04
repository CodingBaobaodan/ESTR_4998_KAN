import argparse
import importlib
import importlib.util
import os
import subprocess
import math
import random
from prettytable import PrettyTable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import lightning.pytorch as L
from lightning.pytorch.callbacks import LearningRateMonitor, EarlyStopping
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from lightning.pytorch.callbacks import Callback

from core.data_runner import DataInterface
from core.ltsf_runner import LTSFRunner
from core.util import cal_conf_hash
from functools import reduce

from fredapi import Fred
import yfinance as yf
import pandas_ta as ta 
import mplfinance as mpf
from scipy.signal import hilbert
from pre_selection import *
import gc
gc.collect()

seed_value = 4999
random.seed(seed_value)

class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

start_end_string = ""
def read_data(start_date, end_date):
    global start_end_string
    start_end_string = f"{start_date}_{end_date}"

    ticker_symbols = ['AAPL', 'MSFT', 'ORCL', 'AMD', 'CSCO', 'ADBE', 
                    'IBM', 'TXN', 'AMAT', 'MU', 'ADI', 'INTC', 
                    'LRCX', 'KLAC', 'MSI', 'GLW', 'HPQ', 'TYL', 
                    'PTC', 'JNJ']

    # 20 stock data
    stock_data = {}

    output_dir = f"{start_end_string}/plots"
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
        stock_series['Money_Flow_Index'] = ta.volume.mfi(
            stock_series['High'], 
            stock_series['Low'], 
            stock_series['Close'], 
            stock_series['Volume'], 
            window=14
        ).astype('float64')

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



    index_ticker_symbols = ['^GSPC']
    index_data = {}

    # Fetch OHLCV data for each ticker
    for ticker_symbol in index_ticker_symbols:
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


    stock_indicators_df = stock_indicators_df.ffill().ffill().bfill()
    stock_indicators_df.index = stock_indicators_df.index.strftime('%d/%m/%Y')

    macro_df_fred = macro_df_fred.reindex(stock_df.index).ffill().bfill()
    macro_df_fred.index = macro_df_fred.index.strftime('%d/%m/%Y')

    macro_df_commidities = macro_df_commidities.reindex(stock_df.index).ffill().bfill()
    macro_df_commidities.index = macro_df_commidities.index.strftime('%d/%m/%Y')

    index_df = index_df.reindex(stock_df.index).ffill().bfill()
    index_df.index = index_df.index.strftime('%d/%m/%Y')

    stock_df.index = stock_df.index.strftime('%d/%m/%Y')

    macro_df = pd.concat([macro_df_fred, macro_df_commidities], axis=1)

    output_dir = f"{start_end_string}/csv"
    os.makedirs(output_dir, exist_ok=True)

    stock_df.to_csv(f"{output_dir}/stock_df.csv", index=True)
    stock_indicators_df.to_csv(f"{output_dir}/stock_indicators_df.csv", index=True)
    macro_df.to_csv(f"{output_dir}/macro_df.csv", index=True)
    index_df.to_csv(f"{output_dir}/index_df.csv", index=True)


    cap_weighted_composite_index_df = cap_weighted_composite_index(stock_df, start_end_string)
    top_k_correlations = cap_weighted_correlation_plots(cap_weighted_composite_index_df, macro_df, start_end_string, k=10) 


    for stock in ticker_symbols:
        
        stock_indicators = stock_indicators_df.xs(stock, level=0, axis=1)
        stock_data = stock_df.xs(stock, level=0, axis=1)

        combined_data = pd.concat([
            index_df,
            stock_indicators,
            stock_data,
            macro_df[top_k_correlations.index[1:]]  
        ], axis=1)

        min_val = combined_data.min()
        max_val = combined_data.max()

        # Ensure the output directory exists
        output_dir = f"{start_end_string}/dataset/{stock}"
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

class Chromosome:
    def __init__(self, features, hyperparameters):

        self.genes = {
            'features': features,
            'hyperparameters': hyperparameters,
        }

        self.fitness = 0

def decode(ind, conf):
    indicators_list_01 = ind.genes['features']
    var_num = sum(indicators_list_01)
    
    if conf['GA_type']==1:
        hist_len_list_01, KAN_experts_list_01 = conf['hist_len_list_01'], conf['KAN_experts_list_01']
        hist_len = conf['min_hist_len'] + 4 * sum(bit << i for i, bit in enumerate(reversed(hist_len_list_01)))
    
    else:
        if conf['model_name'] == "DenseRMoK":
            hist_len_list_01, KAN_experts_list_01 = ind.genes['hyperparameters'][:conf['max_hist_len_n_bit']], ind.genes['hyperparameters'][conf['max_hist_len_n_bit']:]
            hist_len = conf['min_hist_len'] + 4 * sum(bit << i for i, bit in enumerate(reversed(hist_len_list_01)))

        else:
            hist_len_list_01 = ind.genes['hyperparameters'][:conf['max_hist_len_n_bit']]
            hist_len = conf['min_hist_len'] + 4 * sum(bit << i for i, bit in enumerate(reversed(hist_len_list_01)))
            KAN_experts_list_01 = conf['KAN_experts_list_01']

    return var_num, indicators_list_01, hist_len, hist_len_list_01, KAN_experts_list_01

def fitness_function(ind, training_conf, conf):
    conf['var_num'], conf['indicators_list_01'], conf['hist_len'], conf['hist_len_list_01'], conf['KAN_experts_list_01'] = decode(ind, conf)
    print(f"{conf['var_num']} features are selected")
    print(conf['indicators_list_01'])

    print(f"window size: {conf['hist_len']}")
    print(conf['hist_len_list_01'])

    if conf['model_name'] == "DenseRMoK":
        print("Experts Taylor, Wavelet (Morlet), Wavelet (Mexican Hat), Jacobi, Cheby", conf['KAN_experts_list_01']) 

    trainer, data_module, model, callback = train_init(training_conf, conf)
    trainer, data_module, model, test_loss, daily_return_multiplication_train_list, daily_return_multiplication_test_list = train_func(trainer, data_module, model, callback)

    ind.fitness = -1 * test_loss # min MSE == max -MSE 

    return ind

def create_initial_population(conf):
    population = []

    for _ in range(conf['population_size']):
        # 5 (index) : ('^GSPC', 'Open')	('^GSPC', 'High')	('^GSPC', 'Low')	('^GSPC', 'Close')	('^GSPC', 'Volume')	
        # 31 (technical indicators) : Bollinger_Bands_Upper	Bollinger_Bands_Middle	Bollinger_Bands_Lower	DEMA	Midpoint	Midpoint_Price	T3_Moving_Average	ADX	Absolute_Price_Oscillator	Aroon_Up	Aroon_Down	Aroon_Oscillator	Balance_of_Power	CCI	Chande_Momentum_Oscillator	MACD	MACD_Signal	MACD_Histogram	Money_Flow_Index	Normalized_Average_True_Range	Chaikin_A/D_Line	Chaikin_A/D_Oscillator	Median_Price	Typical_Price	Weighted_Closing_Price	Hilbert_Dominant_Cycle_Phase	Hilbert_Phasor_Components_Inphase	Hilbert_Phasor_Components_Quadrature	Hilbert_SineWave	Hilbert_LeadSineWave	Hilbert_Trend_vs_Cycle_Mode	
        # 5( stock) : Open	High	Low	Close	Volume	
        # 9 (macro indicators) : M2	S&P CoreLogic Case-Shiller U.S. National Home Price Index	All-Transactions House Price Index for the United States	M1	Consumer Price Index for All Urban Consumers: All Items in U.S. City Average	Trade Balance: Goods and Services, Balance of Payments Basis	New Privately-Owned Housing Units Started: Total Units	Domestic Auto Production	New One Family Houses Sold

        features = [random.choice([0, 1]) for _ in range(conf['total_n_features'])]
        features[conf['total_n_features']-14:conf['total_n_features']-14+5] = [1, 1, 1, 1, 1] 

        if conf['GA_type']==2:

            hist_len_list_01 = [random.choice([0, 1]) for _ in range(conf['max_hist_len_n_bit'])] 
        
            if conf['model_name'] == "DenseRMoK":
                KAN_experts_list_01 = [random.choice([0, 1]) for _ in range(conf['n_KAN_experts'])] 

                if sum(KAN_experts_list_01)==0:
                    index_to_set = random.randint(0, conf['n_KAN_experts'] - 1)
                    KAN_experts_list_01 = [1 if i == index_to_set else 0 for i in range(conf['n_KAN_experts'])]

                hyperparameters = hist_len_list_01 + KAN_experts_list_01

            else:
                hyperparameters = hist_len_list_01

        else: 
            hist_len_list_01, KAN_experts_list_01 = conf['hist_len_list_01'], conf['KAN_experts_list_01']
            if conf['model_name'] == "DenseRMoK":
                hyperparameters = hist_len_list_01 + KAN_experts_list_01
            else:
                hyperparameters = hist_len_list_01

        population.append(Chromosome(features, hyperparameters))

    return population

def selection(population, all_fitnesses, pop_size, tournament_size=3):
    pop_size = pop_size // 2
    tournament_size = min(tournament_size, pop_size)

    selected = []

    for _ in range(pop_size):   
        tournament = random.sample(list(zip(population, all_fitnesses)), tournament_size)
        winner = max(tournament, key=lambda x: x[1])[0]
        selected.append(winner)

    return selected, pop_size

def inter_chromosome_crossover(conf, ch1, ch2, n_features, n_hyperparameters, max_hist_len_n_bit, n_KAN_experts, model_name):

    features1 = ch1.genes['features']
    hyperparameters1 = ch1.genes['hyperparameters']
    
    features2 = ch2.genes['features']
    hyperparameters2 = ch2.genes['hyperparameters']
    
    crossover_point1 = random.randint(0, n_features - 1)
    crossover_point2 = random.randint(0, n_hyperparameters - 1)
    
    features1[crossover_point1:], features2[crossover_point1:] = features2[crossover_point1:], features1[crossover_point1:]
    if (conf['GA_type']==2) and (conf['model_name']=="DenseRMoK"):
        hyperparameters1[crossover_point2:], hyperparameters2[crossover_point2:] = hyperparameters2[crossover_point2:], hyperparameters1[crossover_point2:]
    
    ch1.genes['features'] = features1
    ch2.genes['features'] = features2

    ch1.genes['features'][n_features-14:n_features-14+5] = [1, 1, 1, 1, 1] 
    ch2.genes['features'][n_features-14:n_features-14+5] = [1, 1, 1, 1, 1] 

    if (conf['GA_type']==2) and (conf['model_name']=="DenseRMoK"):
        ch1.genes['hyperparameters'] = hyperparameters1
        ch2.genes['hyperparameters'] = hyperparameters2

        if sum(ch1.genes['hyperparameters'][max_hist_len_n_bit:])==0:
            index_to_set = random.randint(0, n_KAN_experts - 1)
            ch1.genes['hyperparameters'][max_hist_len_n_bit:] = [1 if i == index_to_set else 0 for i in range(n_KAN_experts)]

        if sum(ch2.genes['hyperparameters'][max_hist_len_n_bit:])==0:
            index_to_set = random.randint(0, n_KAN_experts - 1)
            ch2.genes['hyperparameters'][max_hist_len_n_bit:] = [1 if i == index_to_set else 0 for i in range(n_KAN_experts)]

    return ch1, ch2

def mutation(conf, chromosome, mutation_rate, n_features, max_hist_len_n_bit, n_KAN_experts, model_name):
    # Mutate features
    chromosome.genes['features'] = [
        abs(gene - 1) if random.random() < mutation_rate else gene
        for gene in chromosome.genes['features']
    ]

    if conf['GA_type']==2:
        if model_name == "DenseRMoK":
            chromosome.genes['features'][n_features-14:n_features-14+5] = [1, 1, 1, 1, 1]

            if sum(chromosome.genes['hyperparameters'][max_hist_len_n_bit:])==0:
                index_to_set = random.randint(0, n_KAN_experts - 1)
                chromosome.genes['hyperparameters'][max_hist_len_n_bit:] = [1 if i == index_to_set else 0 for i in range(n_KAN_experts)]

    return chromosome

def genetic_algorithm(training_conf, conf):
    population = create_initial_population(conf)    
    best_performers = []
    all_populations = []

    # Prepare for table
    table_total_generations = PrettyTable()
    table_total_generations.field_names = ["Generation", "Features", "Hyperparameters", "Fitness"]
    table_total_generations.title = f"{conf['start_end_string']} for stock {conf['dataset_name']} with model {conf['model_name']} and GA {str(conf['GA_type'])}"

    pop_size = conf['population_size']

    for generation in range(1, 1+conf['total_generations']):

        list_ind = [fitness_function(ind, training_conf, conf) for ind in population]

        table_each_generation = PrettyTable()
        
        table_each_generation.field_names = ["Chromosome ID", "Features", "Hyperparameters", "Fitness"]
        table_each_generation.add_rows([index+1, ''.join(str(bit) for bit in element.genes['features']), ''.join(str(bit) for bit in element.genes['hyperparameters']), element.fitness] for index, element in enumerate(list_ind))
        table_each_generation.title = f"Generation {generation}: {conf['start_end_string']} for stock {conf['dataset_name']} with model {conf['model_name']} and GA {str(conf['GA_type'])}"

        print(table_each_generation)

        # Store the best performer of the current generation
        best_individual = max(population, key=lambda ch: ch.fitness)
        best_performers.append((best_individual, best_individual.fitness))
        all_populations.append(population[:])

   
        table_total_generations.add_row([generation, ''.join(str(bit) for bit in best_individual.genes['features']), ''.join(str(bit) for bit in best_individual.genes['hyperparameters']), best_individual.fitness])
            
        all_fitnesses = [ch.fitness for ch in population]
        population, pop_size = selection(population, all_fitnesses, pop_size)

        next_population = []

        for i in range(0, pop_size, 2):
            if len(population)>=2:
                parent1 = population[i]
                parent2 = population[i + 1]
            else: 
                next_population.append(population[i])
                break
            

            if generation != conf['total_generations'] :
                child1, child2 = inter_chromosome_crossover(conf, parent1, parent2, conf['total_n_features'], conf['n_hyperparameters'], conf['max_hist_len_n_bit'], conf['n_KAN_experts'], conf['model_name'])

            if generation != conf['total_generations'] :
                next_population.append(mutation(conf, child1, 0.05, conf['total_n_features'], conf['max_hist_len_n_bit'], conf['n_KAN_experts'], conf['model_name']))
                next_population.append(mutation(conf, child2, 0.05, conf['total_n_features'], conf['max_hist_len_n_bit'], conf['n_KAN_experts'], conf['model_name']))


        # Replace the old population with the new one, preserving the best individual
        next_population[0] = best_individual
        population = next_population


    print(table_total_generations)

    generations_list = range(1, len(best_performers) + 1)

    # Plot the fitness values over generations
    best_fitness_values = [fit[1] for fit in best_performers]
    min_fitness_values = [min([ch.fitness for ch in population]) for population in all_populations]
    max_fitness_values = [max([ch.fitness for ch in population]) for population in all_populations]

    plt.plot(generations_list, best_fitness_values, label='Best Fitness', color='black')
    plt.fill_between(generations_list, min_fitness_values, max_fitness_values, color='gray', alpha=0.5, label='Fitness Range')
    plt.xticks(generations_list)
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title(f'Fitness Over Generations for {conf["dataset_name"]}')
    plt.legend()
    plt.savefig(f"{conf['start_end_string']}/plots/GA_{str(conf['GA_type'])}_{conf['dataset_name']}.png")
    plt.close()

    best_ch = max(population, key=lambda ch: ch.fitness) 
    var_num, indicators_list_01, hist_len, hist_len_list_01, KAN_experts_list_01 = decode(best_ch, conf)

    return var_num, indicators_list_01, hist_len, hist_len_list_01, KAN_experts_list_01

class TrainLossLoggerCallback(Callback):
    def __init__(self):
        super().__init__()
        self.train_losses = []

    def on_train_epoch_end(self, trainer, pl_module):
        """
        Called at the end of the training epoch.
        Collects the average training loss and appends it to the train_losses list.
        """
        # Retrieve the average training loss from callback_metrics
        avg_loss = trainer.callback_metrics.get('train/loss')
        if avg_loss is not None:
            self.train_losses.append(avg_loss.item())

class FinalResultLoggerCallback(Callback):
    def __init__(self, optimal, filename="final_results.csv"):
        super().__init__()
        self.optimal = optimal
        self.filename = filename
        # Write header if the file does not exist

        if self.optimal:
            if not os.path.exists(self.filename):
                with open(self.filename, "w") as f:
                    header = ("train_length,"
                            "train_average_daily_return,train_cumulative_return,"
                            "train_downside_deviation,"
                            "train_total_profits,train_loss_days,"
                            "final_train_loss,"
                            "test_length,"
                            "test_average_daily_return,test_cumulative_return,"
                            "test_downside_deviation,"
                            "test_total_profits,test_loss_days,"
                            "test_custom_loss,test_error_percentage,test_mae,test_mse\n")

                    f.write(header)

    def on_test_end(self, trainer, pl_module):

        if self.optimal:
            # Only log from the main process
            if trainer.global_rank != 0:
                return
            
            # Retrieve the final training loss and trading metrics from the model
            final_train_loss = getattr(pl_module, 'final_train_loss', "NA")
            if hasattr(pl_module, 'final_train_metrics'):
                train_length = pl_module.final_train_metrics.get('length', "NA")
                train_avg_return = pl_module.final_train_metrics.get('average_daily_return', "NA")
                train_cum_return = pl_module.final_train_metrics.get('cumulative_return', "NA")
                train_loss_days = pl_module.final_train_metrics.get('loss_days', "NA")
                train_total_profits = pl_module.final_train_metrics.get('total_profits', "NA")
                train_downside_deviation = pl_module.final_train_metrics.get('downside_deviation', "NA")
            else:
                train_avg_return = train_cum_return = train_loss_days = train_total_profits = train_downside_deviation = "NA"

            # Extract the final metrics from trainer.callback_metrics

            metrics = trainer.callback_metrics
            test_length = metrics.get("test/length", "NA")
            test_avg_return = metrics.get("test/average_daily_return", "NA")
            test_cum_return = metrics.get("test/cumulative_return", "NA")
            test_custom_loss = metrics.get("test/custom_loss", "NA")
            test_error_percentage = metrics.get("test/error_percentage", "NA")
            test_loss_days = metrics.get("test/loss_days", "NA")
            test_mae = metrics.get("test/mae", "NA")
            test_mse = metrics.get("test/mse", "NA")
            test_total_profits = metrics.get("test/total_profits", "NA")
            test_downside_deviation = metrics.get("test/downside_deviation", "NA")


            line = f"{train_length}, {train_avg_return}, {train_cum_return}," \
                f"{train_downside_deviation}, {train_total_profits}, {train_loss_days}," \
                f"{final_train_loss}, {test_length}, {test_avg_return}, {test_cum_return}," \
                f"{test_downside_deviation}, {test_total_profits}, {test_loss_days}, {test_custom_loss}," \
                f"{test_error_percentage}, {test_mae}, {test_mse}\n"
                        
            with open(self.filename, "a") as f:
                f.write(line)

class TestLossLoggerCallback(Callback):
    def __init__(self):
        super().__init__()
        self.test_losses = []

    def on_test_epoch_end(self, trainer, pl_module):
        avg_loss = trainer.callback_metrics.get('test/custom_loss')
        
        if avg_loss is not None:
            self.test_losses.append(avg_loss.item())

    def get_last_test_loss(self):
        return self.test_losses[-1]

def train_init(hyper_conf, conf):
    if hyper_conf is not None:
        for k, v in hyper_conf.items():
            conf[k] = v
    conf['conf_hash'] = cal_conf_hash(conf, hash_len=10)

    L.seed_everything(conf["seed"])
    save_dir = f"{conf['save_root']}/{conf['model_name']}/GA{conf['GA_type']}"
    output_dir = save_dir
    os.makedirs(output_dir, exist_ok=True)

    callbacks = [
        TrainLossLoggerCallback(),
        TestLossLoggerCallback(), 
        FinalResultLoggerCallback(conf['optimal'], filename=os.path.join(save_dir, f"{conf['dataset_name']}.csv"))
    ]

    trainer = L.Trainer(
        devices=conf["devices"],
        precision=conf["precision"] if "precision" in conf else "32-true",
        logger=False,
        callbacks=callbacks,
        max_epochs=conf["max_epochs"],
        gradient_clip_algorithm=conf["gradient_clip_algorithm"] if "gradient_clip_algorithm" in conf else "norm", # Not used
        gradient_clip_val=conf["gradient_clip_val"], # Not used
        default_root_dir=conf["save_root"], 
        limit_val_batches=0, # Disable validation
        check_val_every_n_epoch=0, # No validation every n epoch
    )

    data_module = DataInterface(**conf)
    model = LTSFRunner(**conf)

    return trainer, data_module, model, callbacks[1]

def train_func(trainer, data_module, model, callback_testloss):
    trainer.fit(model=model, datamodule=data_module)
    trainer.test(model=model, datamodule=data_module)

    model.train_plot_losses()
    model.test_plot_losses()

    return trainer, data_module, model, callback_testloss.get_last_test_loss(), model.daily_return_multiplication_train_list, model.daily_return_multiplication_test_list


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_root", default="dataset", type=str, help="data root")
    parser.add_argument("-s", "--save_root", default="save", type=str, help="save root")
    parser.add_argument("--devices", default=1, type=int, help="device' id to use")
    parser.add_argument("--use_wandb", default=0, type=int, help="use wandb")
    parser.add_argument("--seed", type=int, default=1, help="seed")
    parser.add_argument("--model_name", default="DenseRMoK", type=str, help="Model name")
    parser.add_argument("--optimal", default="0", type=int, help="Whether this model is optimal")
    parser.add_argument("--GA_type", default="2", type=int, help="0: no GA // 1: 2017 GA // 2: our GA")
    parser.add_argument("--revin_affine", default=False, type=bool, help="Use revin affine") 

    parser.add_argument("--lr", default=0.001, type=float, help="Learning rate")
    parser.add_argument("--batch_size", default=64, type=int, help="Batch size")
    parser.add_argument("--max_epochs", default=8, type=int, help="Maximum number of epochs")
    parser.add_argument("--optimizer", default="AdamW", type=str, help="Optimizer type")
    parser.add_argument("--optimizer_betas", default=(0.9, 0.999), type=eval, help="Optimizer betas")
    parser.add_argument("--optimizer_weight_decay", default=1e-2, type=float, help="Optimizer weight decay")
    parser.add_argument("--lr_scheduler", default='StepLR', type=str, help="Learning rate scheduler")
    parser.add_argument("--lr_step_size", default=32, type=int, help="Learning rate step size")
    parser.add_argument("--lr_gamma", default=0.64, type=float, help="Learning rate gamma")
    parser.add_argument("--gradient_clip_val", default=5, type=float, help="Gradient clipping value") # // Not used
    parser.add_argument("--val_metric", default="val/loss", type=str, help="Validation metric")
    parser.add_argument("--test_metric", default="test/mae", type=str, help="Test metric")
    parser.add_argument("--es_patience", default=10, type=int, help="Early stopping patience") # // Not used
    parser.add_argument("--num_workers", default=1, type=int, help="Number of workers for data loading")

    parser.add_argument("--population_size", default=64, type=int, help="Population Size for GA")
    parser.add_argument("--total_n_features", default=50, type=int, help="Total number of features for GA") 
    parser.add_argument("--min_hist_len", default=4, type=int, help="Minimum window size allowed")
    parser.add_argument("--max_hist_len", default=64, type=int, help="Maximum window size allowed")
    parser.add_argument("--n_KAN_experts", default=5, type=int, help="Number of KAN experts to be used")
    parser.add_argument("--drop", default=0.2, type=float, help="Dropout rate for input features in KAN")

    parser.add_argument("--pred_len", default=1, type=int, help="Number of predicted made each time (should be fixed)")
    parser.add_argument("--data_split", default=[500, 0, 100], type=list, help="Train-Val-Test Ratio (Val should be fixed to 0)")
    parser.add_argument("--freq", default=1440, type=int, help="(should be fixed)") 

    args = parser.parse_args()
    args.max_hist_len_n_bit = math.floor(math.log2( (args.max_hist_len-args.min_hist_len) / 4 + 1 ))
    args.n_hyperparameters = args.max_hist_len_n_bit + args.n_KAN_experts
    args.total_generations = math.floor(math.log2(args.population_size))
    args.start_end_string = ""

    ticker_symbols = ['AMD'] #, 'MSFT', 'ORCL', 'AMD', 'CSCO', 'ADBE', 'IBM', 'TXN', 'AMAT', 'MU'] #, 'ADI', 'INTC', 'LRCX', 'KLAC', 'MSI', 'GLW', 'HPQ', 'TYL', 'PTC', 'JNJ']
    ticker_symbols = ['AAPL', 'MSFT', 'ORCL', 'AMD', 'CSCO', 'ADBE', 'IBM', 'TXN', 'AMAT', 'MU', 'ADI', 'INTC', 'LRCX', 'KLAC', 'MSI', 'GLW', 'HPQ', 'TYL', 'PTC', 'JNJ']

    all_df = pd.read_csv("dataset/data_for_dates.csv")
    max_iteration = math.floor(3242 // args.data_split[2])
    max_iteration = 1
    for symbol in ticker_symbols:
        print(f"Start for stock {color.BOLD}{symbol}{color.END}:")

        for model in ["LSTM", "MLP", "DenseRMoK"]:
            args.model_name = model
            total_check = 0

            args.daily_return_multiplication_train_list = []
            args.daily_return_multiplication_test_list = []

            for i in range(0, max_iteration):
                if total_check>=2520:
                    break
                else:

                    start_index, end_index = (0, sum(args.data_split)) if i == 0 else (start_index + args.data_split[2], end_index + args.data_split[2])
                    start_date, end_date = all_df.loc[start_index, "Date"],  all_df.loc[end_index, "Date"]

                    print(f"From {color.BOLD}{start_date}{color.END} To {color.BOLD}{end_date}{color.END}:")
                    read_data(start_date, end_date)

                    args.dataset_name = symbol

                    start_end_string = f"{start_date}_{end_date}"
                    args.start_end_string = start_end_string
                    df = pd.read_csv(f"{start_end_string}/dataset/{symbol}/all_data.csv")
                    args.var_num = df.shape[1] - 1 # Exclude the dates column

                    args.indicators_list_01 = [1 for i in range(args.total_n_features)] 

                    args.hist_len = 4
                    args.hist_len_list_01 = [1 for i in range(args.max_hist_len_n_bit)]

                    if args.model_name == "DenseRMoK":
                        args.KAN_experts_list_01 = [1 for i in range(args.n_KAN_experts)] 
                    else:
                        args.KAN_experts_list_01 = [0 for i in range(args.n_KAN_experts)]

                    training_conf = {
                        "seed": int(args.seed),
                        "data_root": f"dataset/{symbol}",
                        "save_root": args.save_root,
                        "devices": args.devices,
                        "use_wandb": args.use_wandb
                    }

                    print(f"{color.BOLD}{args.model_name} with GA type {args.GA_type} is built: {color.END}")

                    if (args.GA_type==1 or args.GA_type==2):  
                        if args.GA_type==1:
                            args.optimal = 0
                            args.var_num, args.indicators_list_01, args.hist_len, args.hist_len_list_01, args.KAN_experts_list_01 = genetic_algorithm(training_conf, vars(args))

                            print("Optimal choices: ")
                            print(args.var_num)
                            print(args.indicators_list_01)

                            args.hist_len_list_01 =  [1 for i in range(args.max_hist_len_n_bit)]
                            args.hist_len = args.min_hist_len + 4 * sum(bit << i for i, bit in enumerate(reversed(args.hist_len_list_01)))
                            print(args.hist_len)
                            print(args.hist_len_list_01)

                            if args.model_name == "DenseRMoK":
                                print(args.KAN_experts_list_01)

                        else:
                            args.optimal = 0
                            args.var_num, args.indicators_list_01, args.hist_len, args.hist_len_list_01, args.KAN_experts_list_01 = genetic_algorithm(training_conf, vars(args))

                            print("Optimal choices: ")
                            print(args.var_num)
                            print(args.indicators_list_01)

                            print(args.hist_len)
                            print(args.hist_len_list_01)

                            if args.model_name == "DenseRMoK":
                                print(args.KAN_experts_list_01)
                    
                        print("Optimal model: ")

                        args.optimal = 1
                        trainer, data_module, model, callback = train_init(training_conf, vars(args))
                        trainer, data_module, model, test_loss, daily_return_multiplication_train_list, daily_return_multiplication_test_list = train_func(trainer, data_module, model, callback)
                        total_testing_trading_days = args.data_split[2] - args.hist_len

                    else:
                        args.optimal = 1
                        
                        args.indicators_list_01 = [1 for i in range(args.total_n_features)] 
                        args.var_num = sum(args.indicators_list_01)
                        args.hist_len_list_01 =  [1 for i in range(args.max_hist_len_n_bit)]
                        args.hist_len = args.min_hist_len + 4 * sum(bit << i for i, bit in enumerate(reversed(args.hist_len_list_01)))
                        
                        if args.model_name == "DenseRMoK":
                            args.KAN_experts_list_01 = [1 for i in range(args.n_KAN_experts)] 

                    # *****************************

                        print(args.var_num)
                        print(args.indicators_list_01)
                        print(args.hist_len)
                        print(args.hist_len_list_01)

                        if args.model_name == "DenseRMoK":
                            print(args.KAN_experts_list_01)

                        trainer, data_module, model, callback = train_init(training_conf, vars(args))
                        trainer, data_module, model, test_loss, daily_return_multiplication_train_list, daily_return_multiplication_test_list = train_func(trainer, data_module, model, callback)
                        total_testing_trading_days = args.data_split[2] - args.hist_len

                    total_check += total_testing_trading_days
                    print("\n")

                    args.daily_return_multiplication_train_list.append(daily_return_multiplication_train_list)
                    args.daily_return_multiplication_test_list.append(daily_return_multiplication_test_list)


            daily_return_multiplication_train = 1
            for train_list in args.daily_return_multiplication_train_list:
                for item in train_list:
                    daily_return_multiplication_train = daily_return_multiplication_train * (1+item)
            
            daily_return_multiplication_train = daily_return_multiplication_train - 1


            daily_return_multiplication_test = 1
            for test_list in args.daily_return_multiplication_test_list:
                for item in test_list:
                    daily_return_multiplication_test = daily_return_multiplication_test * (1+item)
            
            daily_return_multiplication_test = daily_return_multiplication_test - 1

            filename = f"{args.save_root}/{args.model_name}/GA{args.GA_type}/{args.dataset_name}_total_return.csv"
            if not os.path.exists(filename):
                with open(filename, "w") as f:
                    header = ("train_total_return,test_total_return\n")
                    f.write(header)
            
            line = f"{daily_return_multiplication_train}, {daily_return_multiplication_test}\n"
            with open(filename, "a") as f:
                f.write(line)

        print(f"End for stock {color.BOLD}{symbol}{color.END}")
