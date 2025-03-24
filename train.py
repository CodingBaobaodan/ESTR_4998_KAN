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

def read_data(start_date, end_date):
    window = 30

    ticker_symbols = ['AAPL', 'MSFT', 'ORCL', 'AMD', 'CSCO', 'ADBE', 
                    'IBM', 'TXN', 'AMAT', 'MU', 'ADI', 'INTC', 
                    'LRCX', 'KLAC', 'MSI', 'GLW', 'HPQ', 'TYL', 
                    'PTC', 'JNJ']

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
    top_k_correlations = cap_weighted_correlation_plots(cap_weighted_composite_index_df, macro_df, k=10) 


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

# Define a basic structure for Chromosome and Population
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
    
    hist_len_list_01, KAN_experts_list_01 = ind.genes['hyperparameters'][:conf['max_hist_len_n_bit']], ind.genes['hyperparameters'][conf['max_hist_len_n_bit']:]
    hist_len = conf['min_hist_len'] + 4 * sum(bit << i for i, bit in enumerate(reversed(hist_len_list_01)))

    return var_num, indicators_list_01, hist_len, hist_len_list_01, KAN_experts_list_01

def fitness_function(ind, training_conf, conf):
    conf['var_num'], conf['indicators_list_01'], conf['hist_len'], conf['hist_len_list_01'], conf['args.KAN_experts_list_01'] = decode(ind, conf)
    print(f"{conf['var_num']} features are selected")
    print(conf['indicators_list_01'])
    print(f"window size: {conf['hist_len']}")
    print(conf['hist_len_list_01'])

    print("Experts Taylor, Wavelet, Jacobi, Cheby, RBF, NaiveFourier", conf['args.KAN_experts_list_01']) # KAN Experts to be changed

    trainer, data_module, model, callback = train_init(training_conf, conf)
    trainer, data_module, model, test_loss = train_func(trainer, data_module, model, callback)

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

        hist_len_list_01 = [random.choice([0, 1]) for _ in range(conf['max_hist_len_n_bit'])] 
        KAN_experts_list_01 = [random.choice([0, 1]) for _ in range(conf['n_KAN_experts'])] 

        if sum(KAN_experts_list_01)==0:
            index_to_set = random.randint(0, conf['n_KAN_experts'] - 1)
            KAN_experts_list_01 = [1 if i == index_to_set else 0 for i in range(conf['n_KAN_experts'])]

        hyperparameters = hist_len_list_01 + KAN_experts_list_01

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

def intra_chromosome_crossover(ch1, n_features, n_hyperparameters, max_hist_len_n_bit, n_KAN_experts):
    n = min(n_features, n_hyperparameters)

    features_filter = [1] * n + [0] * (n_features - n)
    random.shuffle(features_filter)

    selected_indices = [i for i, val in enumerate(features_filter) if val == 1]
    not_selected_index = [i for i in range(n)]

    # Swap the selected pairs
    for idx in selected_indices:
        swap_index = random.sample(not_selected_index, 1)[0]
        not_selected_index.remove(swap_index)
        ch1.genes['features'][idx], ch1.genes['hyperparameters'][swap_index] = ch1.genes['hyperparameters'][swap_index], ch1.genes['features'][idx]
    
    ch1.genes['features'][n_features-14:n_features-14+5] = [1, 1, 1, 1, 1] 

    if sum(ch1.genes['hyperparameters'][max_hist_len_n_bit:])==0:
        index_to_set = random.randint(0, n_KAN_experts - 1)
        ch1.genes['hyperparameters'][max_hist_len_n_bit:] = [1 if i == index_to_set else 0 for i in range(n_KAN_experts)]

    print("Intra Chromosome Crossover applied")
    return ch1

def inter_chromosome_crossover(ch1, ch2, n_features, n_hyperparameters, max_hist_len_n_bit, n_KAN_experts):

    features1 = ch1.genes['features']
    hyperparameters1 = ch1.genes['hyperparameters']
    
    features2 = ch2.genes['features']
    hyperparameters2 = ch2.genes['hyperparameters']
    
    crossover_point1 = random.randint(0, n_features - 1)
    crossover_point2 = random.randint(0, n_hyperparameters - 1)
    
    features1[crossover_point1:], features2[crossover_point1:] = features2[crossover_point1:], features1[crossover_point1:]
    hyperparameters1[crossover_point2:], hyperparameters2[crossover_point2:] = hyperparameters2[crossover_point2:], hyperparameters1[crossover_point2:]
    
    ch1.genes['features'] = features1
    ch1.genes['hyperparameters'] = hyperparameters1
    
    ch2.genes['features'] = features2
    ch2.genes['hyperparameters'] = hyperparameters2

    ch1.genes['features'][n_features-14:n_features-14+5] = [1, 1, 1, 1, 1] 
    ch2.genes['features'][n_features-14:n_features-14+5] = [1, 1, 1, 1, 1] 

    if sum(ch1.genes['hyperparameters'][max_hist_len_n_bit:])==0:
        index_to_set = random.randint(0, n_KAN_experts - 1)
        ch1.genes['hyperparameters'][max_hist_len_n_bit:] = [1 if i == index_to_set else 0 for i in range(n_KAN_experts)]

    if sum(ch2.genes['hyperparameters'][max_hist_len_n_bit:])==0:
        index_to_set = random.randint(0, n_KAN_experts - 1)
        ch2.genes['hyperparameters'][max_hist_len_n_bit:] = [1 if i == index_to_set else 0 for i in range(n_KAN_experts)]

    print("Inter Chromosome Crossover applied")
    return ch1, ch2

def mutation(chromosome, mutation_rate, n_features, max_hist_len_n_bit, n_KAN_experts):
    # Mutate features
    chromosome.genes['features'] = [
        abs(gene - 1) if random.random() < mutation_rate else gene
        for gene in chromosome.genes['features']
    ]

    # Mutate hyperparameters
    chromosome.genes['hyperparameters'] = [
        abs(gene - 1) if random.random() < mutation_rate else gene
        for gene in chromosome.genes['hyperparameters']
    ]

    chromosome.genes['features'][n_features-14:n_features-14+5] = [1, 1, 1, 1, 1]

    if sum(chromosome.genes['hyperparameters'][max_hist_len_n_bit:])==0:
        index_to_set = random.randint(0, n_KAN_experts - 1)
        chromosome.genes['hyperparameters'][max_hist_len_n_bit:] = [1 if i == index_to_set else 0 for i in range(n_KAN_experts)]

    print("Mutation applied")
    return chromosome

def genetic_algorithm(training_conf, conf):
    population = create_initial_population(conf)    
    best_performers = []
    all_populations = []

    # Initialize mutation_rate and fg lists with initial values
    fg = [0] 

    # Prepare for table
    table_total_generations = PrettyTable()
    table_total_generations.field_names = ["Generation", "Features", "Hyperparameters", "Fitness"]

    pop_size = conf['population_size']

    for generation in range(1, 1+conf['total_generations']):
        print(f"Start Generation {generation}")

        list_ind = [fitness_function(ind, training_conf, conf) for ind in population]

        table_each_generation = PrettyTable()
        table_each_generation.field_names = ["Chromosome ID", "Features", "Hyperparameters", "Fitness"]
        table_each_generation.add_rows([index+1, ''.join(str(bit) for bit in element.genes['features']), ''.join(str(bit) for bit in element.genes['hyperparameters']), element.fitness] for index, element in list(enumerate(list_ind)))
        table_each_generation.title = f"Generation {generation}"
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
            parent1 = population[i]
            parent2 = population[i + 1]

            if ( generation == (conf['total_generations']//2) or ((len(fg) >= 2) and (abs(fg[-1]-fg[-2]) >= 1e-3)) ):
                if generation != conf['total_generations'] :
                    parent1 = intra_chromosome_crossover(parent1, conf['total_n_features'], conf['n_hyperparameters'], conf['max_hist_len_n_bit'], conf['n_KAN_experts'])

            if generation != conf['total_generations'] :
                child1, child2 = inter_chromosome_crossover(parent1, parent2, conf['total_n_features'], conf['n_hyperparameters'], conf['max_hist_len_n_bit'], conf['n_KAN_experts'])

            if generation != conf['total_generations'] :
                next_population.append(mutation(child1, 0.2, conf['total_n_features'], conf['max_hist_len_n_bit'], conf['n_KAN_experts']))
                next_population.append(mutation(child2, 0.2, conf['total_n_features'], conf['max_hist_len_n_bit'], conf['n_KAN_experts']))

        # Replace the old population with the new one, preserving the best individual
        next_population[0] = best_individual
        population = next_population
        fg.append(best_individual.fitness)

        print(f"That is all for Generation {generation} for stock {conf['dataset_name']}")

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
    plt.savefig(f'plots/GA_{conf["dataset_name"]}.png')


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
            # Append the average loss to the list
            self.train_losses.append(avg_loss.item())
            # Print the average loss for the epoch
            print(f", Average Train Loss = {avg_loss.item():.4f}")


class TestLossLoggerCallback(Callback):
    def __init__(self):
        super().__init__()
        self.test_losses = []

    def on_test_epoch_end(self, trainer, pl_module):
        avg_loss = trainer.callback_metrics.get('test/custom_loss')
        if avg_loss is not None:
            self.test_losses.append(avg_loss.item())
            print(f", Average Test Loss = {avg_loss.item():.4f}")
        
    def get_last_test_loss(self):
            return self.test_losses[-1]

def train_init(hyper_conf, conf):
    if hyper_conf is not None:
        for k, v in hyper_conf.items():
            conf[k] = v
    conf['conf_hash'] = cal_conf_hash(conf, hash_len=10)


    L.seed_everything(conf["seed"])
    save_dir = os.path.join(conf["save_root"], '{}_{}'.format(conf["model_name"], conf["dataset_name"]))
    if "use_wandb" in conf and conf["use_wandb"]:
        run_logger = WandbLogger(save_dir=save_dir, name=conf["conf_hash"], version='seed_{}'.format(conf["seed"]))
    else:
        run_logger = CSVLogger(save_dir=save_dir, name=conf["conf_hash"], version='seed_{}'.format(conf["seed"]))
    conf["exp_dir"] = os.path.join(save_dir, conf["conf_hash"], 'seed_{}'.format(conf["seed"]))

    callbacks = [
        # EarlyStopping(
        #     monitor=conf["val_metric"],
        #     mode='min',
        #     patience=conf["es_patience"],
        # ),
        LearningRateMonitor(logging_interval="epoch"),
        TrainLossLoggerCallback(),
        TestLossLoggerCallback(), 
    ]

    trainer = L.Trainer(
        devices=conf["devices"],
        precision=conf["precision"] if "precision" in conf else "32-true",
        logger=run_logger,
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


    return trainer, data_module, model, callbacks[2]

def train_func(trainer, data_module, model, callback):
    trainer.fit(model=model, datamodule=data_module)
    trainer.test(model=model, datamodule=data_module)

    model.train_plot_losses()
    model.test_plot_losses()

    return trainer, data_module, model, callback.get_last_test_loss()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_root", default="dataset", type=str, help="data root")
    parser.add_argument("-s", "--save_root", default="save", type=str, help="save root")
    parser.add_argument("--devices", default=1, type=int, help="device' id to use")
    parser.add_argument("--use_wandb", default=0, type=int, help="use wandb")
    parser.add_argument("--seed", type=int, default=1, help="seed")
    parser.add_argument("--model_name", default="DenseRMoK", type=str, help="Model name")
    parser.add_argument("--revin_affine", default=False, type=bool, help="Use revin affine") # // Check!

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
    parser.add_argument("--num_workers", default=10, type=int, help="Number of workers for data loading")

    parser.add_argument("--population_size", default=8, type=int, help="Population Size for GA")
    parser.add_argument("--total_generations", default=3, type=int, help="Total number of generations for GA")
    parser.add_argument("--total_n_features", default=50, type=int, help="Total number of features for GA") 
    parser.add_argument("--min_hist_len", default=4, type=int, help="Minimum window size allowed")
    parser.add_argument("--max_hist_len", default=64, type=int, help="Maximum window size allowed")
    parser.add_argument("--n_KAN_experts", default=6, type=int, help="Number of KAN experts to be used")

    parser.add_argument("--drop", default=0.2, type=float, help="Dropout rate for input features in KAN")

    parser.add_argument("--pred_len", default=1, type=int, help="Number of predicted made each time (should be fixed)")
    parser.add_argument("--data_split", default=[2000, 0, 500], type=list, help="Train-Val-Test Ratio (Val should be fixed to 0)")
    parser.add_argument("--freq", default=1440, type=int, help="(should be fixed)") 

    args = parser.parse_args()
    args.max_hist_len_n_bit = math.floor(math.log2( (args.max_hist_len-args.min_hist_len) / 4 + 1 ))
    args.n_hyperparameters = args.max_hist_len_n_bit + args.n_KAN_experts

    ticker_symbols = ['AAPL', 'MSFT', 'ORCL', 'AMD', 'CSCO', 'ADBE', 'IBM', 'TXN', 'AMAT', 'MU', 'ADI', 'INTC', 'LRCX', 'KLAC', 'MSI', 'GLW', 'HPQ', 'TYL', 'PTC', 'JNJ']

    
    for symbol in ticker_symbols:
        # Before GA
        args.dataset_name = symbol

        df = pd.read_csv(f"dataset/{symbol}/all_data.csv")
        args.var_num = df.shape[1] - 1 # Exclude the dates column

        args.indicators_list_01 = [1 for i in range(args.total_n_features)] 

        args.hist_len = 4
        args.hist_len_list_01 = [1 for i in range(args.max_hist_len_n_bit)]

        args.KAN_experts_list_01 = [1 for i in range(args.n_KAN_experts)] 

        training_conf = {
            "seed": int(args.seed),
            "data_root": f"dataset/{symbol}",
            "save_root": args.save_root,
            "devices": args.devices,
            "use_wandb": args.use_wandb
        }

        # GA
        print(f"For stock {symbol}:")
        print("Doing GA")
        args.var_num, args.indicators_list_01, args.hist_len, args.hist_len_list_01, args.KAN_experts_list_01 = genetic_algorithm(training_conf, vars(args))

        print("After GA, optimal choices: ")
        print(args.var_num)
        print(args.indicators_list_01)
        print(args.hist_len)
        print(args.hist_len_list_01)
        print(args.KAN_experts_list_01)

        print("Optimal model is finally trained below: ")
        trainer, data_module, model, callback = train_init(training_conf, vars(args))
        trainer, data_module, model, test_loss = train_func(trainer, data_module, model, callback)
        print("\n")

        print("Baselinee model is built: ")
        # // Check! Baseline Model