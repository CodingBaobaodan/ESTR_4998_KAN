import random
from prettytable import PrettyTable
import numpy as np
import matplotlib.pyplot as plt


seed_value = 42
random.seed(seed_value)

# Define a basic structure for Chromosome and Population
class Chromosome:
    def __init__(self, features, hyperparameters):
        self.genes = {
            'features': features,
            'hyperparameters': hyperparameters,
        }

def decode(ind):
    window_size_list, KAN_layers_list = ind.genes['hyperparameters'][:5], ind.genes['hyperparameters'][5:]
    window_size = int("".join(map(str, window_size_list)), 2)
    (WaveKAN, NaiveFourierKAN, JacobiKAN, ChebyKAN, TaylorKAN, RBFKAN) = KAN_layers_list

    return window_size, WaveKAN, NaiveFourierKAN, JacobiKAN, ChebyKAN, TaylorKAN, RBFKAN

def fitness_function(ind, trainer=None, data_module=None, model=None):
    # TO DO // Connect Trainer, data_module, model

    selected_list = ind.genes['features'] 
    #selected_columns = [col for col, sel in zip(df.columns, selected_list) if sel == 1]
    #selected_df = df[selected_columns]


    window_size, WaveKAN, NaiveFourierKAN, JacobiKAN, ChebyKAN, TaylorKAN, RBFKAN = decode(ind)
    
    return 1 # TO DO // Actual Loss

def create_initial_population(population_size):
    # Initialize a population of chromosomes with random values
    population = []
    for _ in range(population_size):
        # 5 (index) : ('^GSPC', 'Open')	('^GSPC', 'High')	('^GSPC', 'Low')	('^GSPC', 'Close')	('^GSPC', 'Volume')	
        # 31 (technical indicators) : Bollinger_Bands_Upper	Bollinger_Bands_Middle	Bollinger_Bands_Lower	DEMA	Midpoint	Midpoint_Price	T3_Moving_Average	ADX	Absolute_Price_Oscillator	Aroon_Up	Aroon_Down	Aroon_Oscillator	Balance_of_Power	CCI	Chande_Momentum_Oscillator	MACD	MACD_Signal	MACD_Histogram	Money_Flow_Index	Normalized_Average_True_Range	Chaikin_A/D_Line	Chaikin_A/D_Oscillator	Median_Price	Typical_Price	Weighted_Closing_Price	Hilbert_Dominant_Cycle_Phase	Hilbert_Phasor_Components_Inphase	Hilbert_Phasor_Components_Quadrature	Hilbert_SineWave	Hilbert_LeadSineWave	Hilbert_Trend_vs_Cycle_Mode	
        # 5( stock) : Open	High	Low	Close	Volume	
        # 9 (macro indicators) : M2	S&P CoreLogic Case-Shiller U.S. National Home Price Index	All-Transactions House Price Index for the United States	M1	Consumer Price Index for All Urban Consumers: All Items in U.S. City Average	Trade Balance: Goods and Services, Balance of Payments Basis	New Privately-Owned Housing Units Started: Total Units	Domestic Auto Production	New One Family Houses Sold

        features = [random.choice([0, 1]) for _ in range(50)]

        window_size_list = [random.choice([0, 1]) for _ in range(5)] # 0 to 31
        WaveKAN = [random.choice([0, 1])]
        NaiveFourierKAN = [random.choice([0, 1])]
        JacobiKAN = [random.choice([0, 1])]
        ChebyKAN = [random.choice([0, 1])]
        TaylorKAN = [random.choice([0, 1])]
        RBFKAN = [random.choice([0, 1])]
        KAN_layers_list = WaveKAN + NaiveFourierKAN + JacobiKAN + ChebyKAN + TaylorKAN + RBFKAN
        hyperparameters = window_size_list + KAN_layers_list

        population.append(Chromosome(features, hyperparameters))

    return population


def selection(population, fitnesses, tournament_size=3):
    selected = []
    for _ in range(len(population)):
        tournament = random.sample(list(zip(population, fitnesses)), tournament_size)
        winner = max(tournament, key=lambda x: x[1])[0]
        selected.append(winner)
    return selected


def intra_chromosome_crossover(ch1, n_features, n_hyperparameters):
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
    
    return ch1


def inter_chromosome_crossover(ch1, ch2, n_features, n_hyperparameters):

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
    
    return ch1, ch2


def mutation(chromosome, mutation_rate):
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

    return chromosome


def genetic_algorithm(population_size, total_generations, n_features, n_hyperparameters, trainer, data_module, model):
    population = create_initial_population(population_size)
    
    best_performers = []
    all_populations = []

    # Initialize mutation_rate and fg lists with initial values
    fg = [0] # // TO DO 
    mutation_rate = [0.1]

    # Prepare for table
    table = PrettyTable()
    table.field_names = ["Generation", "Features", "Hyperparameters", "Fitness"]

    for generation in range(total_generations):
        fitnesses = [fitness_function(ind, trainer, data_module, model) for ind in population]

        # Store the best performer of the current generation
        best_individual = max(population, key=fitness_function)
        best_fitness = fitness_function(best_individual, trainer, data_module, model)
        best_performers.append((best_individual, best_fitness))
        all_populations.append(population[:])
        table.add_row([generation + 1, "".join(map(str, best_individual.genes['features'])), "".join(map(str, best_individual.genes['hyperparameters'])), best_fitness])

        population = selection(population, fitnesses)

        next_population = []
        for i in range(0, len(population)-1, 2):
            parent1 = population[i]
            parent2 = population[i + 1]

            if (generation == (total_generations//2)) or ((len(fg) >= 2) and ((fg[-1]-fg[-2]) == 0)): # // TO DO 
                parent1 = intra_chromosome_crossover(parent1, n_features, n_hyperparameters)

            child1, child2 = inter_chromosome_crossover(parent1, parent2, n_features, n_hyperparameters)

            # Calculate increment
            if len(fg) >= 2 and (fg[-1] - fg[-2]) != 0:
                increment = 100 * mutation_rate[generation] / (fg[-1] - fg[-2]) # TO DO
            else:
                increment = 0

            if increment > 0:
                mg = mutation_rate[generation] + increment
            else:
                mg = mutation_rate[generation] - increment

            if (mg>=1) or (mg<=0):
                mg = mutation_rate[generation]

            mutation_rate.append(mg)

            next_population.append(mutation(child1, mg))
            next_population.append(mutation(child2, mg))

        # Replace the old population with the new one, preserving the best individual
        next_population[0] = best_individual
        population = next_population

    # Print the table
    print(table)

    # Plot the population of one generation (last generation)
    final_population = all_populations[-1]
    final_fitnesses = [fitness_function(ind, trainer, data_module, model) for ind in final_population]

    # Plot the values of a, b, and c over generations
    generations_list = range(1, len(best_performers) + 1)

    # Plot the fitness values over generations
    best_fitness_values = [fit[1] for fit in best_performers]
    min_fitness_values = [min([fitness_function(ind, trainer, data_module, model) for ind in population]) for population in all_populations]
    max_fitness_values = [max([fitness_function(ind, trainer, data_module, model) for ind in population]) for population in all_populations]
    fig, ax = plt.subplots()
    ax.plot(generations_list, best_fitness_values, label='Best Fitness', color='black')
    ax.fill_between(generations_list, min_fitness_values, max_fitness_values, color='gray', alpha=0.5, label='Fitness Range')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Fitness')
    ax.set_title('Fitness Over Generations')
    ax.legend()
    plt.savefig('plots/GA.png')

    return max(population, key=fitness_function)


