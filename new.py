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
            if len(population)>=2:
                parent1 = population[i]
                parent2 = population[i + 1]
            else: 
                next_population.append(population[i])

            if ( generation == (conf['total_generations']//2) or ((len(fg) >= 2) and (abs(fg[-1]-fg[-2]) >= 1e-3)) ):
                if generation != conf['total_generations'] :
                    parent1 = intra_chromosome_crossover(parent1, conf['total_n_features'], conf['n_hyperparameters'], conf['max_hist_len_n_bit'], conf['n_KAN_experts'])

            if generation != conf['total_generations'] :
                child1, child2 = inter_chromosome_crossover(parent1, parent2, conf['total_n_features'], conf['n_hyperparameters'], conf['max_hist_len_n_bit'], conf['n_KAN_experts'])

            if generation != conf['total_generations'] :
                next_population.append(mutation(child1, 0.1, conf['total_n_features'], conf['max_hist_len_n_bit'], conf['n_KAN_experts']))
                next_population.append(mutation(child2, 0.1, conf['total_n_features'], conf['max_hist_len_n_bit'], conf['n_KAN_experts']))


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
    plt.savefig(f'{conf['start_end_string']}/plots/GA_{conf["dataset_name"]}.png')
    plt.close()

    best_ch = max(population, key=lambda ch: ch.fitness) 
    var_num, indicators_list_01, hist_len, hist_len_list_01, KAN_experts_list_01 = decode(best_ch, conf)

    return var_num, indicators_list_01, hist_len, hist_len_list_01, KAN_experts_list_01
