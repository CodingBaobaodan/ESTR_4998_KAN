table_each_generation.field_names = ["Chromosome ID", "Features", "Hyperparameters", "Fitness"]
table_each_generation.add_rows(
    
    
[index+1, ''.join(str(bit) for bit in element.genes['features']), ''.join(str(bit) for bit in element.genes['hyperparameters']), element.fitness] 
for index, element in list(enumerate(list_ind))
)