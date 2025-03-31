daily_return_multiplication_train_list = []

def func(optimal, list_metrics):
    
    
    if optimal:
        global daily_return_multiplication_train_list
        daily_return_multiplication_train_list.append(list_metrics)
        print(daily_return_multiplication_train_list)

# Example call to func
func(True, [1, 2, 3])