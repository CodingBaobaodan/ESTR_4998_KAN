import pandas as pd
import numpy as np

all_df = pd.read_csv("dataset/data_for_dates.csv")
start_index, end_index = 0, 600

start_index, end_index = start_index + 100, end_index + 100
start_date, end_date = all_df.loc[start_index, "Date"],  all_df.loc[end_index, "Date"]

print(start_date, end_date)


var_x = np.array([[1, 2, 3], [2, 3, 4]])
print(len(var_x))