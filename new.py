import pandas as pd

df = pd.read_csv("dataset/AAPL/all_data.csv")
print(df.loc[0+500, "Date"])