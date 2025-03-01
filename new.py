import pandas as pd
import numpy as np
import os

df = pd.read_csv("/Users/jeremyity/Downloads/combined.csv")
min_val = df.min().values
max_val = df.max().values


print(min_val)

#self.register_buffer('min', torch.tensor(stat['min']).float())