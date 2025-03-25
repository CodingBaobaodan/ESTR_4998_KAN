import pandas as pd
import numpy as np

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

symbol = "AAPL"
print(f"For stock {color.BOLD}{symbol}{color.END}:")


import warnings
import yfinance as yf
warnings.filterwarnings("ignore", category=UserWarning)
# Your code here
data = yf.download("AAPL", start="2023-01-01", end="2023-12-31")