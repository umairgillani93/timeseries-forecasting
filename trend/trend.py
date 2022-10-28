import os 
import seaborn as sns
import numpy as np 
import pandas as pd 
from pathlib import Path
import statsmodels
import matplotlib.pyplot as plt

from warnings import simplefilter
simplefilter('ignore')

print('\nImports successful')

# Set matplotlib defaults
plt.style.use('seaborn-whitegrid')
plt.rc("figure", autolayout=True, figsize=(11,5))
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=14,
    titlepad=1
    )

# Define plotting parameters
plot_params = dict(
    color='0.75',
    style='.-',
    markeredgecolor='0.25',
    markerfacecolor='0.25',
    legend=False
    )


# Load tunnel traffice dataset
path = os.getenv('HOME') + '/datasets/book_sales_kaggle/tunnel.csv'
tunnel = pd.read_csv(path, parse_dates=["Day"])
tunnel = tunnel.set_index('Day').to_period()

print(f'datafrmae: {tunnel.head()}')

# Let's create a moving average using rolling method from pandas
# Since our series has daily observations, let's choose a window size of 365

moving_avg = tunnel.rolling(
    window=365,
    center=True,
    min_periods=183,
    ).mean()

ax = tunnel.plot(style='.', color='0.5')
moving_avg.plot(
    ax=ax, linewidth=3, title='Tunnel Traffic - 365-Day Moving Average', legend=False)

plt.show()

dp = statsmodels.tsa.deterministic.DeterministicProcess(
    index = tunnel.index, # dates from training data
    constant = True, # dummy feature for bias
    order = 1, # the time dummy trend
    drop = True # drop terms if necessory to avoid co-linearity
    )

X = dp.in_sample() # creates features for the dates given in the 'index' argument


X.head()
