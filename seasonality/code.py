import os
import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt
from pathlib import Path
from warnings import filterwarnings
from sklearn.model_selection import LinearRegression
from statsmodels.tsa.deterministic import CalenderFourier, DeterministicProcess
filterwarnings('ignore')

print('Imports successful')

def fourier_transform(index, freq, order):
  # computer fourier transform to the 4thorder (8 new features)
  # for a series 'y' with daily observations and annual seasonality
  time = np.arange(len(index), dtype=np.float32)
  print(f'time: {time}')
  k = 2 * np.pi * (1 / freq) * time 
  print(f'k: {k}')

  features = {}

  for i in range(1, order + 1):
    features.update(
        {f"sine_{freq}_{i}": np.sine(i * k),
        f"cos{freq}_{i}": np.cos(i * k),
          }

  return pd.DataFrame(features, index=index)


# set plotting parameters
plt.style.use('seaborn-whitegrid')
plt.rc(
  'figure',
  autolayout=True,
  figsize=(11,5),
  )
plt.rc(
  'axes',
  labelweight='bold',
  labelsize='large',
  titleweight='bold',
  titlesize=16,
  titlepad=10,
  )
plot_params = dict(
    color="0.75",
    style=".-",
    markeredgecolor="0.25",
    markerfacecolor="0.25",
    legend=False,
)

def seasonal_plot(X, y, period, freq, ax=None):
  if ax is None:
    _, ax = plt.subplots()
  
  palette = sns.color_palette('hus1', n_colors=X[period].nunique())

  ax = sns.linplot(
    x = freq,
    y = y,
    hue = period,
    data = X,
    ci = False,
    ax = ax,
    palette = palette,
    legend = False
    )

  ax.set_title(f'seasonal plot ({period} / {freq})')

  for line, name in zip(ax.line, X[period].unuqie()):
      y_ = lin.get_ydata()[-1]
      ax.annotate(
          name,
          xy=(1, y_),
          xytext = (6,0)
          color = line.get_color(),
          xycoords = ax.get_yaxis_transform(),
          textcoords = 'offset points',
          size = 14,
          va = 'center'
          )

  return ax


def plot_periodogram(ts, detrend='linear', ax=None):
  from scipy.signal import periodogram
  fs = pd.Timedelta('1Y') / pd.Timedelta('1D')
  frequencies, spectrum = periodogram(
      ts,
      fs=fs,
      detrend = dtrend,
      window='boxcar',
      scaling='spectrum'
      _
  
  if ax is None:
    _, ax = plt.plot()


  ax.step(freqencies, spectrum, color="purple")
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 4, 6, 12, 26, 52, 104])
    ax.set_xticklabels(
        [
            "Annual (1)",
            "Semiannual (2)",
            "Quarterly (4)",
            "Bimonthly (6)",
            "Monthly (12)",
            "Biweekly (26)",
            "Weekly (52)",
            "Semiweekly (104)",
        ],
        rotation=30,
    )
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.set_ylabel("Variance")
    ax.set_title("Periodogram")
    return ax
