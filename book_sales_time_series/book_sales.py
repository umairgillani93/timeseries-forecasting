import os
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('./book_sales.csv',
                index_col='Date',
                parse_dates=['Date'],
                ).drop('Paperback', axis=1)

# Exploratory data analysis
# Check head of dataframe 
# df.plot()
# plt.show()
print(df.head())

print('df index: {}'.format(df.index))

# Add time columns, starting from index 0 to len(df.index)
df['Time'] = np.arange(len(df.index))

print('\ndf: {}'.format(df.head()))

# plot the time-series
plt.style.use('seaborn-whitegrid')
plt.rc(
    "figure",
    autolayout=True,
    figsize=(11,4),
    titlesize=18,
    titleweight="bold"
    )



fig, ax = plt.subplots()
ax.plot('Time', 'Hardcover', data=df, color='0.75')

ax = sns.regplot(x='Time', y='Hardcover', data=df, ci=None)

ax.set_title('Time Plot of Hardcover Sales')
#plt.show()


# Adding lagging feature
df['Lag_1'] = df['Hardcover'].shift(1)

# Reindex the columns
df = df.reindex(columns=['Hardcover', 'Lag_1'])

print('\nDf: {}'.format(df.head()))


# Plotting the lagged series
fig, ax = plt.subplots()
ax = sns.regplot(x='Lag_1', y='Hardcover', data=df, ci=None)
ax.set_aspect('equal')
ax.set_title('Lag Plot of Hardcover Sales')

# Find the co-related between 'Hardcover' and 'Lag_1' 
# Sales with previous day
print(f'co-relation: {df.corr()}')


# IMPORTANT: In time series we have a concept called Serial dependance
# which means dependance of successding time-stamp value on preceeding one
# This shows us higher sales in one-day means higher sales on next-day

df['Time'] = np.arange(len(df.index))
# Lets build a liner regression model on our data
X = df.loc[:, ['Time']] # features
y = df.loc[:, ['Hardcover']] # target

print(f'\nX: {X}')
print(f'\ny: {y}')


# Train model
model = LinearRegression()
print('\nModel loaded..')

# Fit the model
model.fit(X, y)

print('>> Model fitted')
# Save the predictions
y_preds = model.predict(X)

print(f'Y preds: {y_preds}')

plot_params = dict(
    color="0.75",
    style=".-",
    markeredgecolor="0.25",
    markerfacecolor="0.25",
    legend=False,
)

ser = pd.Series([c[0] for c in y_preds], index=X.index)

ax = y.plot(**plot_params)
ax = ser.plot()
plt.show()
