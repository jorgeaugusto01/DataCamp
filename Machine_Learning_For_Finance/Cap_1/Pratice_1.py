from scipy.cluster.hierarchy import linkage, dendrogram
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


lng_df = pd.read_csv('../../../Data/csv/eua/AAPL.csv')
spy_df = pd.read_csv('../../../Data/csv/eua/AMD.csv')

print(lng_df.head())  # examine the DataFrames
print(spy_df.head())  # examine the SPY DataFrame

# Plot the Adj_Close columns for SPY and LNG
spy_df["Adj_Close"].plot(label='SPY', legend=True)
lng_df["Adj_Close"].plot(label='LNG', legend=True, secondary_y=True)
plt.show()  # show the plot
plt.clf()  # clear the plot space

# Histogram of the daily price change percent of Adj_Close for LNG
lng_df['Adj_Close'].pct_change().plot.hist(bins=50)
plt.xlabel('adjusted close 1-day percent change')
plt.show()
