#Explore the data with some EDA
#First, let's explore the data. Any time we begin a machine learning (ML) project, we need to first do some exploratory data analysis (EDA) to familiarize ourselves with the data. This includes things like:
    #raw data plots
    #histograms
    #and more...
#I typically begin with raw data plots and histograms. This allows us to understand our data's distributions.
# If it's a normal distribution, we can use things like parametric statistics.
#There are two stocks loaded for you into pandas DataFrames: lng_df and spy_df (LNG and SPY). Take a look at them with .head().
# We'll use the closing prices and eventually volume as inputs to ML algorithms.
#Note: We'll call plt.clf() each time we want to make a new plot, or f = plt.figure().

from scipy.cluster.hierarchy import linkage, dendrogram
import pandas as pd
import matplotlib.pyplot as plt


df1 = pd.read_csv('../../../Data/csv/Ibovespa/dailyPrices/IBOVESPA.csv')
df2 = pd.read_csv('../../../Data/csv/Ibovespa/dailyPrices/CMIG3.csv')

print(df1.head())  # examine the DataFrames
print(df2.head())  # examine the SPY DataFrame

# Plot the Adj_Close columns for SPY and LNG
df1["Close"].plot(label='IBOVESPA', legend=True)
df2["Close"].plot(label='CMIG3', legend=True, secondary_y=True)
plt.show()  # show the plot
plt.clf()  # clear the plot space

# Histogram of the daily price change percent of Adj_Close for CMIG3
df2['Close'].pct_change().plot.hist(bins=50)
plt.xlabel('adjusted close 1-day percent change')
plt.show()

#Correlations
#Correlations are nice to check out before building machine learning models, because
# we can see which features correlate to the target most strongly. Pearson's correlation coefficient is often used,
# which only detects linear relationships. It's commonly assumed our data is normally distributed,
# which we can "eyeball" from histograms. Highly correlated variables have a Pearson correlation coefficient near 1
# (positively correlated) or -1 (negatively correlated). A value near 0 means the two variables are not linearly correlated.
#If we use the same time periods for previous price changes and future price changes, we can see if the stock price
# is mean-reverting (bounces around) or trend-following (goes up if it has been going up recently).

# Create 5-day % changes of Adj_Close for the current day, and 5 days in the future
df1['5d_future_close'] = df1['Close'].shift(-5)
df1['5d_close_future_pct'] = df1['5d_future_close'].pct_change(5)
df1['5d_close_pct'] = df1['Close'].pct_change(5)

# Calculate the correlation matrix between the 5d close pecentage changes (current and future)
corr = df1[['5d_close_pct', '5d_close_future_pct']].corr()
print(corr)

# Scatter the current 5-day percent change vs the future 5-day percent change
plt.scatter(df1['5d_close_pct'], df1['5d_close_future_pct'])
plt.show()
