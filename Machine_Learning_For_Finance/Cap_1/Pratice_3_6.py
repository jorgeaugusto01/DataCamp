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
import talib
import seaborn as sns


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

feature_names = ['5d_close_pct']  # a list of the feature names for later

# Create moving averages and rsi for timeperiods of 14, 30, 50, and 200
for n in [14,30,50,200]:

    # Create the moving average indicator and divide by Adj_Close
    df1['ma' + str(n)] = talib.SMA(df1['Close'].values, timeperiod=n) / df1['Close']
    # Create the RSI indicator
    df1['rsi' + str(n)] = talib.RSI(df1['Close'].values, timeperiod=n)

    # Add rsi and moving average to the feature name list
    feature_names = feature_names + ['ma' + str(n), 'rsi' + str(n)]

print(feature_names)

#Create features and targets
#We almost have features and targets that are machine-learning ready -- we have features from
# current price changes (5d_close_pct) and indicators (moving averages and RSI), and we created targets of
# future price changes (5d_close_future_pct). Now we need to break these up into
# separate numpy arrays so we can feed them into machine learning algorithms.
#Our indicators also cause us to have missing values at the beginning of the DataFrame due to the calculations.
# We could backfill this data, fill it with a single value, or drop the rows. Dropping the rows is a good choice,
# so our machine learning algorithms aren't confused by any sort of backfilled or 0-filled data. Pandas has a .dropna()
# function which we will use to drop any rows with missing values.

# Drop all na values
df1 = df1.dropna()

# Create features and targets -- use the variable feature_names
features = df1[feature_names]
targets = df1['5d_close_future_pct']

# Create DataFrame from target column and feature columns
feat_targ_df = df1[['5d_close_future_pct'] + feature_names]

# Calculate correlation matrix
corr = feat_targ_df.corr()
print(corr)

#Check the correlations
#Before we fit our first machine learning model, let's look at the correlations between features and targets.
# Ideally we want large (near 1 or -1) correlations between features and targets. Examining correlations can help us
# tweak features to maximize correlation (for example, altering the timeperiod argument in the talib functions).
# It can also help us remove features that aren't correlated to the target.
#To easily plot a correlation matrix, we can use seaborn's heatmap() function. This takes a correlation matrix as the
# first argument, and has many other options. Check out the annot option -- this will help us turn on annotations.
# Plot heatmap of correlation matrix
sns.heatmap(corr, annot=True)
plt.yticks(rotation=0); plt.xticks(rotation=90)  # fix ticklabel directions
plt.tight_layout()  # fits plot area to the plot, "tightly"
plt.show()  # show the plot
plt.clf()  # clear the plot area

# Create a scatter plot of the most highly correlated variable with the target
plt.scatter(df1['ma200'], df1['5d_close_future_pct'])
plt.show()
