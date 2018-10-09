#Join stock DataFrames and calculate returns
#Our first step towards calculating modern portfolio theory (MPT) portfolios
# is to get daily and monthly returns. Eventually we're going to get the best portfolios
# of each month based on the Sharpe ratio. The easiest way to do this is to put all our stock
# prices into one DataFrame, then to resample them to the daily and monthly time frames.
# We need daily price changes to calculate volatility, which we will use as our measure of risk.

# Join 3 stock dataframes together
full_df = pd.concat([lng_df, spy_df, smlv_df], axis=1).dropna()

# Resample the full dataframe to monthly timeframe
monthly_df = full_df.resample('BMS').first()

# Calculate daily returns of stocks
returns_daily = full_df.pct_change()

# Calculate monthly returns of the stocks
returns_monthly = monthly_df.pct_change().dropna()
print(returns_monthly.tail())

#Calculate covariances for volatility
#In MPT, we quantify risk via volatility. The math for calculating portfolio
# volatility is complex, and it requires daily returns covariances.
# We'll now loop through each month in the returns_monthly DataFrame, and
# calculate the covariance of the daily returns.
#With pandas datetime indices, we can access the month and year with df.index.month
# and df.index.year. We'll use this to create a mask for returns_daily that gives us the daily
# returns for the current month and year in the loop. We then use the mask to subset the DataFrame like this: df[mask].
# This gets entries in the returns_daily DataFrame which are in the current month and year in each cycle of the loop.
# Finally, we'll use pandas' .cov() method to get the covariance of daily returns.
# Daily covariance of stocks (for each monthly period)

covariances = {}
for i in returns_monthly.____:
    rtd_idx = returns_daily.index

    # Mask daily returns for each month and year, and calculate covariance
    mask = (rtd_idx.month == i.month) & (rtd_idx.____ == i.____)

    # Use the mask to get daily returns for the current month and year of monthy returns index
    covariances[i] = returns_daily[____].cov()

print(covariances[i])


#Calculate portfolios
#We'll now generate portfolios to find each month's best one. numpy's random.random()
# generates random numbers from a uniform distribution, then we normalize them so they sum to 1 using the /= operator.
# We use these weights to calculate returns and volatility. Returns are sums of weights times individual returns.
# Volatility is more complex, and involves the covariances of the different stocks.
# Finally we'll store the values in dictionaries for later use, with months' dates as keys.
# In this case, we will only generate 10 portfolios for each date so the code will run faster,
# but in a real-world use-case you'd want to use more like 1000 to 5000 randomly-generated portfolios for a few stocks.

portfolio_returns, portfolio_volatility, portfolio_weights = {}, {}, {}

# Get portfolio performances at each month
for date in sorted(covariances.keys()):
    cov = covariances[date]
    for portfolio in range(10):
        weights = np.random.random(3)
        weights /= np.sum(weights)  # /= divides weights by their sum to normalize
        returns = np.dot(weights, returns_monthly.loc[date])
        volatility = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
        portfolio_returns.setdefault(date, []).append(returns)
        portfolio_volatility.setdefault(date, []).append(volatility)
        portfolio_weights.setdefault(date, []).append(weights)

print(portfolio_weights[date][0])

#Plot efficient frontier
#We can finally plot the results of our MPT portfolios, which shows the "efficient frontier".
# This is a plot of the volatility vs the returns. This can help us visualize our risk-return possibilities
# for portfolios. The upper left boundary of the points is the best we can do (highest return for a given risk), and that is the efficient frontier.
#To create this plot, we will use the latest date in our covariances dictionary which we created
# a few exercises ago. This has dates as keys, so we'll get the sorted keys using sorted() and .keys(),
# then get the last entry with Python indexing ([-1]). Lastly we'll use matplotlib to scatter variance vs
# returns and see the efficient frontier for the latest date in the data.

# Get latest date of available data
date = sorted(covariances.keys())[-1]

# Plot efficient frontier
# warning: this can take at least 10s for the plot to execute...
plt.scatter(x=portfolio_volatility[date], y=portfolio_returns[date],  alpha=0.1)
plt.xlabel('Volatility')
plt.ylabel('Returns')
plt.show()