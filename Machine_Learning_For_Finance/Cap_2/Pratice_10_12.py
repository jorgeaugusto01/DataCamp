#Feature engineering from volume
#We're going to use non-linear models to make more accurate predictions.
# With linear models, features must be linearly correlated to the target.
# Other machine learning models can combine features in non-linear ways. For example, what if the price goes up when the moving
# average of price is going up, and the moving average of volume is going down? The only way to capture those interactions
# is to either multiply the features, or to use a machine learning algorithm that can handle non-linearity (e.g. random forests).

#To incorporate more information that may interact with other features, we can add in weakly-correlated features.
# First we will add volume data, which we have in the lng_df as the

import pandas as pd
import matplotlib.pyplot as plt
import talib
import seaborn as sns
import numpy as np
from sklearn.tree import DecisionTreeRegressor


df1 = pd.read_csv('../../../Data/csv/Ibovespa/dailyPrices/IBOVESPA.csv')
df2 = pd.read_csv('../../../Data/csv/Ibovespa/dailyPrices/CMIG3.csv')

df1['Volume'] = df1['Volume'].str.replace(',','.')
df2['Volume'] = df2['Volume'].str.replace(',','.')
df1['Volume']= map(float, df1['Volume'])
df2['Volume']= map(float, df2['Volume'])

df1 = df1.set_index(df1['Date'])
df1 = df1.drop(columns=['Date'])
df1.index = pd.to_datetime(df1.index)

df2 = df2.set_index(df2['Date'])
df2 = df2.drop(columns=['Date'])
df2.index = pd.to_datetime(df2.index)

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

#Create train and test features
#Before we fit our linear model, we want to add a constant to our features, so we have an intercept for our linear model.
#We also want to create train and test features. This is so we can fit our model to the train dataset, and
# evaluate performance on the test dataset. We always want to check performance on data the model has not seen to make sure we're not
# overfitting, which is memorizing patterns in the training data too exactly.
#With a time series like this, we typically want to use the oldest data as our training set, and the newest data as our test set.
# This is so we can evaluate the performance of the model on the most recent data, which will more realistically
# simulate predictions on data we haven't seen yet.

# Import the statsmodels.api library with the alias sm
import statsmodels.api as sm

# Add a constant to the features
linear_features = sm.add_constant(features)

# Create a size for the training set that is 85% of the total number of samples
train_size = int(0.85 * targets.shape[0])
train_features = linear_features[:train_size]
train_targets = targets[:train_size]
test_features = linear_features[train_size:]
test_targets = targets[train_size:]
print(linear_features.shape, train_features.shape, test_features.shape)

#Fit a linear model
#We'll now fit a linear model, because they are simple and easy to understand. Once we've fit our model,
# we can see which predictor variables appear to be meaningfully linearly correlated with the target,
# as well as their magnitude of effect on the target. Our judgment of whether or not predictors are significant
# is based on the p-values of coefficients. This is using a t-test to statistically test if the coefficient
# significantly differs from 0. The p-value is the percent chance that the coefficient for a feature does not
# differ from zero. Typically, we take a p-value of less than 0.05 to mean the coefficient is significantly different from 0.

# Create the linear model and complete the least squares fit
model = sm.OLS(train_targets, train_features)
results = model.fit()
print(results.summary())

# Features with p <= 0.05 are typically considered significantly different from 0
print(results.pvalues)

# Make predictions from our model for train and test sets
train_predictions = results.predict(train_features)
test_predictions = results.predict(test_features)

#Evaluate our results
#Once we have our linear fit and predictions, we want to see how good the predictions are so we can decide if
# our model is any good or not. Ideally, we want to back-test any type of trading strategy. However,
# this is a complex and typically time-consuming experience.
#A quicker way to understand the performance of our model is looking at regression evaluation metrics like R2,
# and plotting the predictions versus the actual values of the targets. Perfect predictions would form a straight,
# diagonal line in such a plot, making it easy for us to eyeball how our predictions are doing in different regions
# of price changes. We can use matplotlib's .scatter() function to create scatter plots of the predictions and actual values.
# Scatter the predictions vs the targets with 80% transparency
plt.scatter(train_predictions, train_targets, alpha=0.2, color='b', label='train')
plt.scatter(test_predictions, test_targets, alpha=0.2, color='r', label='test')

# Plot the perfect prediction line
xmin, xmax = plt.xlim()
plt.plot(np.arange(xmin, xmax, 0.01), np.arange(xmin, xmax, 0.01), c='k')

# Set the axis labels and show the plot
plt.xlabel('predictions')
plt.ylabel('actual')
plt.legend()  # show the legend
plt.show()

#CAP_2
#Feature engineering from volume
#We're going to use non-linear models to make more accurate predictions.
# With linear models, features must be linearly correlated to the target. Other machine learning models can
# combine features in non-linear ways. For example, what if the price goes up when the moving average of price is going up,
# and the moving average of volume is going down? The only way to capture those interactions is to either multiply the features,
# or to use a machine learning algorithm that can handle non-linearity (e.g. random forests).
#To incorporate more information that may interact with other features, we can add in weakly-correlated features.
# First we will add volume data, which we have in the lng_df as the Adj_Volume column.

# Create 2 new volume features, 1-day % change and 5-day SMA of the % change
new_features = ['Adj_Volume_1d_change', 'Adj_Volume_1d_change_SMA']
feature_names.extend(new_features)
df1['Adj_Volume_1d_change'] = df1['Volume'].astype(float).pct_change(1)
df1['Adj_Volume_1d_change_SMA'] = talib.SMA(np.array(df1['Adj_Volume_1d_change'], dtype=float), timeperiod=5)

# Plot histogram of volume % change data
df1[new_features].plot(kind='hist', sharex=False, bins=200)
plt.show()

#Create day-of-week features
#We can engineer datetime features to add even more information for our non-linear models.
# Most financial data has datetimes, which have lots of information in them -- year, month, day, and sometimes hour,
# minute, and second. But we can also get the day of the week, and things like the quarter of the year,
# or the elapsed time since some event (e.g. earnings reports).
#We are only going to get the day of the week here, since our dataset doesn't go back very far in time.
# The dayofweek property from the pandas datetime index will help us get the day of the week.
# Then we will dummy dayofweek with pandas' get_dummies(). This creates columns for each day of the week with binary values (0 or 1).
# We drop the first column because it can be inferred from the others.

# Use pandas' get_dummies function to get dummies for day of the week
days_of_week = pd.get_dummies(df1.index.dayofweek,
                              prefix='weekday',
                              drop_first=True)

# Set the index as the original dataframe index for merging
days_of_week.index = df1.index

# Join the dataframe with the days of week dataframe
df1 = pd.concat([df1, days_of_week], axis=1)

# Add days of week to feature names
feature_names.extend(['weekday_' + str(i) for i in range(1, 5)])
df1.dropna(inplace=True)  # drop missing values in-place
print(df1.head())


#Examine correlations of the new features
#Now that we have our volume and datetime features, we want to check the correlations
# between our new features (stored in the new_features list) and the target
# (5d_close_future_pct) to see how strongly they are related. Recall pandas has the built-in .corr()
# method for DataFrames, and seaborn has a nice heatmap() function to show the correlations.

# Add the weekday labels to the new_features list
# Add the weekday labels to the new_features list
new_features.extend(['weekday_' + str(i) for i in range(1, 5)])

print(new_features)

# Plot the correlations between the new features and the targets
sns.heatmap(df1[['5d_close_future_pct'] + new_features].corr(), annot=True)
plt.yticks(rotation=0)  # ensure y-axis ticklabels are horizontal
plt.xticks(rotation=90)  # ensure x-axis ticklabels are vertical
plt.tight_layout()
plt.show()

#Fit a decision tree
#Random forests are a go-to model for predictions; they work well out of the box.
# But we'll first learn the building block of random forests -- decision trees.
#Decision trees split the data into groups based on the features.
# Decision trees start with a root node, and split the data down until we reach leaf nodes.

# Create a decision tree regression model with default arguments
decision_tree = DecisionTreeRegressor()

# Fit the model to the training features and targets
decision_tree.fit(train_features, train_targets)

# Check the score on train and test
print(decision_tree.score(train_features, train_targets))
print(decision_tree.score(test_features, test_targets))

#Try different max depths
#We always want to optimize our machine learning models to make the best predictions possible.
# We can do this by tuning hyperparameters, which are settings for our models. We will see in more detail how these are useful
# in future chapters, but for now think of them as knobs we can turn to tune our predictions to be as good as possible.
#For regular decision trees, probably the most important hyperparameter is max_depth.
# This limits the number of splits in a decision tree. Let's find the best value of max_depth based on the R2 score of
# our model on the test set, which we can obtain using the score() method of our decision tree models.
# Loop through a few different max depths and check the performance
print("-----------------------------------")
for d in [3,5,10]:
    # Create the tree and fit it
    decision_tree = DecisionTreeRegressor(max_depth=d)
    decision_tree.fit(train_features, train_targets)

    # Print out the scores on train and test
    print('max_depth=', str(d))
    print(decision_tree.score(train_features, train_targets))
    # \n prints a blank line
    print(decision_tree.score(test_features, test_targets), '\n')


#Check our results
#Once we have an optimized model, we want to check how it is performing in more detail.
# We already saw the R2 score, but it can be helpful to see the predictions plotted vs actual values.
# We can use the .predict() method of our decision tree model to get predictions on the train and test sets.
# Ideally, we want to see diagonal lines from the lower left to the upper right. However, due to the simplicity of
# decisions trees, our model is not going to do well on the test set. But it will do well on the train set.

# Use the best max_depth of 3 from last exercise to fit a decision tree
decision_tree = DecisionTreeRegressor(max_depth=3)
decision_tree.fit(train_features, train_targets)

# Predict values for train and test
train_predictions = decision_tree.predict(train_features)
test_predictions = decision_tree.predict(test_features)

# Scatter the predictions vs actual values
plt.scatter(train_predictions, train_targets, label='train')
plt.scatter(test_predictions, test_targets, label='test')
plt.show()

#Fit a random forest
#Data scientists often use random forest models. They perform well out of the box, and have lots of settings to optimize performance.
# Random forests can be used for classification or regression; we'll use it for regression to predict the future price change of LNG.
#we'll create and fit the random forest model similarly to the decision trees using the .fit(features, targets) method.
# With sklearn's RandomForestRegressor, there's a built-in .score() method we can use to evaluate performance.
# This takes arguments (features, targets), and returns the R2 score (the coefficient of determination).

from sklearn.ensemble import RandomForestRegressor

# Create the random forest model and fit to the training data
rfr = RandomForestRegressor(n_estimators=200)
rfr.fit(train_features, train_targets)

# Look at the R^2 scores on train and test
print("-----------------------------------")
print(rfr.score(train_features, train_targets))
print(rfr.score(test_features, test_targets))

#Tune random forest hyperparameters
#As with all models, we want to optimize performance by tuning hyperparameters. We have many hyperparameters
# for random forests, but the most important is often the number of features we sample at each split,
# or max_features in RandomForestRegressor from the sklearn library. For models like random forests that have
# randomness built-in, we also want to set the random_state. This is set for our results to be reproducible.
#Usually, we can use sklearn's GridSearchCV() method to search hyperparameters, but with a financial time series,
# we don't want to do cross-validation due to data mixing. We want to fit our models on the oldest data and evaluate
# on the newest data. So we'll use sklearn's ParameterGrid to create combinations of hyperparameters to search.

from sklearn.model_selection import ParameterGrid

# Create a dictionary of hyperparameters to search
grid = {'n_estimators': [200], 'max_depth': [3], 'max_features': [4, 8], 'random_state': [42]}
test_scores = []

# Loop through the parameter grid, set the hyperparameters, and save the scores
for g in ParameterGrid(grid):
    rfr.set_params(**g)  # ** is "unpacking" the dictionary
    rfr.fit(train_features, train_targets)
    test_scores.append(rfr.score(test_features, test_targets))

# Find best hyperparameters from the test score and print
best_idx = np.argmax(test_scores)
print("-----------------------------------")
print(test_scores[best_idx], ParameterGrid(grid)[best_idx])


#Evaluate performance
#Lastly, and as always, we want to evaluate performance of our best model to check how
# well or poorly we are doing. Ideally it's best to do back-testing, but that's an involved
# process we don't have room to cover in this course.#
# We've already seen the R2 scores, but let's take a look at the scatter plot of predictions vs actual
# results using matplotlib. Perfect predictions would be a diagonal line from the lower left to the upper right.

# Use the best hyperparameters from before to fit a random forest model
rfr = RandomForestRegressor(n_estimators=200, max_depth=3, max_features=4, random_state=42)
rfr.fit(train_features, train_targets)

# Make predictions with our model
train_predictions = rfr.predict(train_features)
test_predictions = rfr.predict(test_features)

# Create a scatter plot with train and test actual vs predictions
plt.scatter(train_targets, train_predictions, label='train')
plt.scatter(test_targets, test_predictions, label='test')
plt.legend()
plt.show()

#Random forest feature importances
#One useful aspect of tree-based methods is the ability to extract feature importances.
# This is a quantitative way to measure how much each feature contributes to our predictions.
# It can help us focus on our best features, possibly enhancing or tuning them, and can also help
# us get rid of useless features that may be cluttering up our model.
#Tree models in sklearn have a .feature_importances_ property that's accessible after fitting the model.
# This stores the feature importance scores. We need to get the indices of the sorted feature importances using np.argsort()
# in order to make a nice-looking bar plot of feature importances (sorted from greatest to least importance).

# Get feature importances from our random forest model
importances = rfr.feature_importances_

# Get the index of importances from greatest importance to least
sorted_index = np.argsort(importances)[::-1]
x = range(len(importances))

# Create tick labels
labels = np.array(feature_names)[sorted_index]
plt.bar(x, importances[sorted_index], tick_label=labels)

# Rotate tick labels to vertical
plt.xticks(rotation=90)
plt.show()


#A gradient boosting model
#Now we'll fit a gradient boosting (GB) model. It's been said a linear model is like a Toyota Camry,
# and GB is like a Black Hawk helicopter. GB has potential to outperform random forests, but doesn't always do so.
# This is called the no free lunch theorem, meaning we should always try lots of different models for each problem.
#GB is similar to random forest models, but the difference is that trees are built successively. With each iteration,
# the next tree fits the residual errors from the previous tree in order to improve the fit.
#For now we won't search our hyperparameters -- they've been searched for you.

from sklearn.ensemble import GradientBoostingRegressor

# Create GB model -- hyperparameters have already been searched for you
gbr = GradientBoostingRegressor(max_features=4,
                                learning_rate=0.01,
                                n_estimators=200,
                                subsample=0.6,
                                random_state=42)
gbr.fit(train_features, train_targets)

print(gbr.score(train_features, train_targets))
print(gbr.score(test_features, test_targets))


#Gradient boosting feature importances
#Just like with random forests, we can extract feature importances from gradient boosting models in
# order to understand which features seem to be the best predictors. Sometimes it's nice to try different tree-based
# models (gradient boosting, random forests, decision trees, adaboost, etc) and look at the feature importances from all of them.
# This can help average out any peculiarities that may arise from one particular model.
# Once again, the feature importances are stored as a numpy array in the .feature_importances_ property of the
# gradient boosting model. We'll need to get the sorted indices of the feature importances, using np.argsort(), in order to make a nice plot.

# Extract feature importances from the fitted gradient boosting model
feature_importances = gbr.feature_importances_

# Get the indices of the largest to smallest feature importances
sorted_index = np.argsort(feature_importances)[::-1]
x = range(len(importances))

# Create tick labels
labels = np.array(feature_names)[sorted_index]

plt.bar(x, feature_importances[sorted_index], tick_label=labels)

# Set the tick lables to be the feature names, according to the sorted feature_idx
plt.xticks(rotation=90)
plt.show()

