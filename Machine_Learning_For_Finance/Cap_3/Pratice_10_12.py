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
import sys
sys.path.append("/Users/jorgea.moreira/Projetos/Python/TropaPyLibs")
#sys.path.append("/Users/jorgeaugusto01/Projetos/Python/TropaPyLibs")
#sys.path.append("/Users/jorgeaugusto01/Projetos/Python/TropaPyLibs")

from dataDeal import csvUtil as dd
from datetime import date

start_ = date(2017, 01, 01)
#end_ = start_ + datetime.timedelta(days=365)
end_ = date(2018, 8, 17)
periodo = pd.date_range(start_, end_)

#df1 = pd.read_csv('../../../Data/csv/Ibovespa/dailyPrices/IBOVESPA.csv')
df1 = dd.get_data_frame_daily_OHLC_prices_from_csv("IBOVESPA", periodo)
df2 = dd.get_data_frame_daily_OHLC_prices_from_csv("CMIG3", periodo)

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


print("##############################CAP_3##############################")
print("----------Exerc1_CAP3--------")
#Standardizing data
#We need to scale our data for some models. K-nearest neighbors (KNN) and neural networks are models that usually work better with scaled data.
#We also need to remove the variables we found were unimportant from last chapter's feature importances.
# We'll simply index the features DataFrames to remove the day of week features with .iloc[].
#KNN uses distances to find similar data points for predictions. If a feature is large, it outweighs small features.
# Scaling data fixes that. Neural networks also work better with scaled data, which we'll cover soon.
#sklearn's scale() will standardize data, which sets the mean to 0 and standard deviation to 1.
#Once we've scaled the data, we'll check that it worked by plotting histograms of the data.

from sklearn.preprocessing import scale

# Remove unimportant features (weekdays)
train_features = train_features.iloc[:, :-4]
test_features = test_features.iloc[:, :-4]

# Standardize the train and test features
scaled_train_features = scale(train_features)
scaled_test_features = scale(test_features)

# Plot histograms of the 14-day SMA RSI before and after scaling
f, ax = plt.subplots(nrows=2, ncols=1)
train_features.iloc[:, 2].hist(ax=ax[0])
ax[1].hist(scaled_train_features[:, 2])
plt.show()

#Optimize n_neighbors
#Now that we have scaled data, we can try using a KNN model. To maximize performance, we should tune our model's hyperparameters.
# For the k-nearest neighbors algorithm, we only have one hyperparameter: n, the number of neighbors. We set this hyperparameter
# when we create the model with KNeighborsRegressor. The argument for the number of neighbors is n_neighbors.
#We want to try a range of values that passes through the setting with the best performance. Usually we will start with 2 neighbors, and
# increase until our scoring metric starts to decrease. We'll use the R2 value from the .score() method on the test set
# (scaled_test_features and test_targets) to optimize n here. We'll use the test set scores to determine the best n.

print("----------Exerc2_CAP3--------")
from sklearn.neighbors import KNeighborsRegressor

for n in range(2,12):
    # Create and fit the KNN model
    knn = KNeighborsRegressor(n_neighbors=n)

    # Fit the model to the training data
    knn.fit(scaled_train_features, train_targets)

    # Print number of neighbors and the score to find the best value of n
    print("n_neighbors =", n)
    print('train, test scores')
    print(knn.score(scaled_train_features, train_targets))
    print(knn.score(scaled_test_features, test_targets))
    print()  # prints a blank line


#Evaluate KNN performance
#We just saw a few things with our KNN scores. For one, the training scores started high and decreased with increasing n,
# which is typical. The test set performance reached a peak at 5 though, and we will use that as our setting in the final KNN model.
#As we have done a few times now, we will check our performance visually. This helps us see how well the model is predicting on
# different regions of actual values. We will get predictions from our knn model using the .predict() method on our scaled features.
# Then we'll use matplotlib's plt.scatter() to create a scatter plot of actual versus predicted values.
# Create the model with the best-performing n_neighbors of 5

print("----------Exerc3_CAP3--------")
knn = KNeighborsRegressor(n_neighbors=5)

# Fit the model
knn.fit(scaled_train_features, train_targets)

# Get predictions for train and test sets
train_predictions = knn.predict(scaled_train_features)
test_predictions = knn.predict(scaled_test_features)

# Plot the actual vs predicted values
plt.scatter(train_predictions, train_targets, label='train')
plt.scatter(test_predictions, test_targets)
plt.legend()
plt.show()

print("----------Exerc4_CAP3--------")
#Build and fit a simple neural net
#The next model we will learn how to use is a neural network.
# Neural nets can capture complex interactions between variables, but are difficult to set up
# and understand. Recently, they have been beating human experts in many fields, including image
# recognition and gaming (check out AlphaGo) -- so they have great potential to perform well.
# To build our nets we'll use the keras library. This is a high-level API that allows us to quickly make neural nets,
# yet still exercise a lot of control over the design. The first thing we'll do is create almost the
# simplest net possible -- a 3-layer net that takes our inputs and predicts a single value. Much like the sklearn
# models, keras models have a .fit() method that takes arguments of (features, targets).

from keras.models import Sequential
from keras.layers import Dense

# Create the model
model_1 = Sequential()
model_1.add(Dense(100, input_dim=scaled_train_features.shape[1], activation='relu'))
model_1.add(Dense(20, activation='relu'))
model_1.add(Dense(1, activation='linear'))

# Fit the model
model_1.compile(optimizer='adam', loss='mse')
history = model_1.fit(scaled_train_features, train_targets, epochs=25)

print("----------Exerc5_CAP3--------")
#Plot losses
#Once we've fit a model, we usually check the training loss
# curve to make sure it's flattened out. The history returned from model.fit()
# is a dictionary that has an entry, 'loss', which is the training loss.
# We want to ensure this has more or less flattened out at the end of our training.
# Plot the losses from the fit
plt.plot(history.history['loss'])

# Use the last loss as the title
plt.title('loss:' + str(round(history.history['loss'][-1], 6)))
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

print("----------Exerc6_CAP3--------")
#Measure performance
#Now that we've fit our neural net, let's check performance to see how well our model is
# predicting new values. There's not a built-in .score() method like with sklearn models, so
# we'll use the r2_score() function from sklearn.metrics. This calculates the R2 score given arguments
# (y_true, y_predicted). We'll also plot our predictions versus actual values again. This will yield some
# interesting results soon (once we implement our own custom loss function).
from sklearn.metrics import r2_score

# Calculate R^2 score
train_preds = model_1.predict(scaled_train_features)
test_preds = model_1.predict(scaled_test_features)
print(r2_score(train_targets, train_preds))
print(r2_score(test_targets, test_preds))

# Plot predictions vs actual
plt.scatter(train_preds, train_targets, label='train')
plt.scatter(test_preds, test_targets, label='test')
plt.legend()
plt.show()

print("----------Exerc7_CAP3--------")
#Custom loss function
#Up to now, we've used the mean squared error as a loss function. This works fine, but with stock price
# prediction it can be useful to implement a custom loss function. A custom loss function can help
# improve our model's performance in specific ways we choose. For example, we're going to create a custom loss
# function with a large penalty for predicting price movements in the wrong direction.
# This will help our net learn to at least predict price movements in the correct direction.
#To do this, we need to write a function that takes arguments of (y_true, y_predicted).
# We'll also use functionality from the backend keras (using tensorflow) to find cases
# where the true value and prediction don't match signs, then penalize those cases.
import keras.losses
import tensorflow as tf

# Create loss function
def sign_penalty(y_true, y_pred):
    penalty = 100.
    loss = tf.where(tf.less(y_true * y_pred, 0), \
                     penalty * tf.square(y_true - y_pred), \
                     tf.square(y_true - y_pred))

    return tf.reduce_mean(loss, axis=-1)

keras.losses.sign_penalty = sign_penalty  # enable use of loss with keras
print(keras.losses.sign_penalty)

print("----------Exerc8_CAP3--------")
#Visualize the results
#We've fit our model with the custom loss function, and it's time to see how it is performing.
# We'll check the R2 values again with sklearn's r2_score() function, and we'll create a scatter
# plot of predictions versus actual values with plt.scatter(). This will yield some interesting results!
# Evaluate R^2 scores
train_preds = model_2.predict(scaled_train_features)
test_preds = model_2.predict(scaled_test_features)
print(r2_score(train_targets, train_preds))
print(r2_score(test_targets, test_preds))

# Scatter the predictions vs actual -- this one is interesting!
plt.scatter(train_preds, train_targets, label='train')
plt.scatter(test_preds, test_targets, label='test')  # plot test set
plt.legend(); plt.show()

print("----------Exerc10_CAP3--------")
#Combatting overfitting with dropout
#A common problem with neural networks is they tend to overfit to training data.
# What this means is the scoring metric, like R2 or accuracy, is high for the training set, but low for testing and
# validation sets, and the model is fitting to noise in the training data.
#We can work towards preventing overfitting by using dropout. This randomly drops some neurons during the
# training phase, which helps prevent the net from fitting noise in the training data. keras has a Dropout
# layer that we can use to accomplish this. We need to set the dropout rate, or fraction of connections dropped
# during training time. This is set as a decimal between 0 and 1 in the Dropout() layer.
#We're going to go back to the mean squared error loss function for this model.
from keras.layers import Dropout

# Create model with dropout
model_3 = Sequential()
model_3.add(Dense(100, input_dim=scaled_train_features.shape[1], activation='relu'))
model_3.add(Dropout(0.2))
model_3.add(Dense(20, activation='relu'))
model_3.add(Dense(1, activation='linear'))

# Fit model with mean squared error loss function
model_3.compile(optimizer="adam", loss="mse")
history = model_3.fit(scaled_train_features, train_targets, epochs=25)
plt.plot(history.history['loss'])
plt.title('loss:' + str(round(history.history['loss'][-1], 6)))
plt.show()

print("----------Exerc11_CAP3--------")
#Ensembling models
#One approach to improve predictions from machine learning models is ensembling.
# A basic approach is to average the predictions from multiple models. A more complex approach is to feed
# predictions of models into another model, which makes final predictions. Both approaches usually
# improve our overall performance (as long as our individual models are good).
# If you remember, random forests are also using ensembling of many decision trees.
#To ensemble our neural net predictions, we'll make predictions with the 3 models we just created -- the basic model,
# the model with the custom loss function, and the model with dropout. Then we'll combine the predictions with numpy's .hstack()
# function, and average them across rows with np.mean(predictions, axis=1).

# Make predictions from the 3 neural net models
train_pred1 = model_1.predict(scaled_train_features)
test_pred1 = model_1.predict(scaled_test_features)

train_pred2 = model_2.predict(scaled_train_features)
test_pred2 = model_2.predict(scaled_test_features)

train_pred3 = model_3.predict(scaled_train_features)
test_pred3 = model_3.predict(scaled_test_features)

# Horizontally stack predictions and take the average across rows
train_preds = np.mean(np.hstack((train_pred1, train_pred2, train_pred3)), axis=1)
test_preds = np.mean(np.hstack((test_pred1, test_pred2, test_pred3)), axis=1)
print(test_preds[-5:])


print("----------Exerc12_CAP3--------")
#See how the ensemble performed
#Let's check performance of our ensembled model to see how it's doing. We should see roughly an
# average of the R2 scores, as well as a scatter plot that is a mix of our previous models' predictions.
# The bow-tie shape from the custom loss function model should still be a bit visible, but the edges near x=0 should be softer.
from sklearn.metrics import r2_score

# Evaluate the R^2 scores
print(r2_score(train_targets, train_preds))
print(r2_score(test_targets, test_preds))

# Scatter the predictions vs actual -- this one is interesting!
plt.scatter(train_preds, train_targets, label='train')
plt.scatter(test_preds, test_targets, label='test')
plt.legend(); plt.show()




