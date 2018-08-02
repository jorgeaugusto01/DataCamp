#Train/test split for regression
#As you learned in Chapter 1, train and test sets are vital to ensure that your supervised learning model
# is able to generalize well to new data. This was true for classification models, and is equally true for
# linear regression models.
#In this exercise, you will split the Gapminder dataset into training and testing sets,
# and then fit and predict a linear regression over all features.
# In addition to computing the R2 score, you will also compute the Root Mean Squared Error (RMSE),
#  which is another commonly used metric to evaluate regression models.
# The feature array X and target variable array y have been pre-loaded for you from the DataFrame df.

# Import necessary modules
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge

import pandas as pd
import numpy as np

# Read the CSV file into a DataFrame: df
df = pd.read_csv('../../DataSets/gapminder/gapminder.csv')

# Create arrays for features and target variable
y = df['life'].values
X = df['fertility'].values

# Reshape X and y
y = y.reshape(-1, 1)
X = X.reshape(-1, 1)

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)

# Create the regressor: reg_all
reg_all = LinearRegression()

# Fit the regressor to the training data
reg_all.fit(X_train, y_train)

# Predict on the test data: y_pred
y_pred = reg_all.predict(X_test)

# Compute and print R^2 and RMSE
print("R^2: {}".format(reg_all.score(X_test, y_test)))
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error: {}".format(rmse))

#5-fold cross-validation
#Cross-validation is a vital step in evaluating a model. It maximizes the amount of data that is used to
# train the model, as during the course of training, the model is not only trained, but also tested on all of
# the available data.
# In this exercise, you will practice 5-fold cross validation on the Gapminder data.
# By default, scikit-learn's cross_val_score() function uses R2 as the metric of choice for regression.
# Since you are performing 5-fold cross-validation, the function will return 5 scores. Your job is to compute
# these 5 scores and then take their average.
# The DataFrame has been loaded as df and split into the feature/target variable arrays X and y. The modules pandas and numpy have been imported as pd and np, respectively.
# Import the necessary modules
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

# Create a linear regression object: reg
reg = LinearRegression()

# Compute 5-fold cross-validation scores: cv_scores
cv_scores = cross_val_score(reg, X, y, cv=5)

# Print the 5-fold cross-validation scores
print(cv_scores)

print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores)))

#Regression with categorical features
# Having created the dummy variables from the 'Region' feature, you can build
# regression models as you did before. Here, you'll use ridge regression to perform 5-fold cross-validation.
# The feature array X and target variable array y have been pre-loaded.

# Instantiate a ridge regressor: ridge
ridge = Ridge(normalize=True, alpha=0.5)

# Perform 5-fold cross-validation: ridge_cv
ridge_cv = cross_val_score(ridge, X, y, cv=5)

# Print the cross-validated scores
print(ridge_cv)
