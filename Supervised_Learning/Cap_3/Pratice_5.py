#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import necessary modules
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from scipy.stats import randint
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd


seeds = pd.read_csv('../../DataSets/seeds/seeds.csv')
varietisSeeds = pd.read_csv('../../DataSets/seeds/varietiesSeeds.csv')
diabetes = pd.read_csv('../../DataSets/diabets/diabetes.csv')
from sklearn.linear_model import ElasticNet
stockMovements = pd.read_csv('../../DataSets/stocks/StockMovements.csv')
stockMovements = stockMovements.set_index('Unnamed: 0')
stockMovements = stockMovements.T


y = diabetes["Outcome"].values
X = diabetes.drop(columns="Outcome").values


#Hyperparameter tuning with GridSearchCV
#Hugo demonstrated how to tune the n_neighbors parameter of the KNeighborsClassifier() using GridSearchCV
# on the voting dataset. You will now practice this yourself, but by using logistic regression on the diabetes dataset instead!
#Like the alpha parameter of lasso and ridge regularization that you saw earlier, logistic
# regression also has a regularization parameter: C. C controls the inverse of the regularization strength,
# and this is what you will tune in this exercise. A large C can lead to an overfit model, while a small C can lead to an underfit model.
#The hyperparameter space for C has been setup for you. Your job is to use GridSearchCV and logistic regression
# to find the optimal C in this hyperparameter space. The feature array is available as X and target variable array is available as y.
#You may be wondering why you aren't asked to split the data into training and test sets. Good observation!
# Here, we want you to focus on the process of setting up the hyperparameter grid and performing grid-search cross-validation.
# In practice, you will indeed want to hold out a portion of your data for evaluation purposes, and you will learn all about this in the next video!

# Setup the hyperparameter grid
c_space = np.logspace(-5, 8, 15)
param_grid = {'C': c_space}

# Instantiate a logistic regression classifier: logreg
logreg = LogisticRegression()

# Instantiate the GridSearchCV object: logreg_cv
logreg_cv = GridSearchCV(logreg, param_grid, cv=5)

# Fit it to the data
logreg_cv.fit(X, y)

# Print the tuned parameters and score
print("Tuned Logistic Regression Parameters: {}".format(logreg_cv.best_params_))
print("Best score is {}".format(logreg_cv.best_score_))


#Hyperparameter tuning with RandomizedSearchCV
# GridSearchCV can be computationally expensive, especially if you are searching over a large hyperparameter
# space and dealing with multiple hyperparameters. A solution to this is to use RandomizedSearchCV,
# in which not all hyperparameter values are tried out. Instead, a fixed number of hyperparameter settings
# is sampled from specified probability distributions. You'll practice using RandomizedSearchCV in this exercise and see how this works.
# Here, you'll also be introduced to a new model: the Decision Tree. Don't worry about the specifics
# of how this model works. Just like k-NN, linear regression, and logistic regression, decision trees in
# scikit-learn have .fit() and .predict() methods that you can use in exactly the same way as before. Decision
# trees have many parameters that can be tuned, such as max_features, max_depth, and min_samples_leaf: This makes it an ideal use case for RandomizedSearchCV.
# As before, the feature array X and target variable array y of the diabetes dataset have been pre-loaded.
# The hyperparameter settings have been specified for you. Your goal is to use RandomizedSearchCV to find the
# optimal hyperparameters. Go for it!


# Setup the parameters and distributions to sample from: param_dist
param_dist = {"max_depth": [3, None],
              "max_features": randint(1, 9),
              "min_samples_leaf": randint(1, 9),
              "criterion": ["gini", "entropy"]}

# Instantiate a Decision Tree classifier: tree
tree = DecisionTreeClassifier()

# Instantiate the RandomizedSearchCV object: tree_cv
tree_cv = RandomizedSearchCV(tree, param_dist, cv=5)

# Fit it to the data
tree_cv.fit(X, y)

# Print the tuned parameters and score
print("Tuned Decision Tree Parameters: {}".format(tree_cv.best_params_))
print("Best score is {}".format(tree_cv.best_score_))

#Hold-out set in practice I: Classification
#You will now practice evaluating a model with tuned hyperparameters on a hold-out set.
# The feature array and target variable array from the diabetes dataset have been pre-loaded as X and y.
# In addition to C, logistic regression has a 'penalty' hyperparameter which specifies whether
# to use 'l1' or 'l2' regularization. Your job in this exercise is to create a hold-out set,
# tune the 'C' and 'penalty' hyperparameters of a logistic regression classifier using GridSearchCV on the training set.
# Create the hyperparameter grid
c_space = np.logspace(-5, 8, 15)
param_grid = {'C': c_space, 'penalty': ['l1', 'l2']}

# Instantiate the logistic regression classifier: logreg
logreg = LogisticRegression()

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state=42)

# Instantiate the GridSearchCV object: logreg_cv
logreg_cv = GridSearchCV(logreg, param_grid, cv=5)

# Fit it to the training data
logreg_cv.fit(X_train, y_train)

# Print the optimal parameters and best score
print("Tuned Logistic Regression Parameter: {}".format(logreg_cv.best_params_))
print("Tuned Logistic Regression Accuracy: {}".format(logreg_cv.best_score_))


#Hold-out set in practice II: Regression
#Remember lasso and ridge regression from the previous chapter? Lasso used the L1 penalty to regularize,
# while ridge used the L2 penalty. There is another type of regularized regression known as the elastic net.
# In elastic net regularization, the penalty term is a linear combination of the L1 and L2 penalties: a∗L1+b∗L2
#In scikit-learn, this term is represented by the 'l1_ratio' parameter: An 'l1_ratio' of 1 corresponds to an L1 penalty,
# and anything lower is a combination of L1 and L2.
#In this exercise, you will GridSearchCV to tune the 'l1_ratio' of an elastic net model trained on the Gapminder data.
# As in the previous exercise, use a hold-out set to evaluate your model's performance.

# Read the CSV file into a DataFrame: df
df = pd.read_csv('../../DataSets/gapminder/gapminder.csv')

y = df['life'].values
df = df.drop(columns=['life', 'Region'])
X = (df.values)

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state=42)

# Create the hyperparameter grid
l1_space = np.linspace(0, 1, 30)
param_grid = {'l1_ratio': l1_space}

# Instantiate the ElasticNet regressor: elastic_net
elastic_net = ElasticNet()

# Setup the GridSearchCV object: gm_cv
gm_cv = GridSearchCV(elastic_net, param_grid, cv=5)

# Fit it to the training data
gm_cv.fit(X_train, y_train)

# Predict on the test set and compute metrics
y_pred = gm_cv.predict(X_test)
r2 = gm_cv.score(X_test, y_test)
mse = mean_squared_error(y_test, y_pred)
print("Tuned ElasticNet l1 ratio: {}".format(gm_cv.best_params_))
print("Tuned ElasticNet R squared: {}".format(r2))
print("Tuned ElasticNet MSE: {}".format(mse))

