from scipy.cluster.hierarchy import linkage, dendrogram
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#Importing data for supervised learning
#In this chapter, you will work with Gapminder data that we have consolidated into one CSV file
# available in the workspace as 'gapminder.csv'. Specifically, your goal will be to use this data
# to predict the life expectancy in a given country based on features such as the country's GDP,
# fertility rate, and population. As in Chapter 1, the dataset has been preprocessed.
# Since the target variable here is quantitative, this is a regression problem. To begin, you will
# fit a linear regression with just one feature: 'fertility', which is the average number of children a woman
# in a given country gives birth to. In later exercises, you will use all the features to build regression models.
# Before that, however, you need to import the data and get it into the form needed by scikit-learn.
# This involves creating feature and target variable arrays. Furthermore, since you are going to use
# only one feature to begin with, you need to do some reshaping using NumPy's .reshape() method.
# Don't worry too much about this reshaping right now, but it is something you will have to do occasionally
# when working with scikit-learn so it is useful to practice.

# Read the CSV file into a DataFrame: df
df = pd.read_csv('../../DataSets/gapminder/gapminder.csv')

# Create arrays for features and target variable
y = df['life'].values
X = df['fertility'].values

# Print the dimensions of X and y before reshaping
print("Dimensions of y before reshaping: {}".format(y.shape))
print("Dimensions of X before reshaping: {}".format(X.shape))

# Reshape X and y
y = y.reshape(-1, 1)
X = X.reshape(-1, 1)

# Print the dimensions of X and y after reshaping
print("Dimensions of y after reshaping: {}".format(y.shape))
print("Dimensions of X after reshaping: {}".format(X.shape))

ax = sns.heatmap(df.corr(), square=True, cmap='RdYlGn')
plt.show()

#Fit & predict for regression
# Now, you will fit a linear regression and predict life expectancy using just one feature.
# You saw Andy do this earlier using the 'RM' feature of the Boston housing dataset.
# In this exercise, you will use the 'fertility' feature of the Gapminder dataset.
# Since the goal is to predict life expectancy, the target variable here is 'life'.
# The array for the target variable has been pre-loaded as y and the array for 'fertility' has been pre-loaded as X_fertility.
# A scatter plot with 'fertility' on the x-axis and 'life' on the y-axis has been generated.
# As you can see, there is a strongly negative correlation, so a linear regression should be able to
# capture this trend. Your job is to fit a linear regression and then predict the life expectancy,
# overlaying these predicted values on the plot to generate a regression line.
# You will also compute and print the R2 score using sckit-learn's .score() method.

# Import LinearRegression
from sklearn.linear_model import LinearRegression


# Create the regressor: reg
reg = LinearRegression()

# Create the prediction space
prediction_space = np.linspace(min(X), max(X)).reshape(-1,1)

# Fit the model to the data
reg.fit(X, y)

# Compute predictions over the prediction space: y_pred
y_pred = reg.predict(prediction_space)

# Print R^2
print(reg.score(X, y))

# Plot regression line
plt.plot(prediction_space, y_pred, color='black', linewidth=3)
plt.plot(x=X, y=y, style='o')
plt.scatter(X, y)
plt.show()
