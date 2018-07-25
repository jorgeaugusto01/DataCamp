#Regularization I: Lasso
#In the video, you saw how Lasso selected out the 'RM' feature as being the most important for predicting
# Boston house prices, while shrinking the coefficients of certain other features to 0.
# Its ability to perform feature selection in this way becomes even more useful when you are dealing
# with data involving thousands of features.
# In this exercise, you will fit a lasso regression to the Gapminder data you have
# been working with and plot the coefficients. Just as with the Boston data, you will find that the
# coefficients of some features are shrunk to 0, with only the most important ones remaining.
# The feature and target variable arrays have been pre-loaded as X and y.

# Import Lasso
from sklearn.linear_model import Lasso
import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file into a DataFrame: df
df = pd.read_csv('../../DataSets/gapminder/gapminder.csv')
print(df.index)

# Create arrays for features and target variable
y = df['life'].values
X = df['fertility'].values

# Reshape X and y
y = y.reshape(-1, 1)
X = X.reshape(-1, 1)

# Instantiate a lasso regressor: lasso
lasso =  Lasso(alpha=0.4, normalize=True)

# Fit the regressor to the data
lasso.fit(X, y)

# Compute and print the coefficients
lasso_coef = lasso.coef_
print(lasso_coef)

# Plot the coefficients
plt.plot(range(len(df.index)), lasso_coef)
plt.xticks(range(len(df.index)), df.index.values, rotation=60)
plt.margins(0.02)
plt.show()