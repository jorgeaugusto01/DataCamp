from scipy.cluster.hierarchy import linkage, dendrogram
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score

#Exploring categorical features
# The Gapminder dataset that you worked with in previous chapters also contained a
# categorical 'Region' feature, which we dropped in previous exercises since you did not have the tools to deal with it.
# Now however, you do, so we have added it back in!
# Your job in this exercise is to explore this feature. Boxplots are particularly
# useful for visualizing categorical features such as this.

# Read the CSV file into a DataFrame: df
df = pd.read_csv('../../DataSets/gapminder/gapminder.csv')

# Create a boxplot of life expectancy per region
df.boxplot('life', 'Region', rot=60)

# Show the plot
plt.show()

#Creating dummy variables
# As Andy discussed in the video, scikit-learn does not accept non-numerical features.
# You saw in the previous exercise that the 'Region' feature contains very useful information
# that can predict life expectancy. For example, Sub-Saharan Africa has a lower life expectancy
# compared to Europe and Central Asia. Therefore, if you are trying to predict life expectancy,
# it would be preferable to retain the 'Region' feature. To do this, you need to binarize it by
# creating dummy variables, which is what you will do in this exercise.

# Create dummy variables: df_region
df_region = pd.get_dummies(df)

# Print the columns of df_region
print(df_region.columns)

# Create dummy variables with drop_first=True: df_region
df_region = pd.get_dummies(df, drop_first=True)

# Print the new columns of df_region
print(df_region.columns)

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

