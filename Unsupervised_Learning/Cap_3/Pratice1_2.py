#Correlated data in nature
#You are given an array grains giving the width and length of samples of grain. You suspect that width and length will be correlated.
#To confirm this, make a scatter plot of width vs length and measure their Pearson correlation.

# Perform the necessary imports
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import pandas as pd
import matplotlib.pyplot as plt

seeds = pd.read_csv('../../DataSets/seeds/seeds.csv')
varietisSeeds = pd.read_csv('../../DataSets/seeds/varietiesSeeds.csv')
fishes = pd.read_csv('../../DataSets/fishes/fishes.csv')
speciesFishes = pd.read_csv('../../DataSets/fishes/speciesFishes.csv')
stockMovements = pd.read_csv('../../DataSets/stocks/StockMovements.csv')
stockMovements = stockMovements.set_index('Unnamed: 0')
grains = pd.DataFrame()
grains.insert(loc=0, column="Width", value=seeds["width of kernel"].values)
grains.insert(loc=0, column="Length", value=seeds["length of kernel"].values)
dfAux = stockMovements.copy()

# Assign the 0th column of grains: width
width = grains["Width"]

# Assign the 1st column of grains: length
length = grains["Length"]

# Scatter plot width vs length
plt.scatter(width, length)
plt.axis('equal')
plt.show()

# Calculate the Pearson correlation
correlation, pvalue = pearsonr(width, length)

# Display the correlation
print(correlation)

#Decorrelating the grain measurements with PCA
#You observed in the previous exercise that the width and length measurements of the grain are correlated.
# Now, you'll use PCA to decorrelate these
# measurements, then plot the decorrelated points and measure their Pearson correlation.
# Import PCA
from sklearn.decomposition import PCA

# Create PCA instance: model
model = PCA()

# Apply the fit_transform method of model to grains: pca_features
pca_features = model.fit_transform(grains)

# Assign 0th column of pca_features: xs
xs = pca_features[:,0]

# Assign 1st column of pca_features: ys
ys = pca_features[:,1]

# Scatter plot xs vs ys
plt.scatter(xs, ys)
plt.axis('equal')
plt.show()


#The first principal component of the data is the direction in which the data varies the most.
# In this exercise, your job is to use PCA to find the first principal component of the length and width measurements
# of the grain samples, and represent it as an arrow on the scatter plot.
#The array grains gives the length and width of the grain samples. PyPlot (plt)
# and PCA have already been imported for you.

