#The first principal component
#The first principal component of the data is the direction in which the data varies the most.
# In this exercise, your job is to use PCA to find the first principal component of the length and width measurements
# of the grain samples, and represent it as an arrow on the scatter plot.
#The array grains gives the length and width of the grain samples. PyPlot (plt) and PCA have already been imported for you.

# Perform the necessary imports
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

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

# Make a scatter plot of the untransformed points
plt.scatter(grains["Length"], grains["Width"])

# Create a PCA instance: model
model = PCA()

# Fit model to points
model.fit(grains.values)

# Get the mean of the grain samples: mean
mean = model.mean_

# Get the first principal component: first_pc
first_pc = model.components_[0,:]

# Plot first_pc as an arrow, starting at mean
plt.arrow(mean[0], mean[1], first_pc[0], first_pc[1], color='red', width=0.01)

# Keep axes on same scale
plt.axis('equal')
plt.show()

#Variance of the PCA features
#The fish dataset is 6-dimensional. But what is its intrinsic dimension? Make a plot of the variances
# of the PCA features to find out. As before, samples is a 2D array,
# where each row represents a fish. You'll need to standardize the features first.

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt

# Create scaler: scaler
scaler = StandardScaler()

# Create a PCA instance: pca
pca = PCA()

# Create pipeline: pipeline
pipeline = make_pipeline(scaler, pca)

# Fit the pipeline to 'samples'
pipeline.fit(fishes)

# Plot the explained variances
features = range(pca.n_components_)
plt.bar(features, pca.explained_variance_)
plt.xlabel('PCA feature')
plt.ylabel('variance')
plt.xticks(features)
plt.show()

# Dimension reduction of the fish measurements
# In a previous exercise, you saw that 2 was a reasonable choice for the "intrinsic dimension" of the
# fish measurements. Now use PCA for dimensionality reduction of the fish measurements,
# retaining only the 2 most important components.
# The fish measurements have already been scaled for you, and are available as scaled_samples.
# Import PCA

# Import PCA
from sklearn.decomposition import PCA

# Create a PCA model with 2 components: pca
pca = PCA(n_components=2)

# Fit the PCA instance to the scaled samples
pca.fit(fishes)
print(fishes)

# Transform the scaled samples: pca_features
pca_features = pca.transform(fishes)

# Print the shape of pca_features
print(pca_features.shape)


