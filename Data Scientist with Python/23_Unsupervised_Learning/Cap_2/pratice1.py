#!/usr/bin/env python
# -*- coding: utf-8 -*-

from scipy.cluster.hierarchy import linkage, dendrogram
import pandas as pd
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt

#Hierarchical clustering of the grain data
#In the video, you learned that the SciPy linkage()
#function performs hierarchical clustering on an array of samples.
## Use the linkage() function to obtain a hierarchical clustering of the grain samples,
## and use dendrogram() to visualize the result.
## A sample of the grain measurements is provided in the array samples,
## while the variety of each grain sample is given by the list varieties.
# Calculate the linkage: mergings
# Perform the necessary imports


seeds = pd.read_csv('C:\Projetos\Python\DataCamp\DataSets\seeds\seeds.csv')
varietisSeeds = pd.read_csv('C:\Projetos\Python\DataCamp\DataSets\seeds\\varietiesSeeds.csv')
#stockMovements = pd.read_csv('C:\Projetos\Python\precos_normalizados.csv').iloc[2:]
stockMovements = pd.read_csv('sumario_15.csv').iloc[2:].set_index(['cod_neg']).join(pd.read_csv('sumario_30.csv').iloc[2:].set_index(['cod_neg']), rsuffix='30').join(pd.read_csv('sumario_60.csv').iloc[2:].set_index(['cod_neg']), rsuffix='60').join(pd.read_csv('sumario_120.csv').iloc[2:].set_index(['cod_neg']), rsuffix='120').join(pd.read_csv('sumario_365.csv').iloc[2:].set_index(['cod_neg']), rsuffix='365')
stockMovements = stockMovements.reset_index()
stockMovements = stockMovements.fillna(0)

stockMovements_values = stockMovements[stockMovements.columns[1:]]
stockMovements_cias = stockMovements[stockMovements.columns[0:1]]


#stockMovements = stockMovements[stockMovements.columns[0:1]].values
#cias = stockMovements[stockMovements.columns[1:]].values    
#stockMovements = stockMovements.set_index('Unnamed: 0')
#stockMovements = stockMovements.T

print(seeds)
# Calculate the linkage: mergings
mergings = linkage(seeds, method='complete')
print(mergings)
print(seeds.values)

print(varietisSeeds.values)
# Plot the dendrogram, using varieties as labels
dendrogram(mergings, leaf_rotation=90, leaf_font_size=5, labels=varietisSeeds.values) 


plt.show()

#Hierarchies of stocks
#In chapter 1, you used k-means clustering to cluster companies according to
# their stock price movements. Now, you'll perform hierarchical
# clustering of the companies. You are given a NumPy array of price movements movements, where the rows correspond to companies, and a list of the company names companies.
# SciPy hierarchical clustering doesn't fit into a sklearn pipeline,
# so you'll need to use the normalize() function from sklearn.preprocessing instead of Normalizer.
# linkage and dendrogram have already been imported from sklearn.cluster.hierarchy, and PyPlot has been imported as plt.

# Normalize the movements: normalized_movements
#normalized_movements = normalize(stockMovements.values)


# Calculate the linkage: mergings
mergings = linkage(stockMovements_values, method='complete')

# Plot the dendrogram
dendrogram(mergings, labels=stockMovements_cias.values, leaf_rotation=90, leaf_font_size=6)

plt.show()