#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

#PRATICE 4
#How many clusters of grain?
#samples containing the measurements (such as area, perimeter, length, and several others) of samples of grain. What's a good number of clusters in this case?
#KMeans and PyPlot (plt) have already been imported for you.
#This dataset was sourced from the UCI Machine Learning Repository.
#Choose an "elbow" in the inertia plot

seeds = pd.read_csv('../../DataSets/seeds/seeds.csv')
varietisSeeds = pd.read_csv('../../DataSets/seeds/varietiesSeeds.csv')
fishes = pd.read_csv('../../DataSets/fishes/fishes.csv')
speciesFishes = pd.read_csv('../../DataSets/fishes/speciesFishes.csv')
stockMovements = pd.read_csv('../../DataSets/stocks/StockMovements.csv')
stockMovements = stockMovements.set_index('Unnamed: 0')
df = pd.DataFrame(data=stockMovements)
df.loc[(df['BBSE3'] == df['IBOVESPA']), 'Signal'] = 1
df.loc[(df['BBSE3'] != df['IBOVESPA']), 'Signal'] = 0

print(df['Signal'].sum())
print(df['Signal'].__len__())
stockMovements = stockMovements.T

ks = range(1, 10)
inertias = []

for k in ks:
    # Create a KMeans instance with k cluster s: model
    model = KMeans(n_clusters=k)

    # Fit model to samples
    model.fit(seeds)

    # Append the inertia to the list of inertias
    inertias.append(model.inertia_)

# Plot ks vs inertias
plt.plot(ks, inertias, '-o')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()

#PRATICE 5
# Create a KMeans model with 3 clusters: model
model = KMeans(n_clusters=3 )

# Use fit_predict to fit model and obtain cluster labels: labels
#In fact, the grain samples come from a mix of 3 different grain varieties: "Kama", "Rosa" and "Canadian"
labels = model.fit_predict(seeds)
print(labels)
print(varietisSeeds['varietis'])
print(labels.shape)
print(varietisSeeds['varietis'].shape)
print(seeds.iloc[0])

plt.scatter(seeds['area A'], seeds['perimeter P'], marker='o', c=labels, edgecolor='k')
plt.xlabel('area A')
plt.ylabel('perimeter P')
plt.show()

# Create a DataFrame with labels and varieties as columns: df
df = pd.DataFrame({'labels': labels, 'varieties': varietisSeeds['varietis']})

# Create crosstab: ct
ct = pd.crosstab(df['labels'], df['varieties'])

# Display ct
print(ct)

#PRATICE 6 (StandardScaler, Pipeline)
##########WITH SEEDS#############
scaler = StandardScaler()
scaled_data = scaler.fit_transform(seeds)
print(scaled_data[0,:])

plt.scatter(scaled_data[:,0], scaled_data[:,1], marker='o', c=labels, edgecolor='k')
plt.xlabel('area A - standardized')
plt.ylabel('perimeter P - standardized')
plt.show()

kmeans = KMeans(n_clusters=3)
pipeline = make_pipeline(scaler, kmeans)

pipeline.fit(seeds)
labels = pipeline.predict(seeds)
print(labels)

# Create a DataFrame with labels and varieties as columns: df
df = pd.DataFrame({'labels': labels, 'varieties': varietisSeeds['varietis']})
ct = pd.crosstab(df['labels'], df['varieties'])
print(ct)

##########WITH FISHES#############
#Scaling fish data for clustering
#You are given an array samples giving measurements of fish. Each row represents an individual fish. The measurements, such as weight in grams, length in centimeters, and the percentage ratio of height to length, have very different scales. In order to cluster this data effectively, you'll need to standardize these features first. In this exercise, you'll build a pipeline to standardize and cluster the data.
#These fish measurement data were sourced from the Journal of Statistics Education.

ks = range(1, 20)
inertias = []

for k in ks:
    # Create a KMeans instance with k cluster s: model
    model = KMeans(n_clusters=k)

    # Fit model to samples
    model.fit(fishes)

    # Append the inertia to the list of inertias
    inertias.append(model.inertia_)

# Plot ks vs inertias
plt.plot(ks, inertias, '-o')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()

# Create scaler: scaler
scaler =  StandardScaler()

# Create KMeans instance: kmeans
kmeans =  KMeans(n_clusters=6)

# Create pipeline: pipeline
pipeline =  make_pipeline(scaler, kmeans)

# Fit the pipeline to samples
pipeline.fit(fishes)
print(fishes)

# Calculate the cluster labels: labels
labels =  pipeline.predict(fishes)

# Create a DataFrame with labels and species as columns: df
df = pd.DataFrame({'labels': labels, 'species': speciesFishes["species"]})

# Create crosstab: ct
ct = pd.crosstab(df['labels'], df['species'])
print(ct)

labels = kmeans.fit_predict(fishes)

# Create a DataFrame with labels and species as columns: df
df = pd.DataFrame({'labels': labels, 'species': speciesFishes["species"]})

# Create crosstab: ct
ct = pd.crosstab(df['labels'], df['species'])
print(ct)

#PRATICE_8
#Which stocks move together?
#In the previous exercise, you clustered companies by their daily stock price movements.
#So which company have stock prices that tend to change in the same way?
#You'll now inspect the cluster labels from your clustering to find out.
#Your solution to the previous exercise has already been run.
#Recall that you constructed a Pipeline pipeline containing a KMeans model and fit it to the
#NumPy array movements of daily stock movements. In addition, a list companies of the company names is available.

# Create a normalizer: normalizer
normalizer = Normalizer()

# Create a KMeans model with 10 clusters: kmeans

kmeans = KMeans(n_clusters=7)
#labels = kmeans.fit_predict(stockMovements)


# Make a pipeline chaining normalizer and kmeans: pipeline
pipeline = make_pipeline(normalizer, kmeans)

# Fit pipeline to the daily price movements
pipeline.fit(stockMovements)

# Predict the cluster labels: labels
labels = pipeline.predict(stockMovements)

# Create a DataFrame aligning labels and companies: df
df = pd.DataFrame({'labels': labels, 'companies': stockMovements.index})

with pd.option_context('display.max_rows', 999, 'display.max_columns', 100):
    # Display df sorted by cluster label
    print(df.sort_values('labels'))


# Create a DataFrame aligning labels and companies: df
#df = pd.DataFrame({'labels': labels, 'companies': companies})

# Display df sorted by cluster label
#print(df.sort_values('labels'))

