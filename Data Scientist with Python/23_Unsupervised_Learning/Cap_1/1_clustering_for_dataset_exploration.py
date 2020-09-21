import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import datasets
#import dataset as ds

#PRATICE 1
#You are given an array POINTS of size 300x2, where each row gives the (x, y) co-ordinates of a point on a map

breast_cancer = datasets.load_breast_cancer()
xs = points[:,0]
ys = points[:,2]
plt.scatter(xs, ys)
plt.show()

#PRATICE 2

# Create a KMeans instance with 3 clusters: model
model = KMeans(n_clusters=3)

# Fit model to points
model.fit(points)

# Determine the cluster labels of new_points: labels
labels = model.predict(new_points)

# Print cluster labels of new_points
print(labels)

#PRATICE 3

# Assign the columns of new_points: xs and ys
xs = new_points[:,0]
ys = new_points[:,1]

# Make a scatter plot of xs and ys, using labels to define the colors
plt.scatter(xs, ys, c=labels, alpha=0.5)
plt.show()

# Assign the cluster centers: centroids
centroids = model.cluster_centers_

# Assign the columns of centroids: centroids_x, centroids_y
centroids_x = centroids[:,0]
centroids_y = centroids[:,1]

# Make a scatter plot of centroids_x and centroids_y
plt.scatter(centroids_x, centroids_y, marker='D', s=50)
plt.show()


#PRATICE 4
#How many clusters of grain?
#samples containing the measurements (such as area, perimeter, length, and several others) of samples of grain. What's a good number of clusters in this case?
#KMeans and PyPlot (plt) have already been imported for you.
#This dataset was sourced from the UCI Machine Learning Repository.
ks = range(1, 6)
inertias = []

for k in ks:
    # Create a KMeans instance with k clusters: model
    model = KMeans(n_clusters=k)

    # Fit model to samples
    model.fit(samples)

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
model = KMeans(n_clusters=3)

# Use fit_predict to fit model and obtain cluster labels: labels
#In fact, the grain samples come from a mix of 3 different grain varieties: "Kama", "Rosa" and "Canadian"
labels = model.fit_predict(samples)

# Create a DataFrame with labels and varieties as columns: df
df = pd.DataFrame({'labels': labels, 'varieties': varieties})
print(samples[0])
print(labels)
print(varieties)

# Create crosstab: ct
ct = pd.crosstab(df['labels'], df['varieties'])

# Display ct
print(ct)

#PRATICE 6
#Scaling fish data for clustering
#You are given an array samples giving measurements of fish. Each row represents an individual fish. The measurements, such as weight in grams, length in centimeters, and the percentage ratio of height to length, have very different scales. In order to cluster this data effectively, you'll need to standardize these features first. In this exercise, you'll build a pipeline to standardize and cluster the data.
#These fish measurement data were sourced from the Journal of Statistics Education.
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Create scaler: scaler
scaler =  StandardScaler()

# Create KMeans instance: kmeans
kmeans =  KMeans(n_clusters=4)

# Create pipeline: pipeline
pipeline =  make_pipeline(scaler, kmeans)

#PRATICE 7
#You'll now use your standardization and clustering pipeline from the previous exercise to cluster the fish by their measurements, and then create a cross-tabulation to compare the cluster labels with the fish species.
# Import pandas
import pandas as pd

# Fit the pipeline to samples
pipeline.fit(samples)
print(samples)

# Calculate the cluster labels: labels
labels =  pipeline.predict(samples)

# Create a DataFrame with labels and species as columns: df
df = pd.DataFrame({'labels': labels, 'species': species})

# Create crosstab: ct
ct = pd.crosstab(df['labels'], df['species'])

# Display ct


#PRATICE 8
# Import Normalizer
# Clustering stocks using KMeans
from sklearn.preprocessing import Normalizer

# Create a normalizer: normalizer
normalizer = Normalizer()

# Create a KMeans model with 10 clusters: kmeans
kmeans = KMeans(n_clusters=10)

# Make a pipeline chaining normalizer and kmeans: pipeline
pipeline = make_pipeline(normalizer, kmeans)

# Fit pipeline to the daily price movements
pipeline.fit(movements)

#PRATICE 9
# Which stocks move together?
# Import pandas
import pandas as pd

# Predict the cluster labels: labels
labels = pipeline.predict(movements)

# Create a DataFrame aligning labels and companies: df
df = pd.DataFrame({'labels': labels, 'companies': companies})

# Display df sorted by cluster label
print(df.sort_values('labels'))
