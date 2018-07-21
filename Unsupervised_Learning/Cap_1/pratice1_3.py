import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn.datasets import make_classification
from sklearn.datasets import make_blobs
from sklearn.datasets import make_gaussian_quantiles
import dataset as ds

#PRATICE 1
#plt.figure(figsize=(8, 8))
#plt.subplots_adjust(bottom=.05, top=.9, left=.05, right=.95)
#plt.subplot(321)

#You are given an array POINTS of size 300x2, where each row gives the (x, y) co-ordinates of a point on a map
x1Train, y1Train = make_blobs(n_samples=300, n_features=2, centers=3, center_box=(-10.0, 10.0), shuffle=False, random_state=None)
plt.scatter(x1Train[:, 0], x1Train[:, 1], marker='o', c=y1Train,  s=25, edgecolor='k')
#plt.show()

#PRATICE 2
# Create a KMeans instance with 3 clusters: model
model = KMeans(n_clusters=3)

# Fit model to points
model.fit(x1Train)

# Determine the cluster labels of new_points: labels
x1NewPoints, y1NewPoints = make_blobs(n_samples=300, n_features=2, centers=3, center_box=(-10.0, 10.0), shuffle=False, random_state=None)
labels = model.predict(x1NewPoints)

#Print cluster labels of new_points
print(labels)

#plt.scatter(x1NewPoints[:, 0], x1NewPoints[:, 1], marker='o', c=labels,  s=25, edgecolor='k')

# Assign the cluster centers: centroids
centroids = model.cluster_centers_

# Assign the columns of centroids: centroids_x, centroids_y
centroids_x = centroids[:,0]
centroids_y = centroids[:,1]

# Make a scatter plot of centroids_x and centroids_y
plt.scatter(centroids_x, centroids_y, marker='D', s=50)
plt.show()



