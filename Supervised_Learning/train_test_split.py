#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
pd.option_context('display.max_rows', 999, 'display.max_columns', 999)

iris = datasets.load_iris()

#Imprime o tipo do dataset, BUNCH quer dizer que é similar a um dicionário com valor e chave
print("=================================================")
print("Dataset description:")
print(iris.DESCR)
print("Dataset type:")
print(type(iris))
print("\nData type:")
print(type(iris.data))
print("\nTargets type:")
print(type(iris.target))
print("\nFeature names:")
print(iris.feature_names)
print("\nDataset keys:")
print(iris.keys())
#Lista a quantidade de linhas e colunas
print("\nDataset data shape [rows, column]:")
print(iris.data.shape)
print("\nDataset target shape [rows, column]:")
print(iris.target.shape)
#Imprime os tipos de classificações
print("\nTarget Names: ")
print(iris.target_names)
print("=================================================")


X = iris.data
y = iris.target

#print(X)

#Criar um data frame com os dados e as fetaures do dataset
df = pd.DataFrame(X, columns=iris.feature_names)

with pd.option_context('display.max_rows', 999, 'display.max_columns', 4):
    print(df.head())

pd.plotting.scatter_matrix(df, c = y, figsize = [8, 8], s=150, marker='D')
plt.show()

X_train,X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=21, stratify=y)

knn =  KNeighborsClassifier(n_neighbors=6)

print(knn.fit(X_train, y_train))

predition = knn.predict(X_test)

print("Test set prediction:\n {}".format(predition))

print(knn.score(X_test, y_test))

#Overfitting Undefitting
# Setup arrays to store train and test accuracies
neighbors = np.arange(1, 9)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

# Loop over different values of k
for i, k in enumerate(neighbors):
    # Setup a k-NN Classifier with k neighbors: knn
    knn = KNeighborsClassifier(n_neighbors=k)

    # Fit the classifier to the training data
    knn.fit(X_train, y_train)

    #Compute accuracy on the training set
    train_accuracy[i] = knn.score(X_train, y_train)

    #Compute accuracy on the testing set
    test_accuracy[i] = knn.score(X_test, y_test)

# Generate plot
plt .title('k-NN: Varying Number of Neighbors')
plt.show()










