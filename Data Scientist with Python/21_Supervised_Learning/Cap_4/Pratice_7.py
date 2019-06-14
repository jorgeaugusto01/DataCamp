#Centering and scaling your data
#In the video, Hugo demonstrated how significantly the performance of a model can improve if the
# features are scaled. Note that this is not always the case: In the Congressional voting records dataset,
# for example, all of the features are binary. In such a situation, scaling will have minimal impact.
#You will now explore scaling for yourself on a new dataset - White Wine Quality! Hugo used the Red Wine
# Quality dataset in the video. We have used the 'quality' feature of the wine to create a binary target variable:
# If 'quality' is less than 5, the target variable is 1, and otherwise, it is 0.
#The DataFrame has been pre-loaded as df, along with the feature and target variable arrays X and y.
# Explore it in the IPython Shell. Notice how some features seem to have different units of measurement. 'density',
# for instance, takes values between 0.98 and 1.04, while 'total sulfur dioxide' ranges from 9 to 440. As a result,
# it may be worth scaling the features here. Your job in this exercise is to scale the features and compute the mean and
# standard deviation of the unscaled features compared to the scaled features.

# Import scale
from sklearn.preprocessing import scale
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd

df = pd.read_csv('../../DataSets/wine/white-wine.csv')
df.loc[df['quality'] < 5, 'quality'] = True
df.loc[df['quality'] >= 5, 'quality'] = False

y = df['quality'].values
X = df.drop(columns='quality').values


# Scale the features: X_scaled
X_scaled =  scale(X)

# Print the mean and standard deviation of the unscaled features
print("Mean of Unscaled Features: {}".format(np.mean(X)))
print("Standard Deviation of Unscaled Features: {}".format(np.std(X)))

# Print the mean and standard deviation of the scaled features
print("Mean of Scaled Features: {}".format(np.mean(X_scaled)))
print("Standard Deviation of Scaled Features: {}".format(np.std(X_scaled)))

#Centering and scaling in a pipeline
#With regard to whether or not scaling is effective, the proof is in the pudding!
# See for yourself whether or not scaling the features of the White Wine Quality dataset has any
# impact on its performance. You will use a k-NN classifier as part of a pipeline that includes scaling,
# and for the purposes of comparison, a k-NN classifier trained on the unscaled data has been provided.
#The feature array and target variable array have been pre-loaded as X and y. Additionally, KNeighborsClassifier
# and train_test_split have been imported from sklearn.neighbors and sklearn.model_selection, respectively.
# Import the necessary modules
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split


print(X)
print(y)
# Setup the pipeline steps: steps
steps = [('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier())]

# Create the pipeline: pipeline
pipeline =  Pipeline(steps)

# Create train and test sets
X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=0.3, random_state=42)

# Fit the pipeline to the training set: knn_scaled
knn_scaled =  pipeline.fit(X_train, y_train)

# Instantiate and fit a k-NN classifier to the unscaled data
knn_unscaled = KNeighborsClassifier().fit(X_train, y_train)

# Compute and print metrics
print('Accuracy with Scaling: {}'.format(knn_scaled.score(X_test, y_test)))
print('Accuracy without Scaling: {}'.format(knn_unscaled.score(X_test, y_test)))

#Bringing it all together I: Pipeline for classification
#It is time now to piece together everything you have learned so far into a pipeline for classification!
# Your job in this exercise is to build a pipeline that includes scaling and hyperparameter tuning to classify wine quality.
#You'll return to using the SVM classifier you were briefly introduced to earlier in this chapter.
# The hyperparameters you will tune are C and gamma. C controls the regularization strength.
# It is analogous to the C you tuned for logistic regression in Chapter 3, while gamma controls the kernel coefficient:
# Do not worry about this now as it is beyond the scope of this course.
#The following modules have been pre-loaded: Pipeline, svm, train_test_split, GridSearchCV, classification_report, accuracy_score.
# The feature and target variable arrays X and y have also been pre-loaded.

# Setup the pipeline
steps = [('scaler', StandardScaler()),
         ('SVM', SVC())]

pipeline = Pipeline(steps)

# Specify the hyperparameter space
parameters = {'SVM__C':[1, 10, 100],
              'SVM__gamma':[0.1, 0.01]}

# Create train and test sets
X_train, X_test, y_train, y_test = X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=21)

# Instantiate the GridSearchCV object: cv
cv = GridSearchCV(pipeline, parameters, cv=3)

# Fit to the training set
cv.fit(X_train, y_train)

# Predict the labels of the test set: y_pred
y_pred = cv.predict(X_test)

# Compute and print metrics
print("Accuracy: {}".format(cv.score(X_test, y_test)))
print(classification_report(y_test, y_pred))
print("Tuned Model Parameters: {}".format(cv.best_params_))

#Bringing it all together II: Pipeline for regression
#For this final exercise, you will return to the Gapminder dataset.
# Guess what? Even this dataset has missing values that we dealt with for you in earlier chapters! Now, you have all the tools to take care of them yourself!
#Your job is to build a pipeline that imputes the missing data, scales the features, and fits an ElasticNet to the Gapminder data.
# You will then tune the l1_ratio of your ElasticNet using GridSearchCV.
#All the necessary modules have been imported, and the feature and target variable arrays have been pre-loaded as X and y.






