#Metrics for classification
#In Chapter 1, you evaluated the performance of your k-NN classifier based on its accuracy.
# However, as Andy discussed, accuracy is not always an informative metric. In this exercise,
# you will dive more deeply into evaluating the performance of binary classifiers by computing a confusion matrix and generating a classification report.
#You may have noticed in the video that the classification report consisted of three rows,
# and an additional support column. The support gives the number of samples of the true response
# that lie in that class - so in the video example, the support was the number of Republicans or
# Democrats in the test set on which the classification report was computed. The precision, recall,
# and f1-score columns, then, gave the respective metrics for that particular class.
#Here, you'll work with the PIMA Indians dataset obtained from the UCI Machine Learning Repository.
# The goal is to predict whether or not a given female patient will contract diabetes based on features
# such as BMI, age, and number of pregnancies. Therefore, it is a binary classification problem.
# A target value of 0 indicates that the patient does not have diabetes, while a value of 1
# indicates that the patient does have diabetes. As in Chapters 1 and 2, the dataset has been
# preprocessed to deal with missing values.
#The dataset has been loaded into a DataFrame df and the feature and target variable
# arrays X and y have been created for you. In addition, sklearn.model_selection.train_test_
# split and sklearn.neighbors.KNeighborsClassifier have already been imported.
#Your job is to train a k-NN classifier to the data and evaluate its performance by generating a
# confusion matrix and classification report.

# Import necessary modules
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score

seeds = pd.read_csv('../../DataSets/seeds/seeds.csv')
varietisSeeds = pd.read_csv('../../DataSets/seeds/varietiesSeeds.csv')
diabetes = pd.read_csv('../../DataSets/diabets/diabetes.csv')
stockMovements = pd.read_csv('../../DataSets/stocks/StockMovements.csv')
stockMovements = stockMovements.set_index('Unnamed: 0')
stockMovements = stockMovements.T


y = diabetes["Outcome"].values
X = diabetes.drop(columns="Outcome").values
# Create training and test set
X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=0.4, random_state=42)

# Instantiate a k-NN classifier: knn
knn = KNeighborsClassifier(n_neighbors=6)

# Fit the classifier to the training data
knn.fit(X_train, y_train)

# Predict the labels of the test data: y_pred
y_pred = knn.predict(X_test)

# Generate the confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

#Building a logistic regression model
#Time to build your first logistic regression model! As Hugo showed in the video, scikit-learn makes it very easy
# to try different models, since the Train-Test-Split/Instantiate/Fit/Predict paradigm applies to all
# classifiers and regressors - which are known in scikit-learn as 'estimators'. You'll see this now for yourself as you
# train a logistic regression model on exactly the same data as in the previous exercise.
# Will it outperform k-NN? There's only one way to find out!
# The feature and target variable arrays X and y have been pre-loaded, and train_test_split has been imported for you from sklearn.model_selection.

# Create the classifier: logreg
logreg =  LogisticRegression()

# Fit the classifier to the training data
logreg.fit(X_train, y_train)

# Predict the labels of the test set: y_pred
y_pred =  logreg.predict(X_test)

# Compute and print the confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Compute predicted probabilities: y_pred_prob
y_pred_prob =  logreg.predict_proba(X_test)[:,1]

# Generate ROC curve values: fpr, tpr, thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

# Plot ROC curve
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()


#AUC computation
# Say you have a binary classifier that in fact is just randomly making guesses. It would be correct
# approximately 50% of the time, and the resulting ROC curve would be a diagonal line in which the
# True Positive Rate and False Positive Rate are always equal. The Area under this ROC curve would be 0.5.
# This is one way in which the AUC, which Hugo discussed in the video, is an informative metric to evaluate
# a model. If the AUC is greater than 0.5, the model is better than random guessing. Always a good sign!
# In this exercise, you'll calculate AUC scores using the roc_auc_score() function from sklearn.metrics
# as well as by performing cross-validation on the diabetes dataset.
# X and y, along with training and test sets X_train, X_test, y_train, y_test,
# have been pre-loaded for you, and a logistic regression classifier logreg has been fit to the training data.

# Compute predicted probabilities: y_pred_prob
y_pred_prob = logreg.predict_proba(X_test)[:,1]

# Compute and print AUC score
print("AUC: {}".format(roc_auc_score(y_test, y_pred_prob)))

# Compute cross-validated AUC scores: cv_auc
cv_auc = cross_val_score(logreg, X, y, cv=5,scoring='roc_auc')

# Print list of AUC scores
print("AUC scores computed using 5-fold cross-validation: {}".format(cv_auc))
