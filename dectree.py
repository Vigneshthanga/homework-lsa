#!/usr/bin/python
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn import preprocessing
from sklearn import tree

data = pd.read_csv('data.csv', sep=',', header = None)
data = data.drop(data.columns[0], axis='columns')
print(data.head())


data = data.apply(preprocessing.LabelEncoder().fit_transform)


# Seperating the target variable 
X = data.values[:, 0:3]
Y = data.values[:, 4]

# Spliting the dataset into train and test 
train=data.sample(frac=0.7,random_state=200)
test=data.drop(train.index)

train_x = train.values[:, 0:4]
train_y = train.values[:, 4]

# Decision tree with entropy
clf_entropy = DecisionTreeClassifier(
            criterion = "entropy", random_state = 100)

# Performing training
clf_entropy.fit(train_x, train_y)

test_x = test.values[:, 0:4]
test_y = test.values[:, 4]

y_pred = clf_entropy.predict(test_x)
print("Predicted values:")
print(y_pred)

print(accuracy_score(test_y,y_pred)*100)

print(tree.plot_tree(clf_entropy))
