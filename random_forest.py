import pandas as pd
import numpy as np
import clean_data
from sklearn import metrics


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=125, criterion='mse')
## full dataset ##
train = pd.read_csv("train.csv").drop(["Unnamed: 0","Unnamed: 0.1"], axis = 1)
test = pd.read_csv("test.csv").drop(["Unnamed: 0","Unnamed: 0.1"], axis = 1)
X_train,y_train = train.loc[:, train.columns != 'Primary Type'], train['Primary Type']
X_test,y_test = test.loc[:, test.columns != 'Primary Type'], test['Primary Type']
clf.fit(X_train, y_train)
y_pred=clf.predict(X_test)
print("Accuracy full dataset:",metrics.accuracy_score(y_test, y_pred))
## No Beats ##
X_train_no_beat = X_train.loc[:,~X_train.columns.str.contains("^Beat")]
X_test_no_beat = X_test.loc[:,~X_test.columns.str.contains("^Beat")]
clf.fit(X_train_no_beat, y_train)
y_pred=clf.predict(X_test_no_beat)
print("Accuracy no beat:",metrics.accuracy_score(y_test, y_pred))
## No orient ##
X_train_no_orient = X_train.loc[:,~X_train.columns.str.contains("^orient")]
X_test_no_orient = X_test.loc[:,~X_test.columns.str.contains("^orient")]
clf.fit(X_train_no_orient, y_train)
y_pred=clf.predict(X_test_no_orient)
print("Accuracy no orient:",metrics.accuracy_score(y_test, y_pred))
## no coordinates##
X_train_no_coordinate = X_train.loc[:,~X_train.columns.str.contains("Coordinate")]
X_test_no_coordinate = X_test.loc[:,~X_test.columns.str.contains("Coordinate")]
clf.fit(X_train_no_coordinate, y_train)
y_pred=clf.predict(X_test_no_coordinate)
print("Accuracy no coordinate:",metrics.accuracy_score(y_test, y_pred))