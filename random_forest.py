import pandas as pd
from sklearn import metrics
import joblib


from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier

## full dataset ##
train = pd.read_csv("train.csv").drop(["Unnamed: 0","Unnamed: 0.1"], axis = 1)
test = pd.read_csv("test.csv").drop(["Unnamed: 0","Unnamed: 0.1"], axis = 1)
X_train,y_train = train.loc[:, train.columns != 'Primary Type'], train['Primary Type']
X_test,y_test = test.loc[:, test.columns != 'Primary Type'], test['Primary Type']

clf = RandomForestClassifier(n_estimators=120)
X_train_no_beat = X_train.loc[:,~X_train.columns.str.contains("^Beat")]
X_test_no_beat = X_test.loc[:,~X_test.columns.str.contains("^Beat")]

clf.fit(X_train, y_train)
y_pred=clf.predict(X_test)
joblib.dump(clf, 'randomForestModel.pkl')
print("Accuracy randomForest no beats:",metrics.accuracy_score(y_test, y_pred))
feature_names = [f'feature {c}' for c in X_train.columns]
feature_imp = pd.Series(clf.feature_importances_,index=feature_names).sort_values(ascending=False)
feature_imp.to_csv("feature_importance.csv")
clf = AdaBoostClassifier(n_estimators=120)
clf.fit(X_train_no_beat, y_train)
y_pred = clf.predict(X_test_no_beat)
joblib.dump(clf, 'adaBoostModel.pkl')
print("Accuracy Adaboost no beats full dataset: ", metrics.accuracy_score(y_test, y_pred))

clf = GradientBoostingClassifier(n_estimators=120)
clf.fit(X_train_no_beat, y_train)
y_pred = clf.predict(X_test_no_beat)
joblib.dump(clf, 'gradientBoostingModel.pkl')
print("Accuracy GradientBoosting full no beats dataset: ", metrics.accuracy_score(y_test, y_pred))


