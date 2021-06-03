import xgboost
import pandas as pd
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import joblib

features = [False, False, True, True, True, True, True, False, False, True, True, False
    , False, False, False, False, False, False, False, False, False, True, False, False,
            False, False, False, True, False, False, False, False, False, True, True, False,
            False, False, False, False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False, False, False, False,
            True, False, False, True, False, True, False, False, False, False, False, False,
            False, False, False, False, False, False, False, False, False, False, False, True,
            False, False, False, False, False, True, True, True, True, False, False]

model = XGBClassifier()
train = pd.read_csv('train.csv', delimiter=",")
train_beats = train.filter(regex="Beat*")
train = train[train.columns.drop(list(train.filter(regex='Beat*')))]

y_train = train['Primary Type']
x_train = train.drop('Primary Type', axis=1)
x_train = x_train.loc[:, features]
x_train = pd.concat([x_train, train_beats], axis=1)
model.fit(x_train, y_train)
print(model)

test = pd.read_csv('test.csv')
test_beats = test.filter(regex="Beat*")
test = test[test.columns.drop(list(test.filter(regex='Beat*')))]

y_test = test['Primary Type']
x_test = test.drop('Primary Type', axis=1)
x_test = x_test.loc[:, features]
x_test = pd.concat([x_test, test_beats], axis=1)

print(model.score(x_test, y_test))
y_pred = model.predict(x_test)
predictions = [round(value) for value in y_pred]

accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

joblib.dump(model, "XGBoost")
