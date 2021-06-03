import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings(action='ignore')


#%%
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


#%%
class LogisticClassifier(object):
    def __init__(self, df):
        y = df[["Primary Type"]]
        self.y = y
        self.models = None
        self.X = self.process_for_logistic(train.drop(["Primary Type"], axis=1))
        self.train(25)

    def process_for_logistic(self, X):
        X = X.drop(["Unnamed: 0", "Unnamed: 0.1"], axis=1)
        X['X Coordinate'] = (X['X Coordinate'] - np.min(X['X Coordinate'])) / \
                            (np.max(X['X Coordinate']) - np.min(X['X Coordinate']))
        X['Y Coordinate'] = (X['Y Coordinate'] - np.min(X['Y Coordinate'])) / \
                            (np.max(X['Y Coordinate']) - np.min(X['Y Coordinate']))
        X = X.drop([c for c in X.columns if c.startswith('Beat_')], axis=1)
        return X

    def get_logistic_regressor(self):
        """
        :return learner, accuracy
        """
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.3)
        clf = LogisticRegression(C=1e2, max_iter=400).fit(X_train, y_train)
        return clf

    def train(self, repeats):
        models = []
        for i in range(repeats):
            model = self.get_logistic_regressor()
            models.append(model)
        self.models = models

    def predict(self, X):
        proc_X = self.process_for_logistic(X)
        prob = np.mean([model.predict_proba(proc_X) for model in self.models], 0)
        predictions = [self.models[0].classes_[np.argmax(q)] for q in prob]
        return predictions, prob


#%%
log_reg = LogisticClassifier(train)
y_test = test["Primary Type"]
prediction, prob = log_reg.predict(test.drop(["Primary Type"], axis=1))
print(np.sum(prediction == y_test) / len(y_test))

#%%
pickle.dump(log_reg, open('logistic.sav', 'wb'))

