import pandas as pd
import numpy as np
import sklearn.tree
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.feature_selection import SelectKBest, chi2
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import joblib


def load_data(path):
    """
    read data from csv file and preprocess it before calculating LR
    :param path: path of csv file
    :return: DataFrame after processing
    """
    df = pd.read_csv(path, delimiter=",")
    df['Time1'] = df['Time1'] + 1
    df['Time2'] = df['Time2'] + 1
    df['Day_of_the_week1'] = df['Day_of_the_week1'] + 1
    df['Day_of_the_week2'] = df['Day_of_the_week2'] + 1
    df['Month1'] = df['Month1'] + 1
    df['Month2'] = df['Month2'] + 1
    # df_beats = df.filter(regex="Beat*")
    df = df[df.columns.drop(list(df.filter(regex='Beat*')))]
    return df


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


def decisionTree():
    train = load_data('train.csv')
    y_train = train['Primary Type']
    x_train = train.drop('Primary Type', axis=1)
    test = load_data('test.csv')
    x_test = test.drop('Primary Type', axis=1)
    y_test = test['Primary Type']
    clf = DecisionTreeClassifier(max_depth=10)
    x_train = x_train.loc[:, features]
    # x_train = pd.concat([x_train, train_beats], axis=1)
    clf.fit(x_train, y_train)
    x_test = x_test.loc[:, features]
    res = clf.score(x_test, y_test)
    joblib.dump(clf, "DecisionTree")
    select_k_best_features(x_train,y_train,x_test,y_test)
    print(res)


def select_k_best_features(x_train, y_train, x_test, y_test):
    features_of_features = list()
    number_of_features = list()
    accuracy = list()
    tree_depth = list()
    for i in range(1, x_train.shape[1]):
        for j in range(1, 50, 3):
            clf = DecisionTreeClassifier(max_depth=j)
            fs = SelectKBest(score_func=chi2, k=i)
            x_reduced_train = fs.fit(x_train, y_train).fit_transform(x_train, y_train)
            clf.fit(x_reduced_train,  y_train)
            a = fs.get_support()
            features_of_features.append(a)
            new_series = pd.Series(a)
            b = new_series.values
            new_x = x_test.loc[:, b]
            res = clf.score(new_x, y_test)
            number_of_features.append(i)
            tree_depth.append(j)
            accuracy.append(res)
    print(accuracy)
    max_accuracy = max(accuracy)
    print('max accuracy:')
    print(max_accuracy)
    print('tree depth:')
    max_index = accuracy.index(max_accuracy)
    print(tree_depth[max_index])
    print('number of features:')
    print(number_of_features[max_index])
    print(features_of_features[max_index])

    plt.title('Accuracy as function of tree depth and K features')
    ax = plt.axes(projection='3d')

    # Data for three-dimensional scattered points
    zdata = accuracy
    xdata = tree_depth
    ydata = number_of_features
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens')
    ax.set_xlabel('tree depth')
    ax.set_ylabel('number of features')
    ax.set_zlabel('accuracy')
    plt.show()

decisionTree()

