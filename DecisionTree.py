import pandas as pd
import numpy as np
import sklearn.tree
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier, tree
from sklearn.feature_selection import SelectKBest, chi2
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

def load_data(path):
    """
    read data from csv file and preprocess it before calculating LR
    :param path: path of csv file
    :return: DataFrame after processing
    """
    df = pd.read_csv(path, delimiter=",")
    df = df.drop('Unnamed: 0', axis=1)
    df = df.drop('Unnamed: 0.1', axis=1)
    df = df.drop('Location Description', axis=1)
    df = df.drop('Block', axis=1)
    df = df.drop('Updated On', axis=1)
    df["Arrest"] = df["Arrest"].astype(float)
    df["Domestic"] = df["Domestic"].astype(int)
    df["Day_of_the_week"] = pd.to_datetime(df['Date']).dt.dayofweek
    df['Day'] = pd.to_datetime(df['Date']).dt.day
    df['Month'] = pd.to_datetime(df['Date']).dt.month
    df['Year'] = pd.to_datetime(df['Date']).dt.year
    df['Time'] = pd.to_datetime(df['Date']).dt.time
    df[['h.m']] = pd.DataFrame([(x.hour + x.minute / 60) for x in df['Time']])
    df = df.drop('Date', axis=1)
    df = df.drop('Time', axis=1)
    df = df.dropna()
    return df


if __name__ == '__main__':
    train = load_data('train.csv')
    y_train = train['Primary Type']
    classification_values = {0: 'BATTERY', 1: 'THEFT', 2: 'CRIMINAL DAMAGE', 3: 'DECEPTIVE PRACTICE', 4: 'ASSAULT'}
    invert_dict = {v: k for k, v in classification_values.items()}
    y_train = y_train.replace(invert_dict)
    x_train = train.drop('Primary Type', axis=1)
    number_of_features = list()
    accuracy = list()
    tree_depth = list()
    test = load_data('test.csv')
    x_test = test.drop('Primary Type', axis=1)
    y_test = test['Primary Type']
    y_test = y_test.replace(invert_dict)
    for i in range(10, x_train.shape[1], 100):
        for j in range(10, 150, 10):
            clf = DecisionTreeClassifier(max_depth=j)
            fs = SelectKBest(score_func=chi2, k=i)
            x_reduced_train = fs.fit(x_train, y_train).fit_transform(x_train, y_train)
            clf.fit(x_reduced_train, y_train)
            a = fs.get_support()
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
