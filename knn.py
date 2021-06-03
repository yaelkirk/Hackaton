#%%
import numpy as np
# from utils import *
# import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
crimes_dict = {0: 'BATTERY', 1: 'THEFT', 2: 'CRIMINAL DAMAGE', 3: 'DECEPTIVE PRACTICE', 4: 'ASSAULT'}
number_dict = {'BATTERY' : 0, 'THEFT' : 1, 'CRIMINAL DAMAGE': 2,
               'DECEPTIVE PRACTICE' : 3, 'ASSAULT' : 4 }

#%% clean data
# data = clean("train.csv")
df = pd.read_csv("train.csv")

#%% Split
test = df.sample(int(len(df.index)*33/100))
train = df[~df.isin(test).all(1)]

#%% take what we need from the data, a little cleanup


def local_cleanup(dataframe):
    """
    Removes Nans and unneeded columns and replaces strings in Y to ints
    :param dataframe: pandas DF
    :return: X,Y as np arrays
    """
    Y = dataframe["Primary Type"]
    x = dataframe["X Coordinate"]
    y = dataframe["Y Coordinate"]
    x = x/max(x) * 10
    y = y/max(y) * 10
    xy_data = pd.concat([x,y, Y], axis=1).dropna().replace(number_dict)
    Y = xy_data["Primary Type"]
    x = xy_data["X Coordinate"]
    y = xy_data["Y Coordinate"]
    xy_data = pd.concat([x,y], axis=1)
    X = xy_data.to_numpy()
    Y = Y.to_numpy()
    X = X * X * X  # Bloat for visualization
    return X,Y


#%% to numpy
X,y = local_cleanup(df)
X_train, Y_train = local_cleanup(train)
X_test, Y_test = local_cleanup(test)


#%% parameters
neighbors = [5,10,15,20,30,50]

n_neighbors = 50
#%% run KNN
knn = KNeighborsClassifier(n_neighbors)

#%% predict

knn.fit(X_train, Y_train)
h=.2
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

#%% Plot
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])

#%% plot
# # Put the result into a color plot
# # Z = Z.reshape(xx.shape)
# plt.figure()
# plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
#
# # Plot also the training points
# plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
# plt.xlim(xx.min(), xx.max())
# plt.ylim(yy.min(), yy.max())
#
# plt.show()


knn.fit(X, Y)

# Plot the decision boundary. For that, we will asign a color to each
# point in the mesh [x_min, m_max]x[y_min, y_max].
x_min, x_max = X[:,0].min() - .5, X[:,0].max() + .5
y_min, y_max = X[:,1].min() - .5, X[:,1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1, figsize=(4, 3))
plt.set_cmap(plt.cm.Paired)
plt.pcolormesh(xx, yy, Z)

# Plot also the training points
plt.scatter(X[:,0], X[:,1],c=Y )
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())

plt.show()

#%% Test performance
Y_hat = knn.predict(X_test)
precision = np.sum(Y_test == Y_hat)/len(Y_hat)
