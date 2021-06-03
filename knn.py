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

# #%% show data
# df.columns

#%% take what we need from the data
Y = df["Primary Type"]
x = df["X Coordinate"]
y = df["Y Coordinate"]
x = x/max(x) * 10
y = y/max(y) * 10
xy_data = pd.concat([x,y, Y], axis=1).dropna().replace(number_dict)

Y = xy_data["Primary Type"]

#%% to numpy
X = xy_data.to_numpy()
Y = Y.to_numpy()

#%% parameters
neighbors = [5,10,15,20,30,50]

n_neighbors = 15
#%% run KNN
clf = KNeighborsClassifier(n_neighbors)




#%% predict

clf.fit(X, Y)
h=.2
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])




#%% Plot
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())

plt.show()
