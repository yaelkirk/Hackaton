import pickle
import datetime
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def from_datetime(date):
    dt = pd.to_datetime(date)
    day_of_week = dt.dayofweek
    month = dt.month

    day_of_week1 = np.sin(day_of_week * 2 / 7 * np.pi)
    day_of_week2 = np.cos(day_of_week * 2 / 7 * np.pi)
    month1 = np.sin(month * 2 / 12 * np.pi)
    month2 = np.cos(month * 2 / 12 * np.pi)

    return day_of_week1, day_of_week2, month1, month2


def get_closest(df, day_of_week1, day_of_week2, month1, month2):
    df["dist"] = (df["Day_of_the_week1"] - day_of_week1) ** 2 + \
                 (df["Day_of_the_week2"] - day_of_week2) ** 2 + \
                 (df["Month1"] - month1) ** 2 + \
                 (df["Month2"] - month2) ** 2
    df = df[df.dist >= df.quantile(0.9)['dist']]
    return df[["X Coordinate", "Y Coordinate", "Time1", "Time2"]]


def get_thirty_center(date, df):
    day_of_week1, day_of_week2, month1, month2 = from_datetime(date)
    closest_points = get_closest(df, day_of_week1, day_of_week2, month1, month2)
    kmeans = KMeans(n_clusters=30, random_state=0).fit(closest_points)
    return kmeans.cluster_centers_


def parse_point(center, minX, maxX, minY, maxY, date):
    x = float(int((center[0] + 1) * maxX / 2 + minX))
    y = float(int((center[1] + 1) * maxY / 2 + minY))
    dt = pd.to_datetime(date)
    year = dt.year
    month = dt.month
    day = dt.day

    t = np.arctan(center[2] / center[3])
    if t < 0:
        t = t + 2 * np.pi
    t = t / 2 * 1440 / np.pi
    new_date = str(datetime.datetime(year, month, day, int(t / 60), int(t % 60)))
    return x, y, new_date


def send_police_cars(date):
    crimedf = pickle.load(open('crimedb.sav', 'rb'))
    minX = np.min(crimedf["X Coordinate"])
    maxX = np.max(crimedf["X Coordinate"])
    minY = np.min(crimedf["Y Coordinate"])
    maxY = np.max(crimedf["Y Coordinate"])
    crimedf["X Coordinate"] = (crimedf["X Coordinate"] - minX) * 2 / maxX - 1
    crimedf["Y Coordinate"] = (crimedf["Y Coordinate"] - minY) * 2 / maxY - 1

    centers = get_thirty_center(date, crimedf)
    return [parse_point(center, minX, maxX, minY, maxY, date) for center in centers]

