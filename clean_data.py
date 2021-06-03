import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point, polygon

def split_data(percent, path):
    df = pd.read_csv(path)
    test = df.sample(int(len(df.index)*percent/100))
    train = df[~df.isin(test).all(1)]
    test.to_csv("test.csv")
    train.to_csv("train.csv")


def clean(path):
    df = pd.read_csv(path)
    ### remove id, case_id, lon, lat, location, year
    df = df.drop(['Location', 'Year', 'ID','Case Number'], axis=1)
    df = df.dropna()
    ### district between 1-31
    df = df[df.District.gt(0)&df.District.le(22)] ## TODO: check validity
    ### arrest and domestic to boolean
    #df = df.replace(to_replace=["TRUE", "FALSE"], value=[1, 0])
    df["Arrest"] = df["Arrest"].astype(int)
    df["Domestic"] = df["Domestic"].astype(int)
    df = df[df.Arrest.isin([0,1])&df.Domestic.isin([0,1])]
    ### beats to dummies
    df = df[df.Beat.ge(111)&df.Beat.le(2535)]
    missing_beats = set(np.arange(111,2535))
    dummies_beats = pd.get_dummies(df["Beat"], prefix='Beat_')
    cols = dummies_beats.columns
    for miss in missing_beats:
        if "Beat_"+str(miss) not in cols:
            dummies_beats["Beat_"+str(miss)] = 0
    df = df.drop(["Beat"], axis=1)
    df = df.join(dummies_beats)
    ### community area between 1-31
    df = df[df["Community Area"].gt(0)&df["Community Area"].le(77)]
    ### Ward between 1-50
    df = df[df.Ward.gt(0)&df.Ward.le(50)]
    ### TODO : x y validation
    ### split date and time
    df["Day_of_the_week"] = pd.to_datetime(df['Date']).dt.dayofweek
    df['Day'] = pd.to_datetime(df['Date']).dt.day
    df['Month'] = pd.to_datetime(df['Date']).dt.month
    df['Year'] = pd.to_datetime(df['Date']).dt.year
    df['Time'] = pd.to_datetime(df['Date']).dt.time


    ### drop illeagal fields
    df = df.drop(["IUCR", "FBI Code","Description","Date","Unnamed: 0"], axis=1)

    ### block
    df["orientation"] = df["Block"].str.slice(6,7)
    orient_dummies = pd.get_dummies(df["orientation"], prefix="orient")
    df = df.drop(["orientation"], axis=1)
    df = df.join(orient_dummies)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df = additional_preprocessing(df)
    return df

def additional_preprocessing(df):
    df["Day_of_the_week1"] = np.sin(df["Day_of_the_week"] * 2 / 7 * np.pi)
    df["Day_of_the_week2"] = np.cos(df["Day_of_the_week"] * 2 / 7 * np.pi)
    df = df.drop(["Day_of_the_week"], axis=1)

    df["Time"] = pd.to_timedelta(df["Time"]) / pd.offsets.Minute(1)
    df["Time1"] = np.sin(df["Time"] * 2 / 1440 * np.pi)
    df["Time2"] = np.cos(df["Time"] * 2 / 1440 * np.pi)
    df = df.drop(["Time"], axis=1)
    return df

clean("Task2/Dataset_crimes.csv").to_csv("clean_data.csv")
split_data(15, "clean_data.csv")

# beats_map = gpd.read_file("geo_export_8ec36d6b-42d6-4d79-ba5f-fa35870e87a0.shp")
# fig, ax = plt.subplots(figsize=(15,15))
# beats_map.plot(ax=ax, alpha=0.8, color='gray')
# crs = {'init': 'epsg:4326', 'no_defs': True}
# proj = pd.read_csv("train.csv")[["Longitude", "Latitude", "orient_N", "orient_W", "orient_E", "orient_S"]]
# geometry = [Point(xy) for xy in zip(proj["Longitude"], proj["Latitude"])]
# geo_proj = gpd.GeoDataFrame(proj, crs=crs, geometry=geometry)
# geo_proj[geo_proj["orient_N"] == 1].plot(ax=ax, markersize=20,color="yellow", marker='^', label="North")
# geo_proj[geo_proj["orient_S"] == 1].plot(ax=ax, markersize=20,color="red", marker='^', label="South")
# geo_proj[geo_proj["orient_W"] == 1].plot(ax=ax, markersize=20,color="green", marker='o', label="West")
# geo_proj[geo_proj["orient_E"] == 1].plot(ax=ax, markersize=20,color="orange", marker='o', label="East")

# plt.show()
