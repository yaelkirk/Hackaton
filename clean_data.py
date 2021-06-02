import pandas as pd
import numpy as np

def split_data(percent, path):
    df = pd.read_csv(path)
    test = df.sample(int(len(df.index)*percent/100))
    train = df[~df.isin(test).all(1)]
    test.to_csv("test.csv")
    train.to_csv("train.csv")


def clean(path):
    df = pd.read_csv(path)
    ### remove id, case_id, lon, lat, location, year
    df = df.drop(['Latitude', 'Longitude', 'Location', 'Year', 'ID','Case Number'], axis=1)
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
    df.drop(["IUCR", "FBI Code","Description"], axis=1)

    ### block
    df["orientation"] = df["Block"].str.slice(6,7)
    orient_dummies = pd.get_dummies(df["orientation"], prefix="orient")
    df = df.drop(["orientation"], axis=1)
    df = df.join(orient_dummies)
    return df

clean("Task2/Dataset_crimes.csv").to_csv("clean_data.csv")


