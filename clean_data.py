import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point, polygon

location_decription_set = {"APARTMENT",
"RESIDENCE",
"CHA APARTMENT",
"RESIDENCE - GARAGE",
"SMALL RETAIL STORE",
"STREET",
"BARBERSHOP",
"CAR WASH",
"CONVENIENCE STORE",
"VEHICLE NON-COMMERCIAL",
"RESIDENCE - YARD (FRONT / BACK)",
"GAS STATION",
"ATM (AUTOMATIC TELLER MACHINE)",
"COMMERCIAL / BUSINESS OFFICE",
"DEPARTMENT STORE",
"PARK PROPERTY",
"GROCERY FOOD STORE",
"",
"CTA TRAIN",
"RESIDENCE - PORCH / HALLWAY",
"PARKING LOT / GARAGE (NON RESIDENTIAL)",
"SIDEWALK",
"HOTEL / MOTEL",
"RESTAURANT",
"TAVERN / LIQUOR STORE",
"POLICE FACILITY / VEHICLE PARKING LOT",
"OTHER (SPECIFY)",
"DRUG STORE",
"ALLEY",
"CONSTRUCTION SITE",
"VEHICLE - OTHER RIDE SHARE SERVICE (LYFT, UBER, ETC.)",
"CTA PLATFORM",
"AIRPORT TERMINAL UPPER LEVEL - NON-SECURE AREA",
"APPLIANCE STORE",
"KENNEL",
"CTA BUS STOP",
"AIRPORT TERMINAL LOWER LEVEL - NON-SECURE AREA",
"ATHLETIC CLUB",
"DRIVEWAY - RESIDENTIAL",
"NURSING / RETIREMENT HOME",
"VEHICLE - COMMERCIAL",
"CTA BUS",
"LIBRARY",
"MEDICAL / DENTAL OFFICE",
"HOSPITAL BUILDING / GROUNDS",
"CTA STATION",
"WAREHOUSE",
"AIRPORT TERMINAL UPPER LEVEL - SECURE AREA",
"VEHICLE - DELIVERY TRUCK",
"COLLEGE / UNIVERSITY - GROUNDS",
"BANK",
"OTHER RAILROAD PROPERTY / TRAIN DEPOT",
"VACANT LOT / LAND",
"CURRENCY EXCHANGE",
"SCHOOL - PUBLIC BUILDING",
"CLEANING STORE",
"AUTO / BOAT / RV DEALERSHIP",
"GOVERNMENT BUILDING / PROPERTY",
"SCHOOL - PUBLIC GROUNDS",
"BAR OR TAVERN",
"FIRE STATION",
"CHA PARKING LOT / GROUNDS",
"FOREST PRESERVE",
"CHURCH / SYNAGOGUE / PLACE OF WORSHIP",
"HIGHWAY / EXPRESSWAY",
"FACTORY / MANUFACTURING BUILDING",
"CTA PARKING LOT / GARAGE / OTHER PROPERTY",
"AIRPORT EXTERIOR - NON-SECURE AREA",
"AIRPORT BUILDING NON-TERMINAL - SECURE AREA",
"AIRPORT BUILDING NON-TERMINAL - NON-SECURE AREA",
"VEHICLE - COMMERCIAL: ENTERTAINMENT / PARTY BUS",
"AIRPORT PARKING LOT",
"SCHOOL - PRIVATE GROUNDS",
"SCHOOL - PRIVATE BUILDING",
"JAIL / LOCK-UP FACILITY",
"PAWN SHOP",
"AIRPORT TERMINAL LOWER LEVEL - SECURE AREA",
"CEMETARY",
"OTHER COMMERCIAL TRANSPORTATION",
"CHA HALLWAY / STAIRWELL / ELEVATOR",
"NEWSSTAND",
"CTA TRACKS - RIGHT OF WAY",
"SPORTS ARENA / STADIUM",
"ABANDONED BUILDING",
"FEDERAL BUILDING",
"BOWLING ALLEY",
"COIN OPERATED MACHINE",
"DAY CARE CENTER",
"AIRCRAFT",
"BRIDGE",
"MOVIE HOUSE / THEATER",
"AIRPORT EXTERIOR - SECURE AREA",
"AIRPORT VENDING ESTABLISHMENT",
"AIRPORT TRANSPORTATION SYSTEM (ATS)",
"POOL ROOM",
"TAXICAB",
"COLLEGE / UNIVERSITY - RESIDENCE HALL",
"BOAT / WATERCRAFT",
"CREDIT UNION",
"ANIMAL HOSPITAL",
"LAKEFRONT / WATERFRONT / RIVERBANK",
"VEHICLE - COMMERCIAL: TROLLEY BUS"}

classes = {0: 'BATTERY', 1: 'THEFT', 2: 'CRIMINAL DAMAGE', 3: 'DECEPTIVE PRACTICE', 4: 'ASSAULT'}
inv_classes = {v:k for k, v in classes.items()}

def split_data(percent, path):
    df = pd.read_csv(path)
    test = df.sample(int(len(df.index)*percent/100))
    train = df[~df.isin(test).all(1)]
    test.to_csv("test.csv")
    train.to_csv("train.csv")


def clean(path):
    df = pd.read_csv(path)
    df = df.dropna()
    ### district between 1-31

    df = df[df.District.gt(0)&df.District.le(22)] ## TODO: check validity
    ### arrest and domestic to boolean

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
    ### split date and time
    df["Day_of_the_week"] = pd.to_datetime(df['Date']).dt.dayofweek
    df['Day'] = pd.to_datetime(df['Date']).dt.day
    df['Month'] = pd.to_datetime(df['Date']).dt.month
    df['Year'] = pd.to_datetime(df['Date']).dt.year
    df['Time'] = pd.to_datetime(df['Date']).dt.time

    ### add W/E/N/S
    df["orientation"] = df["Block"].str.slice(6,7)
    orient_dummies = pd.get_dummies(df["orientation"], prefix="orient")
    df = df.drop(["orientation"], axis=1)
    df = df.join(orient_dummies)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    chicago = gpd.read_file("geo_export_8ec36d6b-42d6-4d79-ba5f-fa35870e87a0.shp").geometry.unary_union
    crs = {'init': 'epsg:4326', 'no_defs': True}
    geometry = [Point(xy) for xy in zip(df["Longitude"], df["Latitude"])]
    geo_df = gpd.GeoDataFrame(df, crs=crs, geometry=geometry)
    geo_df = geo_df[geo_df.geometry.within(chicago)]
    ### location description dummies
    for desc in location_decription_set:
        geo_df[desc] = (desc == geo_df["Location Description"]).astype(int)
    ### drop illeagal and irrelevant fields
    geo_df["Primary Type"] = geo_df["Primary Type"].replace(inv_classes)
    geo_df = additional_preprocessing(geo_df)
    geo_df = geo_df.drop(["IUCR", "FBI Code","Description","Date", "Year","ID","Case Number", "Block", "geometry", "Location Description", "Updated On", "Longitude","Latitude", "Location", "Day","Month"], axis=1)
    return geo_df


def additional_preprocessing(df):
    df["Day_of_the_week1"] = np.sin(df["Day_of_the_week"] * 2 / 7 * np.pi)
    df["Day_of_the_week2"] = np.cos(df["Day_of_the_week"] * 2 / 7 * np.pi)
    df = df.drop(["Day_of_the_week"], axis=1)

    df["Time"] = pd.to_timedelta(df["Time"].astype("str")) / pd.offsets.Minute(1)
    df["Time1"] = np.sin(df["Time"] * 2 / 1440 * np.pi)
    df["Time2"] = np.cos(df["Time"] * 2 / 1440 * np.pi)
    df = df.drop(["Time"], axis=1)
    
    df["Month1"] = np.sin(df["Month"] * 2 / 12 * np.pi)
    df["Month2"] = np.cos(df["Month"] * 2 / 12 * np.pi)

    return df

clean("Task2/Dataset_crimes.csv").to_csv("clean_data.csv")
split_data(15, "clean_data.csv")
def plot_orientation():
    beats_map = gpd.read_file("geo_export_8ec36d6b-42d6-4d79-ba5f-fa35870e87a0.shp")
    fig, ax = plt.subplots(figsize=(15,15))
    beats_map.plot(ax=ax, alpha=0.8, color='gray')
    crs = {'init': 'epsg:4326', 'no_defs': True}
    proj = pd.read_csv("train.csv")[["Longitude", "Latitude", "orient_N", "orient_W", "orient_E", "orient_S"]]
    geometry = [Point(xy) for xy in zip(proj["Longitude"], proj["Latitude"])]
    print(geometry)
    geo_proj = gpd.GeoDataFrame(proj, crs=crs, geometry=geometry)
    geo_proj[geo_proj["orient_N"] == 1].plot(ax=ax, markersize=20,color="yellow", marker='^', label="North")
    geo_proj[geo_proj["orient_S"] == 1].plot(ax=ax, markersize=20,color="red", marker='^', label="South")
    geo_proj[geo_proj["orient_W"] == 1].plot(ax=ax, markersize=20,color="green", marker='o', label="West")
    geo_proj[geo_proj["orient_E"] == 1].plot(ax=ax, markersize=20,color="orange", marker='o', label="East")
    plt.show()

