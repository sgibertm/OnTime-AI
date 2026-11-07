import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# Nota: Random forest is a binary classification problem (0 -> not delayed, 1 -> delayed), while linear regression predicts the continuous delay in minutes.

# Use the plain CSV if present, otherwise the compressed copy shipped in the repo
DATA_PATH = "Files/transport_dataset.csv" if os.path.exists("Files/transport_dataset.csv") else "Files/transport_dataset.csv.gz"

DELAY_THRESHOLD = 0.2   # minuts → if delay > 0.2, is "delayed" (1), if delay < 0.2 (0)

TIME_OF_DAY = {
    "night_early": 0,
    "night":       1,
    "morning_peak":2,
    "midday":      3,
    "afternoon":   4,
    "evening_peak":5,
    "late_night":  6,
}

COLUMNS = [ 
    "route_short_name",
    "direction_id",
    "stop_name",
    "stop_sequence",
    "arrival_time",	
    "departure_time",
    "arrival_sec",	
    "departure_sec",	
    "dwell_time_sec",
    "hour_of_day",
    "time_of_day",
    "stops_in_trip",
    "stop_rank_normalized",
    "dist_to_prev_stop_m",
    "travel_time_to_stop_sec",
    "avg_speed_kmh",
    "route_color_hex",
    
]

numeric_cols = [
    "direction_id",
    "stop_sequence",
    "dwell_time_sec",
    "hour_of_day",
    "time_of_day",
    "stops_in_trip",
    "dist_to_prev_stop_m",
    "travel_time_to_stop_sec",
    "avg_speed_kmh",
    "delay_minutes",
    "is_delayed"
]

# Features used by the Random Forest classifier (These are input features)
RF_FEATURES = [
    "direction_id",
    "stop_sequence",
    "dwell_time_sec",
    "hour_of_day",
    "time_of_day",
    "stops_in_trip",
    "dist_to_prev_stop_m",
    "travel_time_to_stop_sec",
    "avg_speed_kmh",
    "stop_rank_normalized",
]

DAYS = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]



def load_data() -> pd.DataFrame:

    transport = pd.read_csv(DATA_PATH, usecols=COLUMNS)

    print(transport.isnull().sum())

    # Data cleaning
    transport['time_of_day'] = transport['time_of_day'].map(TIME_OF_DAY).fillna(3).astype(int)
    transport['avg_speed_kmh'] = transport['avg_speed_kmh'].fillna(transport['avg_speed_kmh'].median())

    transport['route_short_name'] = transport['route_short_name'].astype(str)

    print(transport.isnull().sum())

    # Converts into a continuous variable (delay in minutes)
    transport["delay_minutes"] = (transport["arrival_sec"] - transport["departure_sec"]) / 60

    
    # Converts into a binary variable (delayed or not)
    transport["is_delayed"] = (transport["delay_minutes"] > DELAY_THRESHOLD).astype(int)


    return transport


def build_routes(df: pd.DataFrame) -> dict:
    routes = {}

    for (route_name, direction), group in df.groupby(["route_short_name", "direction_id"]):

        group = group.sort_values("stop_sequence")

        group = group.drop_duplicates(subset="stop_sequence")

        stops_ordered = group["stop_name"].tolist()

        name_first_stop = stops_ordered[0]
        name_last_stop = stops_ordered[-1]


        key = f"{route_name} - {name_first_stop} - {name_last_stop}"

        routes[key] = {
            "route": route_name,
            "direction": direction,
            "first stop": name_first_stop,
            "last stop": name_last_stop,
            "stops": stops_ordered,
            "color": group["route_color_hex"].iloc[0] if "route_color_hex" in group else None,
            "stops_in_trip": group["stops_in_trip"].iloc[0] if "stops_in_trip" in group else None
        }

    return routes


"""
def run_prediction(subset, full_data):
    
    feature_col = "stop_rank_normalized"
    target_col = "delay_minutes"

   
    X_train = full_data[[feature_col]]
    y_train = full_data[target_col]

    model = linear_model.LinearRegression()
    model.fit(X_train, y_train)

    
    rank = subset[feature_col].mean()
    pred = model.predict([[rank]])[0]

    
    x_line = np.linspace(full_data[feature_col].min(), full_data[feature_col].max(), 100)

    y_line = model.predict(x_line.reshape(-1, 1))

    return pred, x_line, y_line, rank, model
"""

def run_prediction(subset, full_data):

    # Same features and target as Random Forest for consistency
    X = full_data[RF_FEATURES].fillna(0)
    y = full_data["travel_time_to_stop_sec"]

    model = linear_model.LinearRegression()
    model.fit(X, y)

    # Cross-validation MAE (mirrors Random Forest reporting)
    scores = cross_val_score(model, X, y, cv=3, scoring="neg_mean_absolute_error")
    mae = -scores.mean()

    # Predict using the subset's mean feature values (same logic as RF)
    if subset.empty:
        X_pred = X.mean().values.reshape(1, -1)
    else:
        X_pred = subset[RF_FEATURES].fillna(0).mean().values.reshape(1, -1)

    pred_seconds = model.predict(X_pred)[0]

    # Coefficients as a feature-importance analogue for the chart
    feature_coefficients = dict(zip(RF_FEATURES, np.abs(model.coef_)))

    return pred_seconds, feature_coefficients, model, mae

# Steps:
# 1. Converts into yes/no classification (is delayed)
# 2. Builds 100 decision trees
# 3. Estimates “How long will it take to reach this stop?”
def run_random_forest_regressor(subset, full_data):

    # Features
    X = full_data[RF_FEATURES].fillna(0)

    # Target: travel time (seconds)
    y = full_data["travel_time_to_stop_sec"]

    # Train the random forest
    # Builds 100 decision trees
    model = RandomForestRegressor(
        n_estimators=30,
        random_state=42
    )

    model.fit(X, y)

    # Cross-validation
    # Split data into 5 parts
    scores = cross_val_score(model, X, y, cv=3, scoring="neg_mean_absolute_error")
    mae = -scores.mean()

    # Prediction input (same logic as before)
    if subset.empty:
        X_pred = X.mean().values.reshape(1, -1)
    else:
        X_pred = subset[RF_FEATURES].fillna(0).mean().values.reshape(1, -1)

    pred_seconds = model.predict(X_pred)[0]

    feature_importances = dict(zip(RF_FEATURES, model.feature_importances_))

    return pred_seconds, feature_importances, model, mae