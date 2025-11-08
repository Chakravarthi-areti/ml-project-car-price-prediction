# prediction_helper.py
import pandas as pd
import numpy as np
from joblib import load

# Load saved artifacts
best_model = load("artifacts/car_price_model.joblib")
scaler = load("artifacts/car_scaler.joblib")
te = load("artifacts/target_encoder.joblib")
cols_to_scale = load("artifacts/cols_to_scale.joblib")

# Owner mapping
owner_map = {
    'First': 1,
    'Second': 2,
    'Third': 3,
    'Fourth': 4,
    'UnRegistered Car': 5,
    '4 or More': 6
}

def preprocess_input(input_dict):
    """
    Converts input dictionary into preprocessed dataframe suitable for the model.
    Handles missing columns, feature engineering, scaling, and encoding.
    """
    # Convert dict to DataFrame
    df = pd.DataFrame([input_dict])

    # Ensure all expected numeric columns exist
    numeric_cols_defaults = {
        'year': 2018,
        'kilometer': 0,
        'length': 0,
        'width': 0,
        'height': 0,
        'seating_capacity': 0,
        'fuel_tank_capacity': 0,
        'car_age': 0,
        'max_torque_in_nm': 0,
        'max_power_in_bhp': 0,
        'engine_cc': 0
    }

    for col, default in numeric_cols_defaults.items():
        if col not in df.columns:
            df[col] = default

    # Feature Engineering
    df['engine_cc'] = df['engine_cc'].replace(0, np.nan).fillna(df['engine_cc'].median())
    df['power_per_cc'] = df['max_power_in_bhp'] / np.maximum(df['engine_cc'], 300)
    df['torque_per_cc'] = df['max_torque_in_nm'] / np.maximum(df['engine_cc'], 300)
    df['car_volume_ltrs'] = (df['length'] * df['width'] * df['height']) / 1000000
    df['power_torque_ratio'] = df['power_per_cc'] / (df['torque_per_cc'] + 1e-5)
    df['engine_seating_ratio'] = df['engine_cc'] / (df['seating_capacity'] + 1e-5)
    df['age_squared'] = df['car_age'] ** 2

    # Owner mapping
    if 'owner' not in df.columns:
        df['owner'] = 0
    df['owner'] = df['owner'].map(owner_map).fillna(0).astype(int)

    # Transmission mapping
    if 'transmission' not in df.columns:
        df['transmission'] = 0
    df['transmission'] = df['transmission'].map({'Manual': 0, 'Automatic': 1}).fillna(0).astype(int)

    # Target Encoding for make, model, location
    te_cols = ['make', 'model', 'location']
    for col in te_cols:
        if col not in df.columns:
            df[col] = 'Unknown'
    df[te_cols] = te.transform(df[te_cols])

    # Scale numeric columns
    for col in cols_to_scale:
        if col not in df.columns:
            df[col] = 0
    df[cols_to_scale] = scaler.transform(df[cols_to_scale])

    # Drop any extra columns that the model doesn't expect
    model_features = best_model.feature_names_in_
    df = df.reindex(columns=model_features, fill_value=0)

    return df


def predict_car_price(input_dict):
    """
    Preprocess input and predict car price using the trained model.
    Returns the price formatted in Indian style (lakhs, thousands, etc.)
    """
    df = preprocess_input(input_dict)
    log_price = best_model.predict(df)[0]
    price = np.expm1(log_price)  # convert log1p back to original

    # Format price in Indian numbering system
    def format_inr(number):
        if number >= 1_00_00_000:  # crore
            return f"₹ {number/1_00_00_000:.2f} Cr"
        elif number >= 1_00_000:  # lakh
            return f"₹ {number/1_00_000:.2f} L"
        elif number >= 1_000:     # thousand
            return f"₹ {number/1_000:.2f} K"
        else:
            return f"₹ {number:.0f}"

    return format_inr(price)

