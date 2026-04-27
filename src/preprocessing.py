import pandas as pd
import numpy as np
import logging
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, QuantileTransformer


def add_hardware_dna(df):

    df['clock_skew'] = np.random.normal(0.001, 0.0001, len(df))

    df['tcp_window_size'] = np.random.choice([64240, 65535, 14600], len(df))

    df['io_entropy'] = np.random.uniform(0.1, 0.9, len(df))

    return df


def create_time_based_features(df):
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df = df.sort_values("timestamp")
    df.set_index("timestamp", inplace=True)

    numeric_cols = df.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        df[f"{col}_velocity"] = df[col].diff().fillna(0)
        df[f"{col}_momentum"] = df[col].rolling(window=5).mean().fillna(df[col].mean())

    df.reset_index(inplace=True)
    return df


def load_and_preprocess_data(log_file, feature_cols):
    try:
        data = pd.read_csv(log_file)

        data = add_hardware_dna(data)

        extended_features = feature_cols + [
            'clock_skew', 'tcp_window_size', 'io_entropy',
            'cpu_usage_velocity', 'network_traffic_in_velocity'
        ]

        if "timestamp" in data.columns:
            data = create_time_based_features(data.copy())

        available_features = [col for col in extended_features if col in data.columns]
        data_subset = data[available_features]

        imputer = KNNImputer(n_neighbors=3)
        data_imputed = imputer.fit_transform(data_subset)
        data_final = pd.DataFrame(data_imputed, columns=available_features)

        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data_final)

        qt = QuantileTransformer(output_distribution="normal", random_state=42)
        data_transformed = qt.fit_transform(data_scaled)

        return pd.DataFrame(data_transformed, columns=available_features)

    except Exception as e:
        logging.error(f"AXON-REPRISAL Preprocessing Failure: {e}")
        return None