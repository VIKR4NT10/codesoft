from pathlib import Path
import pandas as pd
import numpy as np
from logger import logging
import yaml


# -------------------------------
# Load preprocessed data
# -------------------------------
def load_preprocessed_data(processed_dir: Path) -> pd.DataFrame:
    files = sorted(processed_dir.glob("preprocessed*.parquet"))
    if not files:
        logging.warning(f"No preprocessed files found in {processed_dir}")
        return pd.DataFrame()

    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    logging.info(f"Loaded {len(df)} rows for feature engineering")
    return df


# -------------------------------
# Haversine distance
# -------------------------------
def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 6371 * (2 * np.arcsin(np.sqrt(a)))


# -------------------------------
# Target-based risk encoding
# -------------------------------
def risk_encode(train_df, test_df, col, target="is_fraud", min_samples=50):
    global_rate = train_df[target].mean()

    stats = (
        train_df.groupby(col)[target]
        .agg(["mean", "count"])
        .rename(columns={"mean": "fraudrate", "count": "n"})
    )

    stats["risk"] = (
        stats["fraudrate"] * stats["n"] + global_rate * min_samples
    ) / (stats["n"] + min_samples)

    train_enc = train_df[col].map(stats["risk"]).fillna(global_rate)
    test_enc = test_df[col].map(stats["risk"]).fillna(global_rate)

    return train_enc, test_enc


# -------------------------------
# Feature creation
# -------------------------------
def create_features(df: pd.DataFrame):
    if df.empty:
        logging.warning("Empty DataFrame. Skipping feature engineering.")
        return None, None

    if "is_fraud" not in df.columns:
        raise ValueError("Missing required target column: is_fraud")

    df = df.sort_values(["cc_num", "trans_date_trans_time"]).reset_index(drop=True)

    # Time features
    df["transaction_hour"] = df["trans_date_trans_time"].dt.hour
    df["transaction_day"] = df["trans_date_trans_time"].dt.day
    df["transaction_month"] = df["trans_date_trans_time"].dt.month
    df["transaction_weekday"] = df["trans_date_trans_time"].dt.weekday
    df["is_weekend"] = df["transaction_weekday"].isin([5, 6]).astype(int)
    df["age"] = (df["trans_date_trans_time"] - df["dob"]).dt.days / 365.25

    # Rolling features (past only)
    df = df.set_index("trans_date_trans_time")

    df["txn_count_1h"] = (
        df.groupby("cc_num")["amt"]
        .rolling("1h", closed="left")
        .count()
        .reset_index(level=0, drop=True)
    )

    df["txn_count_24h"] = (
        df.groupby("cc_num")["amt"]
        .rolling("24h", closed="left")
        .count()
        .reset_index(level=0, drop=True)
    )

    df["amt_mean_24h"] = (
        df.groupby("cc_num")["amt"]
        .rolling("24h", closed="left")
        .mean()
        .reset_index(level=0, drop=True)
    )

    df["amt_std_24h"] = (
        df.groupby("cc_num")["amt"]
        .rolling("24h", closed="left")
        .std()
        .reset_index(level=0, drop=True)
    )

    df["amt_zscore_24h"] = (
        df["amt"] - df["amt_mean_24h"]
    ) / (df["amt_std_24h"] + 1e-6)

    df = df.reset_index()

    # Geo distance
    df["geo_distance_km"] = haversine(
        df["lat"], df["long"], df["merch_lat"], df["merch_long"]
    )

    # Time-based split
    split_time = df["trans_date_trans_time"].quantile(0.8)
    train_df = df[df["trans_date_trans_time"] <= split_time].copy()
    test_df = df[df["trans_date_trans_time"] > split_time].copy()

    logging.info(f"Train rows: {len(train_df)}, Test rows: {len(test_df)}")

    # Risk encoding
    for col in ["merchant", "category", "job"]:
        train_df[f"{col}_risk"], test_df[f"{col}_risk"] = risk_encode(
            train_df, test_df, col
        )

    # Missing handling
    for col in ["amt_mean_24h", "amt_std_24h", "amt_zscore_24h"]:
        median = train_df[col].median()
        for d in [train_df, test_df]:
            d[f"{col}_missing"] = d[col].isna().astype(int)
            d[col] = d[col].fillna(median)

    logging.info(
        f"Feature engineering complete. "
        f"Train shape: {train_df.shape}, Test shape: {test_df.shape}"
    )

    return train_df, test_df


# -------------------------------
# Save features
# -------------------------------
def save_features(train_df, test_df, features_dir: Path):
    features_dir.mkdir(parents=True, exist_ok=True)

    train_path = features_dir / "train_features.parquet"
    test_path = features_dir / "test_features.parquet"

    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)

    logging.info(f"Saved features to {features_dir}")
    return train_path, test_path


# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    try:
        with open("params.yaml") as f:
            config = yaml.safe_load(f)

        MODEL_NAME = "credit_card_fraud"
        model_cfg = config["models"][MODEL_NAME]

        processed_dir = Path(config["paths"]["processed_data"]) / MODEL_NAME
        features_dir = Path(config["paths"]["features_data"]) / MODEL_NAME

        df = load_preprocessed_data(processed_dir)

        if df.empty:
            logging.warning("No data available for feature engineering")
        else:
            train_df, test_df = create_features(df)
            save_features(train_df, test_df, features_dir)

            logging.info("Feature engineering pipeline completed successfully")

    except Exception as e:
        logging.exception(f"Feature engineering failed: {e}")
        raise
