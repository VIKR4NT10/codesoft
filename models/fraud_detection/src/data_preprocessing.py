from pathlib import Path
import pandas as pd
import yaml
from logger import logging


# -------------------------------
# Load raw parquet files
# -------------------------------
def load_raw_data(raw_dir: Path) -> pd.DataFrame:
    files = sorted(raw_dir.glob("transactions_*.parquet"))
    if not files:
        logging.warning(f"No raw files found in {raw_dir}")
        return pd.DataFrame()

    logging.info(f"Loading {len(files)} raw files from {raw_dir}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    logging.info(f"Loaded {len(df)} total rows")
    return df


# -------------------------------
# Preprocessing
# -------------------------------
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        logging.warning("Input DataFrame is empty, skipping preprocessing")
        return df

    for col in ["trans_date_trans_time", "dob"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    critical_cols = ["cc_num", "amt", "trans_date_trans_time"]
    before = len(df)
    df = df.dropna(subset=critical_cols)
    logging.info(f"Dropped {before - len(df)} rows with missing critical values")

    if "gender" in df.columns:
        df["gender"] = (
            df["gender"]
            .str.upper()
            .map({"M": 1, "F": 0})
            .fillna(-1)
            .astype(int)
        )

    numeric_cols = ["amt", "city_pop", "lat", "long", "merch_lat", "merch_long"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    logging.info(f"Preprocessed DataFrame shape: {df.shape}")
    return df


# -------------------------------
# Save preprocessed data
# -------------------------------
def save_preprocessed(
    df: pd.DataFrame,
    processed_dir: Path,
    filename: str = "preprocessed.parquet",
) -> Path:
    processed_dir.mkdir(parents=True, exist_ok=True)
    path = processed_dir / filename
    df.to_parquet(path, index=False)
    logging.info(f"Saved preprocessed data to {path}")
    return path


# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    try:
        with open("params.yaml") as f:
            config = yaml.safe_load(f)

        MODEL_NAME = "fraud_detection"
        model_cfg = config["models"][MODEL_NAME]

        raw_dir = Path("data/fraud_detection/raw") 
        processed_dir = Path("data/fraud_detection/processed")

        df_raw = load_raw_data(raw_dir)

        if df_raw.empty:
            logging.warning("No raw data to preprocess. Exiting.")
        else:
            df_processed = preprocess(df_raw)
            save_preprocessed(df_processed, processed_dir)
            logging.info("Data preprocessing completed successfully")

    except Exception as e:
        logging.exception(f"Data preprocessing failed: {e}")
        raise
