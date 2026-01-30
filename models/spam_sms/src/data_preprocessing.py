import re
import yaml
import logging
import pandas as pd

from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from logger import logging

logging.basicConfig(level=logging.INFO)


# -------------------------------
# Load raw data
# -------------------------------
def load_raw_data(raw_dir: Path) -> pd.DataFrame: 
    """
    Expects raw data saved as CSV from data_ingestion step
    """
    csv_files = list(raw_dir.glob("*.csv"))

    if not csv_files:
        logging.warning("No raw CSV files found in %s", raw_dir)
        return pd.DataFrame()

    df_list = []
    for f in csv_files:
        try:
            df_list.append(pd.read_csv(f, encoding="latin-1"))
            logging.info(f"Loaded {f}")
        except UnicodeDecodeError:
            logging.error(f"Failed to read {f} due to encoding issues")
            raise

    return pd.concat(df_list, ignore_index=True)

# -------------------------------
# Text cleaning
# -------------------------------
def clean_text(text: str) -> str:
    text = re.sub(r"http\S+|www\S+", " URL ", text)
    text = re.sub(r"\S+@\S+", " EMAIL ", text)
    text = re.sub(r"\d+", " NUMBER ", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text


# -------------------------------
# Preprocess data
# -------------------------------
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("Starting SMS spam preprocessing")
    df = df[["v1", "v2"]]
    df = df.rename(columns={"v1": "label", "v2": "text"})
    df = df.dropna(subset=["label", "text"])

    # lowercase
    df["text"] = df["text"].str.lower()

    # clean text
    df["text"] = df["text"].apply(clean_text)

    # normalize whitespace
    df["text"] = (
        df["text"]
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    # encode labels
    le = LabelEncoder()
    df["label"] = le.fit_transform(df["label"])

    logging.info("Preprocessing completed")
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

        # MODEL_NAME = "spam_sms"
        # model_cfg = config["models"][MODEL_NAME]

        raw_dir = Path("data/spam_sms/raw")
        processed_dir = Path("data/spam_sms/processed")

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

