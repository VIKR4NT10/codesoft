import yaml
import logging
import pandas as pd
from logger import logging
from connections.mongo_connection import MongoDBClient
from pathlib import Path

logging.basicConfig(level=logging.INFO)


# ----------------------------
# Load parameters
# ----------------------------
def load_params(path: str = "params.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ----------------------------
# Fetch data from MongoDB Atlas
# ----------------------------
def fetch_data_from_mongodb(params_path: str = "params.yaml") -> pd.DataFrame:
    params = load_params(params_path)

    collection_name = (
        params["models"]["spam_sms"]["data_ingestion"]["collection_name"]
    )

    mongo_client = MongoDBClient(params_path)
    collection = mongo_client.get_collection(collection_name)

    data = list(collection.find({}, {"_id": 0}))
    if not data:
        raise ValueError("No data found in MongoDB collection")

    logging.info("Fetched %d SMS records from MongoDB", len(data))
    return pd.DataFrame(data)


# ----------------------------
# Ingest data
# ----------------------------
def ingest_data(params_path: str = "params.yaml", output_path: str = "data/spam_sms/raw/sms_spam.csv") -> pd.DataFrame:
    
    logging.info("Starting data ingestion for SMS spam classification")

    df = fetch_data_from_mongodb(params_path)

    # Validate required columns
    required_cols = {"v1", "v2"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise KeyError(f"Missing required columns in data: {missing_cols}")

    # Keep only needed columns and drop NaNs
    df = df[list(required_cols)].dropna()
    # Ensure output folder exists
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Save CSV with latin-1 encoding
    df.to_csv(output_file, index=False, encoding="latin-1")
    logging.info("Data saved to %s with encoding='latin-1'", output_file)

    logging.info("Data ingestion completed. %d rows ready.", len(df))
    return df


# ----------------------------
# Main (for testing)
# ----------------------------
if __name__ == "__main__":
    df = ingest_data()
    print(df.head())
