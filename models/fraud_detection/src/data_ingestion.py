from pathlib import Path
from datetime import datetime, timedelta
import yaml
import pandas as pd

from logger import logging
from connections.mongo_connection import MongoDBClient


# -------------------------------------------------
# Load params.yaml
# -------------------------------------------------
def load_config(config_path: str = "params.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# -------------------------------------------------
# Identify latest ingested month from raw directory
# -------------------------------------------------
def get_latest_ingested_month(raw_dir: Path) -> str | None:
    """
    Looks for files like: transactions_YYYY_MM.parquet
    Returns: YYYY-MM or None
    """
    files = list(raw_dir.glob("transactions_????_??.parquet"))
    if not files:
        return None

    def extract_year_month(path: Path):
        _, year, month = path.stem.split("_")
        return int(year), int(month)

    latest_file = max(files, key=extract_year_month)
    year, month = extract_year_month(latest_file)
    return f"{year:04d}-{month:02d}"


# -------------------------------------------------
# Get last fully completed calendar month
# -------------------------------------------------
def get_last_complete_month() -> str:
    today = datetime.today().replace(day=1)
    last_month_date = today - timedelta(days=1)
    return last_month_date.strftime("%Y-%m")


# -------------------------------------------------
# Fetch MongoDB data for a given month
# -------------------------------------------------
def fetch_month_data(collection, year_month: str) -> pd.DataFrame:
    year, month = map(int, year_month.split("-"))

    start = datetime(year, month, 1)
    end = start.replace(day=28) + timedelta(days=4)
    end = end.replace(day=1)

    query = {
        "trans_date_trans_time": {
            "$gte": start,
            "$lt": end
        }
    }

    data = list(collection.find(query, {"_id": 0}))
    return pd.DataFrame(data)


# -------------------------------------------------
# Fetch ALL MongoDB data (first run)
# -------------------------------------------------
def fetch_all_data(collection) -> pd.DataFrame:
    data = list(collection.find({}, {"_id": 0}))
    return pd.DataFrame(data)


# -------------------------------------------------
# Main ingestion function
# -------------------------------------------------
def ingest_monthly_data(
    config_path: str = "params.yaml",
    fail_on_no_data: bool = False
) -> bool:
    """
    Returns:
        True  -> data ingested successfully
        False -> no new data, downstream tasks should be skipped
    """

    config = load_config(config_path)
    MODEL_NAME = "fraud_detection"
    model_cfg = config["models"][MODEL_NAME]

    raw_dir = Path("data/fraud_detection/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    latest_ingested = get_latest_ingested_month(raw_dir)
    target_month = get_last_complete_month()

    mongo_client = MongoDBClient(config_path)

    try:
        collection_name = model_cfg["data_ingestion"]["collection_name"]
        collection = mongo_client.get_collection(collection_name)

        # -------------------------------------------------
        # FIRST RUN: no monthly data exists → ingest ALL data
        # -------------------------------------------------
        if latest_ingested is None:
            logging.info(
                "No monthly raw data found. First run detected — ingesting FULL dataset."
            )

            df = fetch_all_data(collection)

            if df.empty:
                logging.warning("MongoDB collection is empty")
                if fail_on_no_data:
                    raise RuntimeError("Ingestion failed: empty dataset")
                return False

            file_path = raw_dir / "transactions_full.parquet"
            df.to_parquet(file_path, index=False)

            logging.info(
                f"Ingested FULL dataset with {len(df)} records "
                f"and saved to {file_path}"
            )
            return True

        # -------------------------------------------------
        # INCREMENTAL: monthly ingestion
        # -------------------------------------------------
        if latest_ingested == target_month:
            logging.info(
                f"No new data available. Month {target_month} already ingested."
            )
            if fail_on_no_data:
                raise RuntimeError("Ingestion skipped: no new monthly data")
            return False

        logging.info(f"Starting monthly ingestion for {target_month}")
        df = fetch_month_data(collection, target_month)

    finally:
        mongo_client.close()

    # -------------------------------------------------
    # Validate fetched data
    # -------------------------------------------------
    if df.empty:
        logging.warning(f"No records found in MongoDB for {target_month}")
        if fail_on_no_data:
            raise RuntimeError("Ingestion failed: empty dataset")
        return False

    # -------------------------------------------------
    # Persist raw monthly data
    # -------------------------------------------------
    file_path = raw_dir / f"transactions_{target_month.replace('-', '_')}.parquet"
    df.to_parquet(file_path, index=False)

    logging.info(
        f"Ingested {len(df)} records for {target_month} "
        f"and saved to {file_path}"
    )

    return True


# -------------------------------------------------
# Entrypoint
# -------------------------------------------------
if __name__ == "__main__":
    try:
        success = ingest_monthly_data(config_path="params.yaml")
        if success:
            logging.info("Data ingestion completed successfully")
        else:
            logging.info("No new data ingested")
    except Exception as e:
        logging.exception(f"Data ingestion failed: {e}")
        raise
