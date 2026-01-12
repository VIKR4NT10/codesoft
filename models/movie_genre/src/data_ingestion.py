import os
import yaml
from logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from connections.mongo_connection import MongoDBClient

logging.basicConfig(level=logging.INFO)


def load_params(path: str = "params.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("Preprocessing movie genre data")
    df = df.dropna()
    return df[["title","description", "genre"]]


def fetch_data_from_mongodb(params_path="params.yaml") -> pd.DataFrame:
    params = load_params(params_path)

    collection_name = params["models"]["movie_genre"]["data_ingestion"]["collection_name"]

    mongo_client = MongoDBClient(params_path)
    collection = mongo_client.get_collection(collection_name)

    data = list(collection.find({}, {"_id": 0}))
    if not data:
        raise ValueError("No data found in MongoDB collection")

    return pd.DataFrame(data)


def save_data(df, model_name, test_size, random_state):

    path = os.path.join( "data", model_name, "raw")
    os.makedirs(path, exist_ok=True)

    train, test = train_test_split(df, test_size=test_size, random_state=random_state)

    train.to_csv(f"{path}/train.csv", index=False)
    test.to_csv(f"{path}/test.csv", index=False)
    logging.info("Train/test data saved for %s model", model_name)


def main():
    model_name = "movie_genre"

    params = load_params()
    cfg = params["models"][model_name]["data_ingestion"]

    df = fetch_data_from_mongodb()   
    df = preprocess(df)

    save_data(
        df,
        model_name,
        cfg["test_size"],
        params["global"]["random_state"]
    )


if __name__ == "__main__":
    main()

