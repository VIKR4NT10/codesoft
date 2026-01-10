import os
import yaml
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from src.connections.mongodb_connection import load_mongo_collection

logging.basicConfig(level=logging.INFO)


def load_params(path="params.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("Preprocessing customer churn data")
    df = df.dropna()
    return df[["customerID", "gender", "SeniorCitizen", "tenure", "Churn"]]


def save_data(df, model_name, test_size, random_state):
    train, test = train_test_split(
        df, test_size=test_size, random_state=random_state
    )
    path = f"data/{model_name}/raw"
    os.makedirs(path, exist_ok=True)
    train.to_csv(f"{path}/train.csv", index=False)
    test.to_csv(f"{path}/test.csv", index=False)
    logging.info("Train/test data saved for %s model", model_name)


def main():
    model_name = "customer_churn"
    params = load_params()
    cfg = params["models"][model_name]["data_ingestion"]

    df = load_mongo_collection(cfg)
    df = preprocess(df)
    save_data(df, model_name, cfg["test_size"], params["global"]["random_state"])


if __name__ == "__main__":
    main()
