import os
import yaml
import logging
from sklearn.model_selection import train_test_split
from src.connections.mysql_connection import load_mysql_table

import pandas as pd

logging.basicConfig(level=logging.INFO)


def load_params(params_path: str = "params.yaml") -> dict:
    with open(params_path, "r") as file:
        params = yaml.safe_load(file)
    logging.info("Params loaded from %s", params_path)
    return params


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("Preprocessing movie genre data")
    df = df.copy().dropna()
    df = df[["plot", "genre"]]
    logging.info("Preprocessing completed")
    return df


def save_data(df: pd.DataFrame, model_name: str, test_size: float, random_state: int):
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    raw_path = os.path.join("data", model_name, "raw")
    os.makedirs(raw_path, exist_ok=True)

    train_df.to_csv(os.path.join(raw_path, "train.csv"), index=False)
    test_df.to_csv(os.path.join(raw_path, "test.csv"), index=False)
    logging.info("Train/test data saved for %s model", model_name)


def main():
    model_name = "movie_genre"
    params = load_params()
    ingestion_params = params["models"][model_name]["data_ingestion"]

    df = load_mysql_table(ingestion_params)
    df = preprocess(df)
    save_data(df, model_name, ingestion_params["test_size"], params["global"]["random_state"])


if __name__ == "__main__":
    main()
