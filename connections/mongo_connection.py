import os
import pymongo
import certifi
import yaml
from logger import logging

ca = certifi.where()


def load_params(path="params.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


class MongoDBClient:
    client = None

    def __init__(self, params_path="params.yaml") -> None:
        try:
            params = load_params(params_path)

            self.database_name = params["global"]["mongodb"]["database_name"]
            env_var_name = params["global"]["mongodb"]["url_env_key"]

            mongo_db_url = os.getenv(env_var_name)
            if not mongo_db_url:
                raise RuntimeError(f"Environment variable '{env_var_name}' is not set")

            if MongoDBClient.client is None:
                MongoDBClient.client = pymongo.MongoClient(
                    mongo_db_url,
                    tlsCAFile=ca,
                    serverSelectionTimeoutMS=5000
                )
                MongoDBClient.client.admin.command("ping")

            self.client = MongoDBClient.client
            self.database = self.client[self.database_name]

            logging.info(f"MongoDB connected: {self.database_name}")

        except Exception as e:
            logging.exception("MongoDB connection failed")
            raise e

    def get_collection(self, collection_name: str):
        if not collection_name:
            raise ValueError("Collection name must be provided")
        return self.database[collection_name]
