# register_model.py

import json
import os
import warnings
import mlflow
from mlflow import MlflowClient
from logger import logging
import dagshub

warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

# -------------------------------------------------------------------------------------
# Production MLflow + DagsHub setup
# -------------------------------------------------------------------------------------
dagshub_token = os.getenv("CAPSTONE_TEST")
if not dagshub_token:
    raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "vikashdas770"
repo_name = "YT-Capstone-Project"

mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")
# -------------------------------------------------------------------------------------


PROMOTION_DELTA = 0.005  # 0.5% improvement threshold


def load_model_info(file_path: str) -> dict:
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except Exception as e:
        logging.error("Failed to load model info: %s", e)
        raise


def get_accuracy_from_run(run_id: str) -> float:
    try:
        run = mlflow.get_run(run_id)
        accuracy = run.data.metrics.get("accuracy")

        if accuracy is None:
            raise ValueError("Accuracy metric not found in MLflow run")

        return accuracy
    except Exception as e:
        logging.error("Failed to fetch accuracy from run %s: %s", run_id, e)
        raise


def get_production_model_accuracy(model_name: str, client: MlflowClient):
    try:
        versions = client.get_latest_versions(model_name, stages=["Production"])
        if not versions:
            return None, None

        prod_version = versions[0]
        prod_run_id = prod_version.run_id
        prod_accuracy = get_accuracy_from_run(prod_run_id)

        return prod_accuracy, prod_version.version
    except Exception as e:
        logging.error("Failed to fetch production model accuracy: %s", e)
        raise


def register_and_maybe_promote(model_name: str, model_info: dict):
    client = MlflowClient()

    model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"

    logging.info("Registering model: %s", model_name)
    model_version = mlflow.register_model(model_uri, model_name)

    new_accuracy = get_accuracy_from_run(model_info["run_id"])
    logging.info("New model accuracy: %.4f", new_accuracy)

    prod_accuracy, prod_version = get_production_model_accuracy(model_name, client)

    # --------------------------------------------------
    # Promotion Decision
    # --------------------------------------------------
    if prod_accuracy is None:
        logging.info("No Production model found. Promoting directly to Production.")

        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Production",
            archive_existing_versions=True
        )
        return

    improvement = new_accuracy - prod_accuracy
    logging.info(
        "Production accuracy: %.4f | Improvement: %.4f",
        prod_accuracy,
        improvement
    )

    if improvement >= PROMOTION_DELTA:
        logging.info("Promotion condition met. Promoting model to Production.")

        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Production",
            archive_existing_versions=True
        )
    else:
        logging.info(
            "Promotion condition NOT met. Keeping model in Staging."
        )

        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Staging"
        )


def main():
    try:
        model_info = load_model_info("reports/movie_genre/experiment_info.json")
        model_name = "my_model"

        register_and_maybe_promote(model_name, model_info)

    except Exception as e:
        logging.error("Model registration failed: %s", e)
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
