# register_model.py
import os
import json
from logger import logging
import warnings

import mlflow
import dagshub
from mlflow.tracking import MlflowClient

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)

# =============================================================================
# MLflow + DagsHub setup
# =============================================================================
# mlflow.set_tracking_uri("https://dagshub.com/VIKR4NT10/codesoft.mlflow")
# dagshub.init(repo_owner="VIKR4NT10", repo_name="codesoft", mlflow=True)


# Below code block is for production use
# -------------------------------------------------------------------------------------
# Set up DagsHub credentials for MLflow tracking
dagshub_token = os.getenv("CODESOFT")
if not dagshub_token:
    raise EnvironmentError("CAPSTONE environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "VIKR4NT10"
repo_name = "codesoft"

# Set up MLflow tracking URI
mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')
# -------------------------------------------------------------------------------------

PROMOTION_DELTA = 0.01  # minimum F1 improvement


# ----------------------------
# Helpers
# ----------------------------
def load_model_info(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def get_metric_from_run(run_id: str, metric_name: str) -> float:
    run = mlflow.get_run(run_id)
    metric = run.data.metrics.get(metric_name)

    if metric is None:
        raise ValueError(f"Metric '{metric_name}' not found in run {run_id}")

    return metric


def get_production_metric(model_name: str, metric_name: str, client: MlflowClient):
    versions = client.get_latest_versions(model_name, stages=["Production"])
    if not versions:
        return None, None

    prod_version = versions[0]
    prod_metric = get_metric_from_run(prod_version.run_id, metric_name)
    return prod_metric, prod_version.version


# ----------------------------
# Register & Promote
# ----------------------------
def register_and_maybe_promote(model_name: str, model_info: dict):
    client = MlflowClient()

    run_id = model_info["run_id"]
    model_path = model_info["model_path"]

    model_uri = f"runs:/{run_id}/{model_path}"
    logging.info(f"Registering model from {model_uri}")

    model_version = client.create_model_version(
        name=model_name,
        source=model_uri,
        run_id=run_id,
    )

    metric_name = "f1"
    new_metric = get_metric_from_run(run_id, metric_name)
    logging.info(f"New model F1: {new_metric:.4f}")

    prod_metric, prod_version = get_production_metric(
        model_name, metric_name, client
    )

    if prod_metric is None:
        logging.info("No Production model found. Promoting directly.")
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Production",
            archive_existing_versions=True,
        )
        return

    improvement = new_metric - prod_metric
    logging.info(
        f"Production F1: {prod_metric:.4f} | Improvement: {improvement:.4f}"
    )

    if improvement >= PROMOTION_DELTA:
        logging.info("Promotion condition met. Promoting to Production.")
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Production",
            archive_existing_versions=True,
        )
    else:
        logging.info("Promotion condition NOT met. Moving to Staging.")
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Staging",
        )



# ----------------------------
# Main
# ----------------------------
def main():
    model_info_path = "reports/spam_sms/experiment_info.json"
    model_info = load_model_info(model_info_path)

    model_name = "spam_sms_cnn"
    register_and_maybe_promote(model_name, model_info)


if __name__ == "__main__":
    main()
