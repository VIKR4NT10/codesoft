import json
import time
import warnings
import mlflow
import dagshub
import os
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException
from logger import logging

warnings.filterwarnings("ignore")

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
    raise EnvironmentError("CODESOFT environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "VIKR4NT10"
repo_name = "codesoft"

# Set up MLflow tracking URI
mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')
# -------------------------------------------------------------------------------------

PROMOTION_DELTA = 0.005  # 0.5% improvement threshold
MAX_RETRIES = 5
RETRY_DELAY = 10  # seconds


# =============================================================================
# Helpers
# =============================================================================
def load_model_info(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def get_metric_from_run(run_id: str, metric_name: str) -> float:
    run = mlflow.get_run(run_id)
    value = run.data.metrics.get(metric_name)

    if value is None:
        raise ValueError(f"Metric '{metric_name}' not found in run {run_id}")

    return value


def get_production_metric(model_name: str, metric_name: str, client: MlflowClient):
    versions = client.get_latest_versions(model_name, stages=["Production"])
    if not versions:
        return None, None

    prod_version = versions[0]
    prod_metric = get_metric_from_run(prod_version.run_id, metric_name)
    return prod_metric, prod_version.version


def create_model_version_with_retry(
    client: MlflowClient,
    model_name: str,
    model_uri: str,
    run_id: str,
):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logging.info("Creating model version (attempt %d)", attempt)
            return client.create_model_version(
                name=model_name,
                source=model_uri,
                run_id=run_id,
            )
        except MlflowException as e:
            logging.warning(
                "Registry attempt %d failed (%s). Retrying in %ds...",
                attempt,
                str(e),
                RETRY_DELAY,
            )
            if attempt == MAX_RETRIES:
                raise
            time.sleep(RETRY_DELAY)


# =============================================================================
# Register & Promote
# =============================================================================
def register_and_maybe_promote(model_info: dict):
    client = MlflowClient()

    model_name = model_info["model_name"]
    run_id = model_info["run_id"]
    metric_name = model_info["promotion_metric"]
    model_path = model_info["model_path"]  # artifact path

    model_uri = f"runs:/{run_id}/{model_info['model_path']}"

    logging.info("Registering model %s from %s", model_name, model_uri)

    model_version = create_model_version_with_retry(
        client, model_name, model_uri, run_id
    )
    time.sleep(2)
    
    new_metric = get_metric_from_run(run_id, metric_name)
    logging.info("New model %s: %.4f", metric_name, new_metric)

    prod_metric, _ = get_production_metric(model_name, metric_name, client)

    # --------------------------------------------------
    # Promotion logic
    # --------------------------------------------------
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
        "Production %s: %.4f | Improvement: %.4f",
        metric_name,
        prod_metric,
        improvement,
    )

    target_stage = (
        "Production" if improvement >= PROMOTION_DELTA else "Staging"
    )

    client.transition_model_version_stage(
        name=model_name,
        version=model_version.version,
        stage=target_stage,
        archive_existing_versions=True,
    )


# =============================================================================
# Main
# =============================================================================
def main():
    model_info = load_model_info("reports/movie_genre/experiment_info.json")
    register_and_maybe_promote(model_info)


if __name__ == "__main__":
    main()
