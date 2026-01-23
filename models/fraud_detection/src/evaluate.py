from pathlib import Path
import json
import yaml
import joblib
import pandas as pd
import mlflow
import dagshub

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
)

from logger import logging

# =============================================================================
# MLflow + DagsHub setup (production)
# =============================================================================
# dagshub_token = os.getenv("CAPSTONE_TEST")
# if not dagshub_token:   
#     raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

# os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
# os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

# dagshub_url = "https://dagshub.com"
# repo_owner = "VIKR4NT"
# repo_name = "codesoft"

# mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")
# dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
# =============================================================================
# =============================================================================
# MLflow + DagsHub setup (local / prod safe)
# =============================================================================
mlflow.set_tracking_uri("https://dagshub.com/VIKR4NT10/codesoft.mlflow")
dagshub.init(repo_owner="VIKR4NT10", repo_name="codesoft", mlflow=True)
# =============================================================================

# -------------------------------------------------
# Load params
# -------------------------------------------------
def load_params(path: str = "params.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

# -------------------------------------------------
# Load model
# -------------------------------------------------
def load_model(path: Path):
    model = joblib.load(path)
    logging.info("Model loaded from %s", path)
    return model

# -------------------------------------------------
# Load test data
# -------------------------------------------------
def load_test_data(features_dir: Path):
    test_path = features_dir / "test_features.parquet"

    if not test_path.exists():
        raise FileNotFoundError(f"Missing test features: {test_path}")

    df = pd.read_parquet(test_path)

    X = df.drop(columns=["is_fraud"])
    y = df["is_fraud"]

    logging.info("Loaded test data: X=%s, y=%s", X.shape, y.shape)
    return X, y


# -------------------------------------------------
# Evaluate model
# -------------------------------------------------
def evaluate_model(model, X, y) -> tuple[dict, str]:
    logging.info("Evaluating fraud detection model")

    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    metrics = {
        "roc_auc": roc_auc_score(y, y_proba),
        "pr_auc": average_precision_score(y, y_proba),
        "precision": precision_score(y, y_pred),
        "recall": recall_score(y, y_pred),
        "f1": f1_score(y, y_pred),
    }

    report = classification_report(y, y_pred)

    logging.info("Evaluation metrics computed")
    return metrics, report


# -------------------------------------------------
# Save JSON helper
# -------------------------------------------------
def save_json(data: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    logging.info("Saved file to %s", path)


# -------------------------------------------------
# Main
# -------------------------------------------------
def main():
    MODEL_NAME = "credit_card_fraud"
    EXPERIMENT_NAME = "credit-card-fraud-random-forest"

    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run() as run:
        try:
            params = load_params()

            features_dir = Path("data/fraud_detection/features")
            artifacts_dir = Path("artifacts") / MODEL_NAME
            reports_dir = Path("reports") / MODEL_NAME

            model_path = artifacts_dir / "random_forest_fraud.joblib"

            # Load artifacts
            model = load_model(model_path)
            X_test, y_test = load_test_data(features_dir)

            # Evaluate
            metrics, clf_report = evaluate_model(model, X_test, y_test)

            # Save local reports
            save_json(metrics, reports_dir / "metrics.json")

            with open(reports_dir / "classification_report.txt", "w") as f:
                f.write(clf_report)

            # -------------------------------
            # MLflow logging
            # -------------------------------
            for k, v in metrics.items():
                mlflow.log_metric(k, v)

            # Log model hyperparameters (RF only, not SMOTE)
            rf = model.named_steps["rf"]
            for k, v in rf.get_params().items():
                mlflow.log_param(k, v)

            # Log model artifact
            mlflow.sklearn.log_model(model, artifact_path="random_forest_model")

            # Save run metadata
            run_info = {
                "run_id": run.info.run_id,
                "experiment": EXPERIMENT_NAME,
                "model_type": "random_forest",
                "artifact_path": str(model_path),
            }

            save_json(run_info, reports_dir / "experiment_info.json")

            # Log reports
            mlflow.log_artifact(str(reports_dir / "metrics.json"))
            mlflow.log_artifact(str(reports_dir / "classification_report.txt"))
            mlflow.log_artifact(str(reports_dir / "experiment_info.json"))

            logging.info("Model evaluation completed successfully")

        except Exception as e:
            logging.exception("Model evaluation failed: %s", e)
            raise


if __name__ == "__main__":
    main()
