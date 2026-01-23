import os
import json
import pickle
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub

from sklearn.metrics import (
     accuracy_score, precision_score, recall_score, f1_score, classification_report)

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
# Below code block is for local use
# -------------------------------------------------------------------------------------
mlflow.set_tracking_uri('https://dagshub.com/VIKR4NT10/codesoft.mlflow')
dagshub.init(repo_owner='VIKR4NT10', repo_name='codesoft', mlflow=True)
# -------------------------------------------------------------------------------------


def load_model(path: str):
    """Load trained model."""
    with open(path, "rb") as f:
        model = pickle.load(f)
    logging.info("Model loaded from %s", path)
    return model


def load_data(path: str) -> pd.DataFrame:
    """Load test feature data."""
    df = pd.read_csv(path)
    logging.info("Test data loaded from %s", path)
    return df


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Evaluate multi-class SVM model
    """
    logging.info("Evaluating SVM model")

    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(y_test, y_pred, average="macro"),
        "recall_macro": recall_score(y_test, y_pred, average="macro"),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
        "precision_weighted": precision_score(y_test, y_pred, average="weighted"),
        "recall_weighted": recall_score(y_test, y_pred, average="weighted"),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
    }

    logging.info("Evaluation metrics computed")
    return metrics, classification_report(y_test, y_pred)


def save_json(data: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    logging.info("Saved file to %s", path)


def main():
    mlflow.set_experiment("movie-genre-svm")

    with mlflow.start_run() as run:
        try:
            # Load artifacts
            model_path = os.path.join(
                "artifacts", "movie_genre", "model.pkl"
            )
            test_path = os.path.join(
                "data", "movie_genre", "features", "test_tfidf.csv"
            )

            model = load_model(model_path)
            test_df = load_data(test_path)

            X_test = test_df.iloc[:, :-1].values
            y_test = test_df.iloc[:, -1].values

            # Evaluate
            metrics, clf_report = evaluate_model(model, X_test, y_test)

            # Save metrics locally
            reports_dir = os.path.join("reports", "movie_genre")
            save_json(metrics, os.path.join(reports_dir, "metrics.json"))

            with open(os.path.join(reports_dir, "classification_report.txt"), "w") as f:
                f.write(clf_report)

            # Log metrics to MLflow
            for k, v in metrics.items():
                mlflow.log_metric(k, v)

            # Log model params
            if hasattr(model, "get_params"):
                for k, v in model.get_params().items():
                    mlflow.log_param(k, v)

            # Log model artifact
            mlflow.sklearn.log_model(model, "svm_model")

            # Save run info
            run_info = {
            "run_id": run.info.run_id,
            "experiment": "movie-genre",
            "model_type": "svm",
            "model_artifact_path": "artifacts/movie_genre/model.pkl"
            }

            save_json(
                run_info,
                os.path.join(reports_dir, "experiment_info.json")
            )

            # Log artifacts
            mlflow.log_artifact(os.path.join(reports_dir, "metrics.json"))
            mlflow.log_artifact(os.path.join(reports_dir, "classification_report.txt"))

            logging.info("Model evaluation completed successfully")

        except Exception as e:
            logging.error("Model evaluation failed: %s", e)
            raise


if __name__ == "__main__":
    main()
