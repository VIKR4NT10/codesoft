import os
import json
import pickle
import warnings
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
import mlflow
import mlflow.sklearn
import dagshub
import os
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

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
EXPERIMENT_NAME = "movie-genre-svm"

# =============================================================================
# Helpers
# =============================================================================
def load_model(path: str):
    with open(path, "rb") as f:
        model = pickle.load(f)
    logging.info("Loaded model from %s", path)
    return model


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    logging.info("Loaded test data from %s", path)
    return df


def evaluate_model(model, X, y) -> dict:
    y_pred = model.predict(X)

    metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y, y_pred, average="weighted", zero_division=0),
        "f1_macro": f1_score(y, y_pred, average="macro"),

    }

    # if hasattr(model, "predict_proba"):
    #     y_proba = model.predict_proba(X)
    #     metrics["roc_auc"] = roc_auc_score(y, y_proba, multi_class="ovr", average="weighted")

    logging.info("Evaluation metrics calculated")
    return metrics


def save_json(data: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    logging.info("Saved file to %s", path)


# =============================================================================
# Main
# =============================================================================
def main():
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run() as run:
        try:
            # --------------------------------------------------
            # Load model and data
            # --------------------------------------------------
            def load_vectorizer(path: str):
                with open(path, "rb") as f:
                    vectorizer = pickle.load(f)
                logging.info("Loaded vectorizer from %s", path)
                return vectorizer

            clf = load_model("artifacts/movie_genre/model.pkl")
            vectorizer = load_vectorizer("artifacts/movie_genre/vectorizer.pkl")

            model = Pipeline([
                ("tfidf", vectorizer),
                ("clf", clf),
            ])
            test_df = load_data("data/movie_genre/processed/test_processed.csv")
            X_test = test_df["text"].values
            y_test = test_df["genre"].values
    
            # --------------------------------------------------
            # Evaluate
            # --------------------------------------------------
            metrics = evaluate_model(model, X_test, y_test)

            # --------------------------------------------------
            # Log metrics
            # --------------------------------------------------
            for k, v in metrics.items():
                mlflow.log_metric(k, v)

            # --------------------------------------------------
            # Log model parameters
            # --------------------------------------------------
            if hasattr(model, "get_params"):
                for param, value in model.get_params().items():
                    mlflow.log_param(param, value)

            # -------------------------------------------------
            # Log the model as an MLflow artifact
            # --------------------------------------------------
            mlflow.sklearn.log_model(model, artifact_path="model")

            # --------------------------------------------------
            # Save experiment info for promotion pipeline
            # --------------------------------------------------
            experiment_info = {
                "model_name": "movie_genre_svm",
                "run_id": run.info.run_id,
                "model_path": "model",
                "promotion_metric": "f1_macro"
            }
            save_json(
                experiment_info,
                "reports/movie_genre/experiment_info.json",
            )

            mlflow.log_artifact("reports/movie_genre/experiment_info.json")

            logging.info("Model evaluation and logging completed successfully")

        except Exception as e:
            logging.error("Model evaluation failed: %s", e)
            raise


if __name__ == "__main__":
    main()
