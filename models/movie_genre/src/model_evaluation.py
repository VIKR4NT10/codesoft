import os
import json
import pickle
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

from logger import logging

# =============================================================================
# MLflow + DagsHub setup
# =============================================================================
mlflow.set_tracking_uri("https://dagshub.com/VIKR4NT10/codesoft.mlflow")
dagshub.init(repo_owner="VIKR4NT10", repo_name="codesoft", mlflow=True)
# =============================================================================


def load_model(path: str):
    with open(path, "rb") as f:
        model = pickle.load(f)
    logging.info("Model loaded from %s", path)
    return model


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    logging.info("Test data loaded from %s", path)
    return df


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray):
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

    return metrics, classification_report(y_test, y_pred)


def save_json(data: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    logging.info("Saved file to %s", path)


def main():
    mlflow.set_experiment("movie-genre-svm")

    with mlflow.start_run() as run:
        # --------------------------------------------------
        # Paths
        # --------------------------------------------------
        model_path = "artifacts/movie_genre/model.pkl"
        test_path = "data/movie_genre/features/test_tfidf.csv"
        reports_dir = "reports/movie_genre"

        # --------------------------------------------------
        # Load
        # --------------------------------------------------
        model = load_model(model_path)
        test_df = load_data(test_path)

        # 🔒 SAFE FEATURE / LABEL SPLIT
        if "label" not in test_df.columns:
            raise ValueError("Expected column 'label' not found in test data")

        X_test = test_df.drop(columns=["label"]).values
        y_test = test_df["label"].values

        # --------------------------------------------------
        # Evaluate
        # --------------------------------------------------
        metrics, clf_report = evaluate_model(model, X_test, y_test)

        # --------------------------------------------------
        # Save reports locally
        # --------------------------------------------------
        save_json(metrics, os.path.join(reports_dir, "metrics.json"))

        with open(os.path.join(reports_dir, "classification_report.txt"), "w") as f:
            f.write(clf_report)

        # --------------------------------------------------
        # Log to MLflow
        # --------------------------------------------------
        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        if hasattr(model, "get_params"):
            mlflow.log_params(model.get_params())

        MODEL_ARTIFACT_PATH = "model"

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=MODEL_ARTIFACT_PATH
        )

        # --------------------------------------------------
        # Save run metadata for registry step
        # --------------------------------------------------
        run_info = {
            "run_id": run.info.run_id,
            "experiment": "movie-genre-svm",
            "model_name": "movie_genre_svm",
            "model_path": MODEL_ARTIFACT_PATH,
            "promotion_metric": "f1_macro",
        }

        save_json(run_info, os.path.join(reports_dir, "experiment_info.json"))

        mlflow.log_artifact(os.path.join(reports_dir, "metrics.json"))
        mlflow.log_artifact(os.path.join(reports_dir, "classification_report.txt"))
        mlflow.log_artifact(os.path.join(reports_dir, "experiment_info.json"))

        logging.info("Model evaluation completed successfully")


if __name__ == "__main__":
    main()
