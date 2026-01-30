# model_evaluation.py

import json
from pathlib import Path
import yaml
import logging
import tensorflow as tf

import pandas as pd
import mlflow
import dagshub

from sklearn.metrics import (
    precision_score,
    recall_score,
    accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
)

from tensorflow.keras.models import load_model as keras_load_model

logging.basicConfig(level=logging.INFO)

# ----------------------------
# MLflow + DagsHub setup
# ----------------------------
mlflow.set_tracking_uri("https://dagshub.com/VIKR4NT10/codesoft.mlflow")
dagshub.init(repo_owner="VIKR4NT10", repo_name="codesoft", mlflow=True)


# ----------------------------
# Load params
# ----------------------------
def load_params(path: str = "params.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ----------------------------
# Load Keras model
# ----------------------------
def load_keras_model(path: Path):
    model = tf.keras.models.load_model(path)
    logging.info(f"TensorFlow SavedModel loaded from {path}")
    return model



# ----------------------------
# Load test data
# ----------------------------
def load_test_data(processed_dir: Path):
    X_test = pd.read_parquet(processed_dir / "X_test_pad.parquet").values
    y_test = pd.read_parquet(processed_dir / "y_test.parquet")["label"].values
    logging.info(f"Loaded test data: X={X_test.shape}, y={y_test.shape}")
    return X_test, y_test


# ----------------------------
# Evaluate model
# ----------------------------
def evaluate_model(model, X, y):
    logging.info("Evaluating SMS spam model")

    y_proba = model.predict(X, batch_size=64).reshape(-1)
    y_pred = (y_proba >= 0.7).astype(int)

    metrics = {
        "roc_auc": roc_auc_score(y, y_proba),
        "pr_auc": average_precision_score(y, y_proba),
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred),
        "recall": recall_score(y, y_pred),
        "f1": f1_score(y, y_pred),
    }

    report = classification_report(y, y_pred)
    return metrics, report


# ----------------------------
# Save JSON helper
# ----------------------------
def save_json(data: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


# ----------------------------
# Main
# ----------------------------
def main():
    MODEL_NAME = "spam_sms"
    EXPERIMENT_NAME = "spam_sms_cnn"

    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run() as run:
        params = load_params()

        processed_dir = Path(f"data/{MODEL_NAME}/features")
        artifacts_dir = Path(f"artifacts/{MODEL_NAME}")
        reports_dir = Path(f"reports/{MODEL_NAME}")
        reports_dir.mkdir(parents=True, exist_ok=True)

        model_path = artifacts_dir / "model"

        # Load artifacts
        model = load_keras_model(model_path)
        X_test, y_test = load_test_data(processed_dir)

        # Evaluate
        metrics, clf_report = evaluate_model(model, X_test, y_test)

        # Save reports
        save_json(metrics, reports_dir / "metrics.json")
        with open(reports_dir / "classification_report.txt", "w") as f:
            f.write(clf_report)

        # Log metrics
        for k, v in metrics.items():
            mlflow.log_metric(k, float(v))

        # Log model
        # ----------------------------
        # Save model as SavedModel (Keras 3 compatible)
        # ----------------------------
        model_save_path = Path("cnn_model")

        # Remove if exists (important for reruns)
        if model_save_path.exists():
            import shutil
            shutil.rmtree(model_save_path)

        # Export SavedModel directory
        model.export(model_save_path)

        # Log entire directory as MLflow artifact
        mlflow.log_artifacts(str(model_save_path), artifact_path="cnn_model")



        # Save run metadata
        run_info = {
            "run_id": run.info.run_id,
            "experiment": EXPERIMENT_NAME,
            "model_type": "cnn",
            "model_path": "cnn_model",
        }
        save_json(run_info, reports_dir / "experiment_info.json")

        # Log artifacts
        mlflow.log_artifact(str(reports_dir / "metrics.json"))
        mlflow.log_artifact(str(reports_dir / "classification_report.txt"))
        mlflow.log_artifact(str(reports_dir / "experiment_info.json"))

        logging.info("Model evaluation completed successfully")


if __name__ == "__main__":
    main()
