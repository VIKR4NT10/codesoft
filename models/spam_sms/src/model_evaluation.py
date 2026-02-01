# model_evaluation.py

import json
from pathlib import Path
import yaml
from logger import logging
import tensorflow as tf
import warnings
import os
import pandas as pd
import mlflow
import mlflow.tensorflow
import dagshub

from sklearn.metrics import (
    precision_score,  recall_score,  accuracy_score, f1_score, roc_auc_score, average_precision_score,
    classification_report,)
import mlflow.pyfunc

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
    raise EnvironmentError("CODESOFT environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "VIKR4NT10"
repo_name = "codesoft"

# Set up MLflow tracking URI
mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')
# -------------------------------------------------------------------------------------

# =============================================================================
# Helpers
# =============================================================================
def load_params(path: str = "params.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_keras_model(path: Path):
    model = tf.keras.models.load_model(path)
    logging.info("TensorFlow SavedModel loaded from %s", path)
    return model


def load_test_data(processed_dir: Path):
    X_test = pd.read_parquet(processed_dir / "X_test_pad.parquet").values
    y_test = pd.read_parquet(processed_dir / "y_test.parquet")["label"].values
    logging.info("Loaded test data: X=%s, y=%s", X_test.shape, y_test.shape)
    return X_test, y_test


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


def save_json(data: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

def load_tokenizer(path: Path):
    with open(path, "r") as f:
        return tf.keras.preprocessing.text.tokenizer_from_json(f.read())

class SpamSMSModel(mlflow.pyfunc.PythonModel):
    def __init__(self, model, tokenizer, max_len: int):
        self.model = model
        self.tokenizer = tokenizer
        self.max_len = max_len

    def predict(self, context, model_input):
        # model_input: pandas DataFrame with column "text"
        texts = model_input["text"].astype(str).tolist()

        sequences = self.tokenizer.texts_to_sequences(texts)
        padded = tf.keras.preprocessing.sequence.pad_sequences(
            sequences,
            maxlen=self.max_len,
            padding="post"
        )

        probs = self.model.predict(padded, verbose=0).reshape(-1)
        return probs

# =============================================================================
# Main
# =============================================================================
def main():
    MODEL_NAME = "spam_sms"
    EXPERIMENT_NAME = "spam_sms_cnn"

    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run() as run:
        # --------------------------------------------------
        # Paths
        # --------------------------------------------------
        processed_dir = Path(f"data/{MODEL_NAME}/features")
        artifacts_dir = Path(f"artifacts/{MODEL_NAME}")
        reports_dir = Path(f"reports/{MODEL_NAME}")
        reports_dir.mkdir(parents=True, exist_ok=True)

        model_path = artifacts_dir / "model"

        # --------------------------------------------------
        # Load model & data
        # --------------------------------------------------
        model = load_keras_model(model_path)
        X_test, y_test = load_test_data(processed_dir)

        # --------------------------------------------------
        # Evaluate
        # --------------------------------------------------
        metrics, clf_report = evaluate_model(model, X_test, y_test)

        # --------------------------------------------------
        # Save reports
        # --------------------------------------------------
        save_json(metrics, reports_dir / "metrics.json")
        with open(reports_dir / "classification_report.txt", "w") as f:
            f.write(clf_report)

        # --------------------------------------------------
        # Log metrics
        # --------------------------------------------------
        for k, v in metrics.items():
            mlflow.log_metric(k, float(v))

        # --------------------------------------------------
        # log model properly to MLflow
        # --------------------------------------------------
        # Load tokenizer
        tokenizer = load_tokenizer(artifacts_dir / "tokenizer.json")

        pyfunc_model = SpamSMSModel(
            model=model,
            tokenizer=tokenizer,
            max_len=100,  # must match training
        )

        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=pyfunc_model,
        )


        # --------------------------------------------------
        # Save experiment info for promotion pipeline
        # --------------------------------------------------
        experiment_info = {
            "model_name": MODEL_NAME,
            "run_id": run.info.run_id,
            "model_path": "model",
            "promotion_metric": "f1",
        }

        save_json(experiment_info, reports_dir / "experiment_info.json")

        # --------------------------------------------------
        # Log artifacts
        # --------------------------------------------------
        mlflow.log_artifact(str(reports_dir / "metrics.json"))
        mlflow.log_artifact(str(reports_dir / "classification_report.txt"))
        mlflow.log_artifact(str(reports_dir / "experiment_info.json"))

        logging.info("Spam SMS model evaluation completed successfully")


if __name__ == "__main__":
    main()
