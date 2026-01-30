import os
import pickle
import yaml
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from logger import logging


def load_params(params_path: str = "params.yaml") -> dict:
    """Load parameters from YAML."""
    with open(params_path, "r") as f:
        params = yaml.safe_load(f)
    logging.info("Parameters loaded from %s", params_path)
    return params


def load_data(path: str) -> pd.DataFrame:
    """Load feature-engineered training data."""
    df = pd.read_csv(path)
    logging.info("Training data loaded from %s", path)
    return df


def train_model(X: np.ndarray, y: np.ndarray, C: float) -> LinearSVC:
    """
    Train SVM classifier (Linear SVM for TF-IDF)
    """
    try:
        logging.info("Training SVM model with C=%s", C)
        clf = LinearSVC(C=C, random_state=42)
        clf.fit(X, y)
        logging.info("SVM model training completed")
        return clf
    except Exception as e:
        logging.error("Model training failed: %s", e)
        raise


def save_artifact(obj, path: str):
    """Persist model or vectorizer."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    logging.info("Artifact saved to %s", path)


def main():
    try:
        params = load_params()

        # Optuna-tuned hyperparameters
        svm_C = params["models"]["movie_genre"]["training"]["C"]

        # Load training data
        feature_path = os.path.join(
            "data", "movie_genre", "features", "train_tfidf.csv"
        )
        train_df = load_data(feature_path)

        X_train = train_df.iloc[:, :-1].values
        y_train = train_df.iloc[:, -1].values

        # Train model
        model = train_model(X_train, y_train, svm_C)

        # Save model
        artifact_dir = os.path.join("artifacts", "movie_genre")
        save_artifact(model, os.path.join(artifact_dir, "model.pkl"))

        logging.info("Model building pipeline completed successfully")

    except Exception as e:
        logging.error("Model building failed: %s", e)
        raise


if __name__ == "__main__":
    main()
