from pathlib import Path
import yaml
import joblib
import pandas as pd
from logger import logging

from sklearn.ensemble import RandomForestClassifier
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline


# -------------------------------------------------
# Load params
# -------------------------------------------------
def load_params(path: str = "params.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# -------------------------------------------------
# Load training features
# -------------------------------------------------
def load_train_data(features_dir: Path) -> tuple[pd.DataFrame, pd.Series]:
    train_path = features_dir / "train_features.parquet"

    if not train_path.exists():
        raise FileNotFoundError(f"Missing training features: {train_path}")

    df = pd.read_parquet(train_path)

    X = df.drop(columns=["is_fraud"])
    y = df["is_fraud"]

    logging.info("Loaded training data: X=%s, y=%s", X.shape, y.shape)
    return X, y


# -------------------------------------------------
# Build model pipeline
# -------------------------------------------------
def build_model(model_cfg: dict, random_state: int) -> Pipeline:
    rf = RandomForestClassifier(
        n_estimators=model_cfg["rf__n_estimators"],
        min_samples_leaf=model_cfg["rf__min_samples_leaf"],
        max_features=model_cfg["rf__max_features"],
        max_depth=model_cfg["rf__max_depth"],
        bootstrap=model_cfg["rf__bootstrap"],
        n_jobs=-1,
        random_state=random_state,
    )

    pipeline = Pipeline(
        steps=[
            ("smotetomek", SMOTETomek(random_state=random_state)),
            ("rf", rf),
        ]
    )

    return pipeline


# -------------------------------------------------
# Main
# -------------------------------------------------
def main():
    try:
        MODEL_NAME = "credit_card_fraud"

        params = load_params()
        model_cfg = params["models"][MODEL_NAME]["model"]
        random_state = params["global"]["random_state"]

        features_dir = Path("data/fraud_detection/features")
        artifacts_dir = Path("artifacts") / MODEL_NAME
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Load data
        X_train, y_train = load_train_data(features_dir)

        # Build + train model
        model = build_model(model_cfg, random_state)

        logging.info("Starting model training")
        model.fit(X_train, y_train)
        logging.info("Model training completed")

        # Save model
        model_path = artifacts_dir / "random_forest_fraud.joblib"
        joblib.dump(model, model_path)

        logging.info("Model saved to %s", model_path)

    except Exception as e:
        logging.exception("Model training failed: %s", e)
        raise


if __name__ == "__main__":
    main()
