import os
import yaml
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from src.logger import logging


def load_params(params_path: str = "params.yaml") -> dict:
    """Load parameters from YAML."""
    with open(params_path, "r") as f:
        params = yaml.safe_load(f)
    logging.info("Parameters loaded from %s", params_path)
    return params


def load_data(path: str) -> pd.DataFrame:
    """Load processed CSV data."""
    df = pd.read_csv(path)
    df.fillna("", inplace=True)
    logging.info("Data loaded from %s", path)
    return df


def apply_tfidf(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    tfidf_params: dict
) -> tuple:
    """
    Apply TF-IDF vectorization on movie plot text
    """
    logging.info("Applying TF-IDF Vectorizer")

    vectorizer = TfidfVectorizer(
        max_features=tfidf_params["max_features"],
        ngram_range=tuple(tfidf_params["ngram_range"]),
        min_df=tfidf_params.get("min_df", 1),
        max_df=tfidf_params.get("max_df", 1.0)
    )

    X_train = train_df["text"].values
    y_train = train_df["GENRE"].values

    X_test = test_df["text"].values
    y_test = test_df["GENRE"].values

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Convert to DataFrame
    train_features = pd.DataFrame(X_train_tfidf.toarray())
    train_features["label"] = y_train

    test_features = pd.DataFrame(X_test_tfidf.toarray())
    test_features["label"] = y_test

    # Save vectorizer
    artifact_path = os.path.join("artifacts", "movie_genre")
    os.makedirs(artifact_path, exist_ok=True)

    with open(os.path.join(artifact_path, "tfidf_vectorizer.pkl"), "wb") as f:
        pickle.dump(vectorizer, f)

    logging.info("TF-IDF vectorizer saved")

    return train_features, test_features


def save_data(df: pd.DataFrame, path: str):
    """Save feature-engineered data."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    logging.info("Feature data saved to %s", path)


def main():
    try:
        params = load_params()
        fe_params = params["models"]["movie_genre"]["feature_engineering"]

        base_path = os.path.join("data", "movie_genre", "processed")
        train_df = load_data(os.path.join(base_path, "train_processed.csv"))
        test_df = load_data(os.path.join(base_path, "test_processed.csv"))

        train_features, test_features = apply_tfidf(
            train_df, test_df, fe_params
        )

        feature_path = os.path.join("data", "movie_genre", "features")
        save_data(train_features, os.path.join(feature_path, "train_tfidf.csv"))
        save_data(test_features, os.path.join(feature_path, "test_tfidf.csv"))

        logging.info("Feature engineering completed successfully")

    except Exception as e:
        logging.error("Feature engineering failed: %s", e)
        raise


if __name__ == "__main__":
    main()
