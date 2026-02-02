import unittest
import os
import mlflow
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score


class TestProductionModels(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # --------------------------------------------------
        # MLflow + DagsHub authentication
        # --------------------------------------------------
        dagshub_token = os.getenv("CODESOFT")
        if not dagshub_token:
            raise EnvironmentError("CODESOFT environment variable is not set")

        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

        dagshub_url = "https://dagshub.com"
        repo_owner = "VIKR4NT10"
        repo_name = "codesoft"

        mlflow.set_tracking_uri(
            f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow"
        )

        cls.client = mlflow.MlflowClient()

        # --------------------------------------------------
        # Load PRODUCTION models (pyfunc – same as Flask app)
        # --------------------------------------------------
        cls.movie_model_uri = "models:/movie_genre_svm/Production"
        cls.spam_model_uri = "models:/spam_sms_cnn/Production"

        cls.movie_model = mlflow.pyfunc.load_model(cls.movie_model_uri)
        cls.spam_model = mlflow.pyfunc.load_model(cls.spam_model_uri)

        # --------------------------------------------------
        # Load preprocessing artifacts
        # --------------------------------------------------
        cls.movie_vectorizer = cls._load_artifact(
            "movie_genre_svm", "vectorizer.pkl"
        )

        cls.spam_tokenizer = cls._load_artifact(
            "spam_sms_cnn", "tokenizer.pkl"
        )

        # --------------------------------------------------
        # Load holdout test data
        # --------------------------------------------------
        cls.movie_test = pd.read_csv(
            "data/movie_genre/features/test.csv"
        )

        cls.spam_X = pd.read_parquet(
            "data/spam_sms/features/X_test_pad.parquet"
        ).values

        cls.spam_y = pd.read_parquet(
            "data/spam_sms/features/y_test.parquet"
        )["label"].values

    # --------------------------------------------------
    # Helper to load MLflow artifacts
    # --------------------------------------------------
    @classmethod
    def _load_artifact(cls, model_name, artifact_name):
        versions = cls.client.get_latest_versions(
            model_name, stages=["Production"]
        )

        if not versions:
            raise RuntimeError(f"No Production version found for {model_name}")

        run_id = versions[0].run_id
        artifact_path = cls.client.download_artifacts(run_id, artifact_name)

        with open(artifact_path, "rb") as f:
            return pickle.load(f)

    # --------------------------------------------------
    # Smoke tests
    # --------------------------------------------------
    def test_models_loaded(self):
        self.assertIsNotNone(self.movie_model)
        self.assertIsNotNone(self.spam_model)

    def test_preprocessors_loaded(self):
        self.assertIsNotNone(self.movie_vectorizer)
        self.assertIsNotNone(self.spam_tokenizer)

    # --------------------------------------------------
    # Signature / interface tests (mirror Flask app)
    # --------------------------------------------------
    def test_movie_genre_signature(self):
        df = pd.DataFrame(
            {"text": ["A detective investigates a mysterious murder"]}
        )

        preds = self.movie_model.predict(df)
        self.assertEqual(len(preds), 1)

    def test_spam_sms_signature(self):
        df = pd.DataFrame(
            {"text": ["Congratulations! You won a free prize"]}
        )

        preds = self.spam_model.predict(df)
        self.assertEqual(len(preds), 1)

    # --------------------------------------------------
    # Performance regression tests
    # --------------------------------------------------
    def test_movie_genre_performance(self):
        df = self.movie_test[["text"]]
        y_true = self.movie_test["label"]

        preds = self.movie_model.predict(df)
        f1 = f1_score(y_true, preds, average="macro")

        self.assertGreaterEqual(
            f1, 0.40,
            "Movie genre macro-F1 below acceptable threshold"
        )

    def test_spam_sms_performance(self):
        probs = self.spam_model.predict(
            pd.DataFrame({"text": self.spam_X.flatten()})
        ).astype(float)

        preds = (probs >= 0.7).astype(int)

        acc = accuracy_score(self.spam_y, preds)
        f1 = f1_score(self.spam_y, preds)

        self.assertGreaterEqual(acc, 0.85)
        self.assertGreaterEqual(f1, 0.85)


if __name__ == "__main__":
    unittest.main()
