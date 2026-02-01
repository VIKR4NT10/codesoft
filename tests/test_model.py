import unittest
import os
import mlflow
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import tensorflow as tf


class TestProductionModels(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # --------------------------------------------------
        # MLflow + DagsHub auth
        # --------------------------------------------------
        dagshub_token = os.getenv("CODESOFT")
        if not dagshub_token:
            raise EnvironmentError("CAPSTONE environment variable is not set")

        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

        dagshub_url = "https://dagshub.com"
        repo_owner = "VIKR4NT10"
        repo_name = "codesoft"

        # Set up MLflow tracking URI
        mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')
        cls.client = mlflow.MlflowClient()

        # --------------------------------------------------
        # Load PRODUCTION models
        # --------------------------------------------------
        cls.movie_model_uri = "models:/movie_genre_svm/Production"
        cls.spam_model_uri = "models:/spam_sms/Production"

        cls.movie_model = mlflow.sklearn.load_model(cls.movie_model_uri)
        cls.spam_model = mlflow.tensorflow.load_model(cls.spam_model_uri)

        # --------------------------------------------------
        # Load preprocessing artifacts from MLflow
        # --------------------------------------------------
        cls.movie_vectorizer = cls._load_artifact(
            "movie_genre_svm", "vectorizer.pkl"
        )

        cls.spam_tokenizer = cls._load_artifact(
            "spam_sms", "tokenizer.pkl"
        )

        # --------------------------------------------------
        # Load holdout data
        # --------------------------------------------------
        cls.movie_test = pd.read_csv(
            "data/movie_genre/features/test_tfidf.csv"
        )

        cls.spam_X = pd.read_parquet(
            "data/spam_sms/features/X_test_pad.parquet"
        ).values

        cls.spam_y = pd.read_parquet(
            "data/spam_sms/features/y_test.parquet"
        )["label"].values

    # --------------------------------------------------
    # Helpers
    # --------------------------------------------------
    @classmethod
    def _load_artifact(cls, model_name, artifact_name):
        versions = cls.client.get_latest_versions(
            model_name, stages=["Production"]
        )
        run_id = versions[0].run_id
        path = cls.client.download_artifacts(run_id, artifact_name)
        with open(path, "rb") as f:
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
    # Signature tests
    # --------------------------------------------------
    def test_movie_genre_signature(self):
        text = ["A detective investigates a mysterious murder"]
        X = self.movie_vectorizer.transform(text)

        preds = self.movie_model.predict(X)

        self.assertEqual(len(preds), 1)

    def test_spam_sms_signature(self):
        text = ["Congratulations! You won a free prize"]
        seq = self.spam_tokenizer.texts_to_sequences(text)
        pad = tf.keras.preprocessing.sequence.pad_sequences(
            seq, maxlen=self.spam_X.shape[1]
        )

        preds = self.spam_model.predict(pad)
        self.assertEqual(preds.shape[0], 1)

    # --------------------------------------------------
    # Performance tests
    # --------------------------------------------------
    def test_movie_genre_performance(self):
        X = self.movie_test.iloc[:, :-1].values
        y = self.movie_test.iloc[:, -1].values

        preds = self.movie_model.predict(X)
        f1 = f1_score(y, preds, average="macro")

        self.assertGreaterEqual(
            f1, 0.40, "Movie genre F1_macro below threshold"
        )

    def test_spam_sms_performance(self):
        probs = self.spam_model.predict(self.spam_X).reshape(-1)
        preds = (probs >= 0.7).astype(int)

        acc = accuracy_score(self.spam_y, preds)
        f1 = f1_score(self.spam_y, preds)

        self.assertGreaterEqual(acc, 0.85)
        self.assertGreaterEqual(f1, 0.85)


if __name__ == "__main__":
    unittest.main()
