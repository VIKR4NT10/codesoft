import unittest
from unittest.mock import patch
from flask_app.app import app

class FlaskAppTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        app.testing = True
        cls.client = app.test_client()

    # --------------------------------------------------
    # Home / health check
    # --------------------------------------------------
    def test_home_page(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)

    # --------------------------------------------------
    # Spam SMS prediction
    # --------------------------------------------------
    @patch("flask_app.app.ensure_models_loaded")
    @patch("flask_app.app.model_manager.predict_spam_sms")
    def test_spam_sms_prediction(self, mock_predict_spam, mock_ensure_loaded):
        mock_ensure_loaded.return_value = None
        mock_predict_spam.return_value = ("SPAM", 0.99)

        response = self.client.post(
            "/predict/spam_sms",
            data=dict(text="Congratulations! You have won a free prize"),
        )

        self.assertEqual(response.status_code, 200)

        data = response.get_json()
        self.assertEqual(data["prediction"], "SPAM")
        self.assertAlmostEqual(data["probability"], 0.99, places=2)


    # --------------------------------------------------
    # Movie genre prediction
    # --------------------------------------------------
    @patch("flask_app.app.ensure_models_loaded")
    @patch("flask_app.app.model_manager.predict_movie_genre")
    def test_movie_genre_prediction(self, mock_predict_movie, mock_ensure_loaded):
        mock_ensure_loaded.return_value = None
        mock_predict_movie.return_value = "Action"

        response = self.client.post(
            "/predict/movie_genre",
            data=dict(
                text="A young wizard discovers his magical powers and attends a school of magic"
            ),
        )

        self.assertEqual(response.status_code, 200)

        data = response.get_json()
        self.assertEqual(data["prediction"], "Action")


if __name__ == "__main__":
    unittest.main()
