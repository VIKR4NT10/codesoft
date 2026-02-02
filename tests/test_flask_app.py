import unittest
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
    def test_spam_sms_prediction(self):
        response = self.client.post(
            "/predict/spam_sms",
            data=dict(text="Congratulations! You have won a free prize"),
        )

        self.assertEqual(response.status_code, 200)

        data = response.get_json()
        self.assertIn(data["prediction"], ["SPAM", "HAM"])


    # --------------------------------------------------
    # Movie genre prediction
    # --------------------------------------------------
    def test_movie_genre_prediction(self):
        response = self.client.post(
            "/predict/movie_genre",
            data=dict(
                text="A young wizard discovers his magical powers and attends a school of magic"
            ),
        )

        self.assertEqual(response.status_code, 200)

        data = response.get_json()
        self.assertTrue(len(data["prediction"]) > 0)

if __name__ == "__main__":
    unittest.main()
