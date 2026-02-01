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
            "/predict/spam-sms",
            data=dict(text="Congratulations! You have won a free prize"),
            follow_redirects=True
        )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(
            b"Spam" in response.data or b"Ham" in response.data,
            "Response should contain either 'Spam' or 'Ham'"
        )

    # --------------------------------------------------
    # Movie genre prediction
    # --------------------------------------------------
    def test_movie_genre_prediction(self):
        response = self.client.post(
            "/predict/movie-genre",
            data=dict(
                text="A young wizard discovers his magical powers and attends a school of magic"
            ),
            follow_redirects=True
        )

        self.assertEqual(response.status_code, 200)

        # We don't hardcode genres, just ensure something sensible is returned
        self.assertTrue(
            b"Genre" in response.data or b"Predicted" in response.data,
            "Response should contain a predicted genre"
        )


if __name__ == "__main__":
    unittest.main()
