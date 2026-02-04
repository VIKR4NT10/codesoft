from flask import Flask, render_template, request, jsonify
import mlflow
from mlflow.tracking import MlflowClient
import os
import tensorflow as tf
import pandas as pd
import time
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from prometheus_client import Counter, Histogram, generate_latest, CollectorRegistry, CONTENT_TYPE_LATEST
import dagshub
import warnings
from dotenv import load_dotenv
import re
warnings.filterwarnings("ignore")
load_dotenv()
from threading import Lock
_model_lock = Lock()
# ==================== NLTK SETUP ====================
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

# ==================== FLASK APP ====================
app = Flask(__name__)

# ==================== PROMETHEUS METRICS ====================
registry = CollectorRegistry()

REQUEST_COUNT = Counter(
    "app_request_count",
    "Total number of requests",
    ["method", "endpoint", "model"],
    registry=registry,
)

REQUEST_LATENCY = Histogram(
    "app_request_latency_seconds",
    "Request latency",
    ["endpoint", "model"],
    registry=registry,
)

PREDICTION_COUNT = Counter(
    "model_prediction_count",
    "Prediction count",
    ["model", "prediction"],
    registry=registry,
)

# # ==================== TEXT PREPROCESSING ====================
# class TextPreprocessor:
#     def __init__(self):
#         self.stop_words = set(stopwords.words("english"))
#         self.lemmatizer = WordNetLemmatizer()

#     def clean_text(self, text, for_spam=False):
#         if not isinstance(text, str):
#             return ""

#         text = text.lower()
#         text = re.sub(r"https?://\S+|www\.\S+", "", text)
#         text = re.sub(r"\d+", "", text)
#         text = text.translate(str.maketrans("", "", string.punctuation))
#         text = re.sub(r"\s+", " ", text).strip()

#         if not for_spam:
#             words = [
#                 self.lemmatizer.lemmatize(w)
#                 for w in text.split()
#                 if w not in self.stop_words
#             ]
#             text = " ".join(words)

#         return text


# preprocessor = TextPreprocessor()
# ================= MOVIE GENRE PREPROCESSING =================

_movie_stop_words = set(stopwords.words("english"))
_movie_lemmatizer = WordNetLemmatizer()

def preprocess_movie_genre(text: str) -> str:
    """
    User must include movie title + description in one text field.
    """
    if not isinstance(text, str) or text.strip() == "":
        return ""

    text = re.sub(r"http\S+|www\S+", "", text)
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    words = [
        _movie_lemmatizer.lemmatize(word)
        for word in text.split()
        if word not in _movie_stop_words
    ]

    return " ".join(words)


# ================= SMS SPAM PREPROCESSING =================

def preprocess_sms_spam(text: str) -> str:

    if not isinstance(text, str) or text.strip() == "":
        return ""
    text = text.lower()

    # replace urls, emails, numbers
    text = re.sub(r"http\S+|www\S+", " URL ", text)
    text = re.sub(r"\S+@\S+", " EMAIL ", text)
    text = re.sub(r"\d+", " NUMBER ", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = text.lower().strip()
    return text


# ==================== MLFLOW SETUP ====================
def setup_mlflow():
    dagshub_token = os.getenv("CODESOFT")
    if dagshub_token:
        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

    mlflow.set_tracking_uri("https://dagshub.com/VIKR4NT10/codesoft.mlflow" )
    
    # mlflow.set_tracking_uri("https://dagshub.com/VIKR4NT10/codesoft.mlflow")
    # dagshub.init(repo_owner="VIKR4NT10", repo_name="codesoft", mlflow=True)

def get_production_model_uri(model_name: str) -> str:
    client = MlflowClient()
    versions = client.get_latest_versions(model_name, stages=["Production"])
    if not versions:
        raise RuntimeError(f"No Production model found for {model_name}")
    return f"models:/{model_name}/{versions[0].version}"

def ensure_models_loaded():
    if not model_manager.loaded:
        with _model_lock:
            if not model_manager.loaded:
                setup_mlflow()
                model_manager.load_models()
# ==================== MODEL MANAGER ====================
class ModelManager:
    def __init__(self):
        self.models = {}
        self.loaded = False
        self.tokenizers = {}
    def load_models(self):
        print("Loading models from MLflow Registry (Production)...")

        # ---------- Movie Genre ----------
        movie_uri = get_production_model_uri("movie_genre_svm")
        self.models["movie_genre"] = mlflow.pyfunc.load_model(movie_uri)

        # ---------- Spam SMS ----------
        spam_uri = get_production_model_uri("spam_sms_cnn")
        self.models["spam_sms"] = mlflow.tensorflow.load_model(spam_uri)

        
        tokenizer_path = mlflow.artifacts.download_artifacts(
        artifact_uri=f"{spam_uri}/tokenizer.json"
        )
        
        with open(tokenizer_path, "r") as f:
            self.tokenizers["spam_sms"] = tf.keras.preprocessing.text.tokenizer_from_json(f.read()
            )


        self.loaded = True
        print("✓ All Production models loaded")

    def predict_movie_genre(self, text):
        cleaned = preprocess_movie_genre(text)
        return self.models["movie_genre"].predict([cleaned])[0]

    def predict_spam_sms(self, text):
        cleaned = preprocess_sms_spam(text)

        tokenizer = self.tokenizers["spam_sms"]
        model = self.models["spam_sms"]

        # text → sequence
        seq = tokenizer.texts_to_sequences([cleaned])
        padded = tf.keras.preprocessing.sequence.pad_sequences(
            seq,
            maxlen=100,     
            padding="post"
        )
        
        prob = float(model.predict(padded, verbose=0)[0][0])

        label = "SPAM" if prob > 0.5 else "HAM"
        return label, prob



model_manager = ModelManager()

# ==================== ROUTES ====================

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/movie_genre")
def movie_genre_page():
    return render_template("movie_genre.html")


@app.route("/spam_sms")
def spam_sms_page():
    return render_template("spam_sms.html")


# -------------------- Movie Genre Prediction --------------------
@app.route("/predict/movie_genre", methods=["POST"])
def predict_movie_genre():
    ensure_models_loaded()
    start = time.time()

    text = request.form.get("text", "")
    prediction = model_manager.predict_movie_genre(text)

    latency = time.time() - start

    REQUEST_COUNT.labels(
        method="POST",
        endpoint="/predict/movie_genre",
        model="movie_genre",
    ).inc()

    REQUEST_LATENCY.labels(
        "/predict/movie_genre",
        "movie_genre",
    ).observe(latency)

    PREDICTION_COUNT.labels(
        "movie_genre",
        prediction,
    ).inc()

    return jsonify(
        {
            "prediction": prediction,
            "latency": round(latency, 3),
        }
    )


# -------------------- Spam SMS Prediction --------------------
@app.route("/predict/spam_sms", methods=["POST"])
def predict_spam_sms():
    ensure_models_loaded()
    start = time.time()

    text = request.form.get("text", "")
    label, prob = model_manager.predict_spam_sms(text)

    latency = time.time() - start

    REQUEST_COUNT.labels(
        method="POST",
        endpoint="/predict/spam_sms",
        model="spam_sms",
    ).inc()

    REQUEST_LATENCY.labels(
        "/predict/spam_sms",
        "spam_sms",
    ).observe(latency)

    PREDICTION_COUNT.labels(
        "spam_sms",
        label,
    ).inc()

    return jsonify(
        {
            "prediction": label,
            "probability": round(prob, 3),
            "latency": round(latency, 3),
        }
    )


# -------------------- Metrics --------------------
@app.route("/metrics")
def metrics():
    return generate_latest(registry), 200, {
        "Content-Type": CONTENT_TYPE_LATEST
    }


# -------------------- Health --------------------
@app.route("/health")
def health():
    return jsonify(
        {
            "status": "healthy",
            "models_loaded": model_manager.loaded,
        }
    ), 200


# ==================== MAIN ====================
if __name__ == "__main__":
    setup_mlflow()
    model_manager.load_models()
    app.run(host="0.0.0.0", port=5000, debug=True)
