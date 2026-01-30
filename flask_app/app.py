from flask import Flask, render_template, request, jsonify
import mlflow
import pickle
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from prometheus_client import Counter, Histogram, generate_latest, CollectorRegistry, CONTENT_TYPE_LATEST
import time
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import warnings
import dagshub
from dotenv import load_dotenv
from tensorflow.keras.preprocessing.text import tokenizer_from_json

warnings.filterwarnings("ignore")
load_dotenv()

# Download NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Initialize Flask app
app = Flask(__name__)

# ==================== PROMETHEUS METRICS ====================
registry = CollectorRegistry()

# Request metrics
REQUEST_COUNT = Counter(
    "app_request_count", 
    "Total number of requests to the app", 
    ["method", "endpoint", "model"],
    registry=registry
)

REQUEST_LATENCY = Histogram(
    "app_request_latency_seconds", 
    "Latency of requests in seconds", 
    ["endpoint", "model"],
    registry=registry
)

PREDICTION_COUNT = Counter(
    "model_prediction_count", 
    "Count of predictions for each model and class", 
    ["model", "prediction"],
    registry=registry
)

# ==================== TEXT PREPROCESSING ====================
class TextPreprocessor:
    """Unified text preprocessing for both models"""
    
    def __init__(self):
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()
    
    def clean_text(self, text, for_spam=False):
        """Clean text with model-specific adjustments"""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs (especially important for spam)
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # For spam detection, we keep stopwords (they can be indicators)
        if not for_spam:
            words = [self.lemmatizer.lemmatize(word) 
                    for word in text.split() 
                    if word not in self.stop_words]
            text = ' '.join(words)
        
        return text

preprocessor = TextPreprocessor()

# ==================== MODEL LOADING ====================
class ModelManager:
    """Manages loading and prediction for both models"""
    
    def __init__(self):
        self.models = {}
        self.vectorizers = {}
        self.loaded = False
    
    def load_models(self):
        """Load both models and their vectorizers"""
        try:
            # ========== MOVIE GENRE MODEL ==========
            print("Loading Movie Genre Model...")
            
            # Load TF-IDF vectorizer
            with open('artifacts/movie_genre/vectorizer.pkl', 'rb') as f:
                self.vectorizers['movie_genre'] = pickle.load(f)
            
            # Load SVM model
            with open('artifacts/movie_genre/model.pkl', 'rb') as f:
                self.models['movie_genre'] = pickle.load(f)
            
            # ========== SPAM SMS MODEL ==========
            print("Loading Spam SMS Model...")
            
            # ========== SPAM SMS MODEL ==========
            print("Loading Spam SMS Model...")

            # Load tokenizer from JSON
            with open("artifacts/spam_sms/tokenizer.json", "r") as f:
                self.vectorizers["spam_sms"] = tokenizer_from_json(f.read())
                # Load TensorFlow SavedModel
            self.models["spam_sms"] = tf.keras.models.load_model(
                "artifacts/spam_sms/model"
                )

            
            self.loaded = True
            print("✓ All models loaded successfully!")
            
        except Exception as e:
            print(f"Error loading models: {e}")
            raise
    
    def predict_movie_genre(self, text):
        """Predict movie genre from text"""
        # Preprocess
        cleaned_text = preprocessor.clean_text(text, for_spam=False)
        
        # Vectorize
        features = self.vectorizers['movie_genre'].transform([cleaned_text])
        
        # Predict
        prediction = self.models['movie_genre'].predict(features)[0]
        
        return prediction
    
    def predict_spam_sms(self, text):
        """Predict if SMS is spam or ham"""
        # Preprocess (keep stopwords for spam detection)
        cleaned_text = preprocessor.clean_text(text, for_spam=True)
        
        # Tokenize and pad
        tokenizer = self.vectorizers['spam_sms']
        sequence = tokenizer.texts_to_sequences([cleaned_text])
        padded = tf.keras.preprocessing.sequence.pad_sequences(
            sequence, 
            maxlen=100,  # From params.yaml
            padding='post'
        )
        
        # Predict
        prediction_prob = self.models['spam_sms'].predict(padded, verbose=0)[0][0]
        prediction = 1 if prediction_prob > 0.5 else 0
        
        return prediction, prediction_prob

# Initialize model manager
model_manager = ModelManager()

# ==================== MLFLOW SETUP ====================
def setup_mlflow():
    """Setup MLflow tracking with DagsHub"""
    try:
        dagshub_token = os.getenv("DAGSHUB_TOKEN")
        if dagshub_token:
            os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
            os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
        
        # Set tracking URI
        mlflow.set_tracking_uri("https://dagshub.com/VIKR4NT10/capstone-project.mlflow")
        print("✓ MLflow tracking setup complete")
    except Exception as e:
        print(f"MLflow setup warning: {e}")

# ==================== ROUTES ====================
@app.route("/")
def home():
    """Home page with model selection"""
    REQUEST_COUNT.labels(method="GET", endpoint="/", model="home").inc()
    start_time = time.time()
    
    response = render_template("index.html")
    
    REQUEST_LATENCY.labels(endpoint="/", model="home").observe(time.time() - start_time)
    return response

@app.route("/movie_genre")
def movie_genre_page():
    """Movie genre classification page"""
    return render_template("movie_genre.html")

@app.route("/spam_sms")
def spam_sms_page():
    """Spam SMS classification page"""
    return render_template("spam_sms.html")

@app.route("/predict/movie_genre", methods=["POST"])
def predict_movie_genre():
    """Predict movie genre endpoint"""
    REQUEST_COUNT.labels(method="POST", endpoint="/predict/movie_genre", model="movie_genre").inc()
    start_time = time.time()
    
    try:
        text = request.form.get("text", "").strip()
        
        if not text:
            return jsonify({"error": "No text provided"}), 400
        
        print(f"📝 Received text for movie genre: {text[:50]}...")
        
        # Check if model is loaded
        if not model_manager.loaded:
            return jsonify({
                "error": "Models not loaded",
                "prediction": "DEMO: Action",  # Demo response
                "text": text[:100] + "..." if len(text) > 100 else text,
                "latency": 0.1
            })
        
        # Predict
        prediction = model_manager.predict_movie_genre(text)
        print(f"🎬 Predicted genre: {prediction}")
        
        # Update metrics
        PREDICTION_COUNT.labels(model="movie_genre", prediction=prediction).inc()
        
        # Measure latency
        latency = time.time() - start_time
        REQUEST_LATENCY.labels(endpoint="/predict/movie_genre", model="movie_genre").observe(latency)
        
        # Log prediction to MLflow
        try:
            with mlflow.start_run(run_name="movie_genre_prediction"):
                mlflow.log_param("text_length", len(text))
                mlflow.log_param("prediction", prediction)
                mlflow.log_metric("prediction_latency", latency)
        except Exception as e:
            print(f"⚠️ MLflow logging warning: {e}")
        
        return jsonify({
            "success": True,
            "prediction": str(prediction),
            "text": text[:100] + "..." if len(text) > 100 else text,
            "latency": round(latency, 3)
        })
        
    except Exception as e:
        print(f"❌ Error in movie genre prediction: {e}")
        return jsonify({
            "error": str(e),
            "success": False
        }), 500

@app.route("/predict/spam_sms", methods=["POST"])
def predict_spam_sms():
    """Predict spam SMS endpoint"""
    REQUEST_COUNT.labels(method="POST", endpoint="/predict/spam_sms", model="spam_sms").inc()
    start_time = time.time()
    
    try:
        text = request.form.get("text", "").strip()
        
        if not text:
            return jsonify({"error": "No text provided"}), 400
        
        print(f"📝 Received text for spam check: {text[:50]}...")
        
        # Check if model is loaded
        if not model_manager.loaded:
            # Return demo response
            is_spam_demo = "win" in text.lower() or "free" in text.lower() or "click" in text.lower()
            return jsonify({
                "error": "Models not loaded - DEMO MODE",
                "prediction": "SPAM" if is_spam_demo else "HAM",
                "probability": 0.85 if is_spam_demo else 0.15,
                "text": text[:100] + "..." if len(text) > 100 else text,
                "latency": 0.1
            })
        
        # Predict
        prediction, probability = model_manager.predict_spam_sms(text)
        
        # Map to labels
        label = "SPAM" if prediction == 1 else "HAM"
        print(f"📱 Predicted: {label} (probability: {probability:.3f})")
        
        # Update metrics
        PREDICTION_COUNT.labels(model="spam_sms", prediction=label).inc()
        
        # Measure latency
        latency = time.time() - start_time
        REQUEST_LATENCY.labels(endpoint="/predict/spam_sms", model="spam_sms").observe(latency)
        
        # Log prediction to MLflow
        try:
            with mlflow.start_run(run_name="spam_sms_prediction"):
                mlflow.log_param("text_length", len(text))
                mlflow.log_param("prediction", label)
                mlflow.log_metric("probability", float(probability))
                mlflow.log_metric("prediction_latency", latency)
        except Exception as e:
            print(f"⚠️ MLflow logging warning: {e}")
        
        return jsonify({
            "success": True,
            "prediction": label,
            "probability": float(probability),
            "text": text[:100] + "..." if len(text) > 100 else text,
            "latency": round(latency, 3)
        })
        
    except Exception as e:
        print(f"❌ Error in spam prediction: {e}")
        return jsonify({
            "error": str(e),
            "success": False
        }), 500

@app.route("/metrics")
def metrics():
    """Prometheus metrics endpoint"""
    return generate_latest(registry), 200, {"Content-Type": CONTENT_TYPE_LATEST}

@app.route("/health")
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "models_loaded": model_manager.loaded,
        "models_available": list(model_manager.models.keys()) if model_manager.loaded else []
    })

# ==================== INITIALIZATION ====================
if __name__ == "__main__":
    print("=" * 50)
    print("Starting Multi-Model Flask Application")
    print("=" * 50)
    
    # Setup MLflow
    setup_mlflow()
    
    # Load models
    try:
        model_manager.load_models()
    except Exception as e:
        print(f"⚠️  Warning: Could not load models: {e}")
        print("⚠️  Running in demo mode without models")
    
    # Start Flask app
    app.run(debug=True, host="0.0.0.0", port=5000)