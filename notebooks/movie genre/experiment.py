# ========================== IMPORTS ==========================
import os
import re
import string
import warnings

import numpy as np
import pandas as pd

import mlflow
import mlflow.sklearn
import dagshub

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

from xgboost import XGBClassifier
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

warnings.filterwarnings("ignore")

# ========================== CONFIG ==========================
CONFIG = {
    "data_dir": r"C:\Users\DELL\Desktop\mlops\codesoft\notebooks\Genre Classification Dataset",
    "train_file": "train_data.txt",
    "subset_frac": 0.2,
    "test_size": 0.2,
    "random_state": 42,
    "experiment_name": "Fast_Model_Selection",
    "mlflow_uri": "https://dagshub.com/VIKR4NT10/codesoft.mlflow",
    "repo_owner": "VIKR4NT10",
    "repo_name": "codesoft"
}

# ========================== MLflow + DAGsHub ==========================
mlflow.set_tracking_uri(CONFIG["mlflow_uri"])
dagshub.init(
    repo_owner=CONFIG["repo_owner"],
    repo_name=CONFIG["repo_name"],
    mlflow=True
)
mlflow.set_experiment(CONFIG["experiment_name"])

# ========================== TEXT PREPROCESSING ==========================
lemmatizer = WordNetLemmatizer()
STOPWORDS = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(f"[{string.punctuation}]", " ", text)
    tokens = [lemmatizer.lemmatize(w) for w in text.split() if w not in STOPWORDS]
    return tokens

# ========================== LOAD DATA ==========================
def load_data():
    path = os.path.join(CONFIG["data_dir"], CONFIG["train_file"])
    data = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(" ::: ")
            if len(parts) == 4:
                _, _, genre, text = parts
                data.append((text, genre))

    df = pd.DataFrame(data, columns=["text", "label"])

    if df.empty:
        raise ValueError("Loaded DataFrame is empty.")

    df = df.sample(frac=CONFIG["subset_frac"], random_state=CONFIG["random_state"])

    df["tokens"] = df["text"].apply(clean_text)
    df["clean_text"] = df["tokens"].apply(lambda x: " ".join(x))

    return df

# ========================== WORD2VEC HELPERS ==========================
def build_word2vec(sentences, vector_size=100):
    return Word2Vec(sentences, vector_size=vector_size, min_count=3, workers=2)

def sentence_vector(tokens, model):
    vectors = [model.wv[w] for w in tokens if w in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

# ========================== MODELS ==========================
MODELS = {
    "NaiveBayes": MultinomialNB(alpha=0.1),
    "LogisticRegression": LogisticRegression(max_iter=300, n_jobs=1, class_weight="balanced"),
    "SVM": LinearSVC(C=1.0, class_weight="balanced"),
    "XGBoost": XGBClassifier(
        objective="multi:softmax",
        eval_metric="mlogloss",
        tree_method="hist",
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        n_jobs=2
    )
}

# ========================== TRAINING & EVALUATION ==========================
def run_experiments(df):

    # 🔑 Encode labels ONCE
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["label"])

    with mlflow.start_run(run_name="Parent_Subset_Run"):

        # ================= TF-IDF =================
        tfidf = TfidfVectorizer(
            max_features=3000,
            ngram_range=(1, 2),
            min_df=3
        )

        X_tfidf = tfidf.fit_transform(df["clean_text"])

        X_train, X_test, y_train, y_test = train_test_split(
            X_tfidf,
            y,
            test_size=CONFIG["test_size"],
            random_state=CONFIG["random_state"],
            stratify=y
        )

        for name, model in MODELS.items():
            with mlflow.start_run(run_name=f"{name}_TFIDF", nested=True):
                model.fit(X_train, y_train)
                preds = model.predict(X_test)

                mlflow.log_params({"vectorizer": "TF-IDF", "model": name})
                mlflow.log_metrics({
                    "accuracy": accuracy_score(y_test, preds),
                    "f1": f1_score(y_test, preds, average="weighted")
                })

        # ================= WORD2VEC =================
        w2v_model = build_word2vec(df["tokens"], vector_size=100)

        X_w2v = np.vstack(
            df["tokens"].apply(lambda x: sentence_vector(x, w2v_model))
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X_w2v,
            y,
            test_size=CONFIG["test_size"],
            random_state=CONFIG["random_state"],
            stratify=y
        )

        for name, model in MODELS.items():
            if name == "NaiveBayes":
                continue  # NB doesn't work with dense Word2Vec

            with mlflow.start_run(run_name=f"{name}_Word2Vec", nested=True):
                model.fit(X_train, y_train)
                preds = model.predict(X_test)

                f1 = f1_score(y_test, preds, average="weighted")

                mlflow.log_params({"vectorizer": "Word2Vec", "model": name})
                mlflow.log_metrics({
                    "accuracy": accuracy_score(y_test, preds),
                    "f1": f1
                })

                print(f"{name}_Word2Vec -> F1: {f1:.4f}")

# ========================== MAIN ==========================
if __name__ == "__main__":
    df = load_data()
    print("Dataset loaded and preprocessed.")
    run_experiments(df)
