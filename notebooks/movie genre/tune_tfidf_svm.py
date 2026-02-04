# ============================================================
# Optuna Hyperparameter Tuning: TF-IDF + Linear SVM
# ============================================================

import os
import re
import string
import warnings

import optuna
import numpy as np
import pandas as pd

import mlflow
import dagshub

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

warnings.filterwarnings("ignore")

# ========================== CONFIG ==========================
CONFIG = {
    "data_dir": r"C:\Users\DELL\Desktop\mlops\codesoft\notebooks\Genre Classification Dataset",
    "train_file": "train_data.txt",
    "subset_frac": 0.4,
    "random_state": 42,
    "n_trials": 30,
    "experiment_name": "Optuna_TFIDF_SVM",
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
    return " ".join(tokens)

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

    df = df.sample(frac=CONFIG["subset_frac"], random_state=CONFIG["random_state"])
    df["clean_text"] = df["text"].apply(clean_text)

    return df

# ========================== OPTUNA OBJECTIVE ==========================
def objective(trial):

    tfidf_max_features = trial.suggest_int("tfidf_max_features", 2000, 6000, step=1000)
    tfidf_min_df = trial.suggest_int("tfidf_min_df", 2, 5)
    tfidf_ngram = trial.suggest_categorical("tfidf_ngram_range", [(1, 1), (1, 2)])
    C = trial.suggest_float("C", 0.05, 3.0, log=True)

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=tfidf_max_features,
            min_df=tfidf_min_df,
            ngram_range=tfidf_ngram
        )),
        ("svm", LinearSVC(
            C=C,
            class_weight="balanced",
            loss="squared_hinge"
        ))
    ])

    scores = cross_val_score(
        pipeline,
        X,
        y,
        scoring="f1_weighted",
        cv=3,
        n_jobs=-1
    )

    return scores.mean()

# ========================== MAIN ==========================
def main():

    global X, y

    if mlflow.active_run():
        mlflow.end_run()

    df = load_data()

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["label"])
    X = df["clean_text"]

    study = optuna.create_study(
        direction="maximize",
        study_name="TFIDF_SVM_Study"
    )

    with mlflow.start_run(run_name="Optuna_TFIDF_SVM"):
        study.optimize(objective, n_trials=CONFIG["n_trials"])

        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_f1", study.best_value)

        print("\n Best F1:", study.best_value)
        print("Best Parameters:")
        for k, v in study.best_params.items():
            print(f"  {k}: {v}")

# ========================== ENTRY POINT ==========================
if __name__ == "__main__":
    main()
