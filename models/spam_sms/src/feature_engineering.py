import logging
import pickle
import yaml
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

logging.basicConfig(level=logging.INFO)


# ----------------------------
# Load preprocessed data
# ----------------------------
def load_preprocessed(processed_dir: Path, filename: str = "preprocessed.parquet") -> pd.DataFrame:
    path = processed_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Preprocessed data not found at {path}")
    return pd.read_parquet(path)


# ----------------------------
# Add text-based features
# ----------------------------
def add_text_features(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("Adding text-based features")
    df["char_len"] = df["text"].apply(len)
    df["word_len"] = df["text"].apply(lambda x: len(x.split()))
    df["num_exclamations"] = df["text"].apply(lambda x: x.count("!"))
    df["num_caps"] = df["text"].apply(lambda x: sum(1 for c in x if c.isupper()))
    df["has_url"] = df["text"].str.contains("URL").astype(int)
    return df

def normalize_text(text: str) -> str:
    text = text.lower()
    text = text.strip()
    return text

# ----------------------------
# Train/test split
# ----------------------------
def split_data(df: pd.DataFrame, test_size: float, random_state: int):
    df["text"] = df["text"].astype(str).apply(normalize_text)
    X = df["text"]
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logging.info("Data split into train/test")
    return X_train, X_test, y_train, y_test


# ----------------------------
# Tokenization and padding
# ----------------------------
def tokenize_and_pad(X_train, X_test, vocab_size=10000, max_len=100):
    logging.info("Tokenizing and padding text data")
    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)

    X_train_pad = pad_sequences(
        tokenizer.texts_to_sequences(X_train),
        maxlen=max_len,
        padding="post"
    )
    X_test_pad = pad_sequences(
        tokenizer.texts_to_sequences(X_test),
        maxlen=max_len,
        padding="post",
        truncating="post"
    )

    logging.info("Tokenization and padding completed")
    return tokenizer, X_train_pad, X_test_pad


# ----------------------------
# Save tokenizer to artifacts
# ----------------------------
def save_tokenizer(tokenizer, artifacts_dir: Path, filename: str = "vectorizer.pkl"):
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    path = artifacts_dir / filename
    with open(path, "wb") as f:
        tokenizer_json = tokenizer.to_json()
        with open(artifacts_dir / "tokenizer.json", "w") as f:
            f.write(tokenizer_json)

    logging.info(f"Tokenizer saved at {path}")


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    try:
        # ----------------------------
        # Load parameters
        # ----------------------------
        with open("params.yaml") as f:
            config = yaml.safe_load(f)

        MODEL_NAME = "spam_sms"
        model_cfg = config["models"][MODEL_NAME]
        vocab_size = model_cfg["feature_engineering"]["vocab_size"]
        max_len = model_cfg["feature_engineering"]["max_len"]
        test_size = model_cfg["feature_engineering"]["test_size"]
        random_state = model_cfg["feature_engineering"]["random_state"]

        # ----------------------------
        # Paths
        # ----------------------------
        processed_dir = Path(f"data/{MODEL_NAME}/processed")
        artifacts_dir = Path(f"artifacts/{MODEL_NAME}")
        features_dir = Path(f"data/{MODEL_NAME}/features")
        features_dir.mkdir(parents=True, exist_ok=True)

        # ----------------------------
        # Load preprocessed data
        # ----------------------------
        df = load_preprocessed(processed_dir)

        if df.empty:
            logging.warning("No data found in processed directory. Exiting.")
        else:
            # ----------------------------
            # Feature engineering
            # ----------------------------
            df = add_text_features(df)

            # ----------------------------
            # Split
            # ----------------------------
            X_train, X_test, y_train, y_test = split_data(df, test_size=test_size, random_state=random_state)

            # ----------------------------
            # Tokenize + pad
            # ----------------------------
            tokenizer, X_train_pad, X_test_pad = tokenize_and_pad(X_train, X_test, vocab_size=vocab_size, max_len=max_len)

            # ----------------------------
            # Save artifacts
            # ----------------------------
            save_tokenizer(tokenizer, artifacts_dir)

            # ----------------------------
            # Save padded sequences and labels
            # ----------------------------
            pd.DataFrame(X_train_pad).to_parquet(features_dir / "X_train_pad.parquet", index=False)
            pd.DataFrame(X_test_pad).to_parquet(features_dir / "X_test_pad.parquet", index=False)
            pd.DataFrame(y_train).to_parquet(features_dir / "y_train.parquet", index=False)
            pd.DataFrame(y_test).to_parquet(features_dir / "y_test.parquet", index=False)

            logging.info("Feature engineering completed successfully")

    except Exception as e:
        logging.exception(f"Feature engineering failed: {e}")
        raise
