import os
import re
import string
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from logger import logging

# Download required resources (safe to call multiple times)
nltk.download("stopwords")
nltk.download("wordnet")


def clean_text(text: str, stop_words, lemmatizer) -> str:
    """
    Clean and normalize a single text string
    """
    if pd.isna(text):
        return ""

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Lowercase
    text = text.lower()

    # Remove numbers
    text = re.sub(r"\d+", "", text)

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Remove stopwords + lemmatize
    words = [
        lemmatizer.lemmatize(word)
        for word in text.split()
        if word not in stop_words
    ]

    return " ".join(words)


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess movie genre dataset
    Expected columns:
    ID, TITLE, GENRE, DESCRIPTION
    """
    logging.info("Starting movie genre preprocessing")

    df = df.copy()

    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    # Combine TITLE and DESCRIPTION (optional but powerful)
    df["text"] = (
        df["TITLE"].fillna("") + " " + df["DESCRIPTION"].fillna("")
    )

    # Apply cleaning
    df["text"] = df["text"].apply(
        lambda x: clean_text(x, stop_words, lemmatizer)
    )

    # Drop empty texts
    df = df[df["text"].str.split().str.len() > 3]

    # Keep only required columns
    df = df[["ID", "text", "GENRE"]]

    logging.info("Movie genre preprocessing completed")
    return df


def main():
    try:
        raw_path = os.path.join("data", "movie_genre", "raw")
        processed_path = os.path.join("data", "movie_genre", "processed")
        os.makedirs(processed_path, exist_ok=True)

        # Load raw data
        train_df = pd.read_csv(os.path.join(raw_path, "train.csv"))
        test_df = pd.read_csv(os.path.join(raw_path, "test.csv"))
        logging.info("Raw data loaded successfully")

        # Preprocess
        train_processed = preprocess_dataframe(train_df)
        test_processed = preprocess_dataframe(test_df)

        # Save processed data
        train_processed.to_csv(
            os.path.join(processed_path, "train_processed.csv"),
            index=False
        )
        test_processed.to_csv(
            os.path.join(processed_path, "test_processed.csv"),
            index=False
        )

        logging.info("Processed data saved to %s", processed_path)

    except Exception as e:
        logging.error("Data preprocessing failed: %s", e)
        raise


if __name__ == "__main__":
    main()
