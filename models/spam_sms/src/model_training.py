import logging
import yaml
import numpy as np
from pathlib import Path
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout, Concatenate, Input)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import pandas as pd
import pickle

logging.basicConfig(level=logging.INFO)


# ----------------------------
# Load data
# ----------------------------
def load_data(processed_dir: Path):
    X_train = pd.read_parquet(processed_dir / "X_train_pad.parquet").values
    X_test = pd.read_parquet(processed_dir / "X_test_pad.parquet").values
    y_train = pd.read_parquet(processed_dir / "y_train.parquet")["label"].values
    y_test = pd.read_parquet(processed_dir / "y_test.parquet")["label"].values
    return X_train, X_test, y_train, y_test


# ----------------------------
# Load tokenizer'''inference-only'''
# ----------------------------
def load_tokenizer(artifacts_dir: Path):
    tokenizer_path = artifacts_dir / "tokenizer.json"
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")

    with open(tokenizer_path) as f:
        tokenizer_json = f.read()

    tokenizer = tokenizer_from_json(tokenizer_json)
    return tokenizer


# ----------------------------
# CNN model builder with multiple kernel sizes
# ----------------------------
def build_cnn_model(vocab_size, embed_dim, filters, kernel_sizes, dropout_rate, l2_reg, learning_rate, max_len):
    inputs = Input(shape=(max_len,))
    embedding = Embedding(vocab_size, embed_dim)(inputs)

    convs = []
    for ks in kernel_sizes:
        conv = Conv1D(filters=filters, kernel_size=ks, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(embedding)
        pool = GlobalMaxPooling1D()(conv)
        convs.append(pool)

    if len(convs) > 1:
        x = Concatenate()(convs)
    else:
        x = convs[0]

    x = Dropout(dropout_rate)(x)
    x = Dense(64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(x)
    outputs = Dense(1, activation="sigmoid")(x)

    model = Model(inputs, outputs)
    model.compile(
        loss="binary_crossentropy",
        optimizer=Adam(learning_rate=learning_rate),
        metrics=["accuracy"]
    )
    return model


# ----------------------------
# Compute class weights
# ----------------------------
def get_class_weights(y_train):
    classes = np.unique(y_train)
    class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    return dict(zip(classes, class_weights))


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    try:
        # Load parameters
        with open("params.yaml") as f:
            config = yaml.safe_load(f)

        MODEL_NAME = "spam_sms"
        model_cfg = config["models"][MODEL_NAME]["training"]
        feat_cfg = config["models"][MODEL_NAME]["feature_engineering"]

        processed_dir = Path(f"data/{MODEL_NAME}/features")
        artifacts_dir = Path(f"artifacts/{MODEL_NAME}")
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Load data
        X_train, X_test, y_train, y_test = load_data(processed_dir)

        # Load tokenizer (optional: needed if embedding layer weights are pretrained)
        tokenizer = load_tokenizer(artifacts_dir)
        vocab_size = len(tokenizer.word_index) + 1
        max_len = feat_cfg.get("max_len", 100)

        # Build model
        model = build_cnn_model(
            vocab_size=vocab_size,
            embed_dim=model_cfg.get("embed_dim", 128),
            filters=model_cfg.get("filters", 128),
            kernel_sizes=model_cfg.get("kernel_sizes", [3,5,7]),
            dropout_rate=model_cfg.get("dropout_rate", 0.5),
            l2_reg=model_cfg.get("l2_reg", 1e-4),
            learning_rate=model_cfg.get("learning_rate", 1e-3),
            max_len=max_len
        )

        model.summary()

        # Class imbalance handling
        class_weights = get_class_weights(y_train)
        logging.info(f"Class weights: {class_weights}")

        # Train
        history = model.fit(
            X_train, y_train,
            epochs=model_cfg.get("epochs", 10),
            batch_size=model_cfg.get("batch_size", 64),
            validation_split=model_cfg.get("val_split", 0.1),
            class_weight=class_weights,
            verbose=1
        )

        # Save trained model
        # model_path = artifacts_dir / "model.h5"
        model.save(artifacts_dir / "model", save_format="tf")
        logging.info(f"Trained model saved at {artifacts_dir / 'model'}")

    except Exception as e:
        logging.exception(f"Model training failed: {e}")
        raise
