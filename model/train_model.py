from __future__ import annotations

"""
Simple training script for the text similarity model.

This trains a TF-IDF vectorizer on a small corpus of generic
news-like text so we can later compute cosine similarity between
the user's query (e.g. "cricket") and article texts.

Run once to create model/model.pkl:

    python -m model.train_model
"""

from pathlib import Path
from typing import List

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer


def build_corpus() -> List[str]:
    # Small generic corpus; in a real project this would come from
    # a larger labelled dataset.
    texts = [
        "cricket match score team tournament bowler batsman run wicket over pitch",
        "football league match goal striker goalkeeper defender stadium",
        "technology company launches new smartphone with camera and processor",
        "stock market indices trade business finance economy shares investors",
        "health research vaccine medical study hospital doctor treatment",
        "science space mission satellite rocket nasa isro astronomy planets",
        "entertainment movie release actor actress music album trailer",
        "politics election government policy parliament minister",
        "artificial intelligence machine learning neural network data model training",
    ]
    return texts


def train_and_save_model(model_path: Path) -> None:
    corpus = build_corpus()
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000, ngram_range=(1, 2))
    vectorizer.fit(corpus)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"vectorizer": vectorizer}, model_path)
    print(f"Saved TF-IDF model to {model_path}")


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    model_file = here / "model.pkl"
    train_and_save_model(model_file)

