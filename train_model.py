"""Train and evaluate the ensemble on Fake.csv and True.csv or demo data.

Usage:
    python train_model.py
    python train_model.py --fake data/Fake.csv --true data/True.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from ml_engine import build_demo_corpus, clean_text, majority_vote


def read_dataset(fake_path: Path, true_path: Path):
    texts, labels = [], []
    for path, label in ((fake_path, "FAKE"), (true_path, "REAL")):
        with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                combined = " ".join([row.get("title", ""), row.get("text", "")]).strip()
                if combined:
                    texts.append(combined)
                    labels.append(label)
    return texts, labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fake", type=Path, default=Path("data/Fake.csv"))
    parser.add_argument("--true", type=Path, default=Path("data/True.csv"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/metrics.json"))
    args = parser.parse_args()

    if args.fake.exists() and args.true.exists():
        texts, labels = read_dataset(args.fake, args.true)
        dataset_name = f"{args.fake.name} + {args.true.name}"
    else:
        texts, labels = build_demo_corpus()
        dataset_name = "Bundled balanced classroom demonstration corpus"

    x_train, x_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.20, random_state=42, stratify=labels
    )
    vectorizer = TfidfVectorizer(
        preprocessor=clean_text,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=25000,
        sublinear_tf=True,
    )
    train_matrix = vectorizer.fit_transform(x_train)
    test_matrix = vectorizer.transform(x_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1200, class_weight="balanced", random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5, metric="cosine", algorithm="brute", weights="distance"),
        "Decision Tree": DecisionTreeClassifier(max_depth=18, min_samples_leaf=2, class_weight="balanced", random_state=42),
    }
    predictions = {}
    accuracy = {}
    for name, model in models.items():
        model.fit(train_matrix, y_train)
        predictions[name] = model.predict(test_matrix)
        accuracy[name] = float(accuracy_score(y_test, predictions[name]))

    ensemble = np.array([
        majority_vote(predictions[name][i] for name in predictions)
        for i in range(len(y_test))
    ])
    payload = {
        "dataset": dataset_name,
        "samples": len(texts),
        "training_samples": len(x_train),
        "testing_samples": len(x_test),
        "split": "80/20 stratified, random_state=42",
        "model_accuracy": accuracy,
        "ensemble_accuracy": float(accuracy_score(y_test, ensemble)),
        "confusion_matrix_labels": ["FAKE", "REAL"],
        "confusion_matrix": confusion_matrix(y_test, ensemble, labels=["FAKE", "REAL"]).tolist(),
        "classification_report": classification_report(y_test, ensemble, labels=["FAKE", "REAL"], output_dict=True, zero_division=0),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

