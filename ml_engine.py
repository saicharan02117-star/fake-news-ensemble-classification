"""Machine-learning engine for the fake-news classroom project.

The deployed demo uses a bundled, balanced classroom corpus so it can run
without downloading external data. For the final lab experiment, use
train_model.py with Fake.csv and True.csv from the project dataset.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Iterable, List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


REAL_ARTICLES = [
    "The city council approved the annual budget after a public meeting on Tuesday. The finance department published the complete expenditure table on its official website.",
    "The university announced the semester examination schedule in a notice signed by the controller of examinations. Students can download the timetable from the college portal.",
    "The health department opened a vaccination centre at the district hospital. Officials said eligible residents may register through the government portal.",
    "The weather office forecast moderate rain in coastal districts and advised fishermen to check local warnings before sailing.",
    "The central bank kept the policy interest rate unchanged after its scheduled review meeting, according to the published policy statement.",
    "Researchers reported the results of a peer reviewed study on crop disease resistance. The paper describes the sample size, method and limitations.",
    "The railway authority added two special trains for the festival period. The notification lists train numbers, routes and operating dates.",
    "The district administration declared a school holiday because of heavy rainfall. The signed order applies only to the affected mandals.",
    "A technology company released its quarterly earnings report. Revenue and operating costs were included in the regulatory filing.",
    "The state agriculture department issued an advisory on pest management for cotton farmers after field officers observed an increase in infestation.",
    "The court published its judgment in the official case database. The order explains the evidence and the legal reasons for the decision.",
    "Scientists recorded a small earthquake and released the magnitude, depth and epicentre through the national seismology centre.",
    "The municipal corporation began repairing a damaged water pipeline. Engineers expect supply to resume after safety testing is complete.",
    "The election commission released the final voter list following the claims and objections period. Citizens can verify their names online.",
    "A public sector bank revised selected service charges from next month. The official circular gives the new rates and exemptions.",
    "The education ministry published a scholarship application notice with eligibility rules, required documents and the closing date.",
    "The transport department introduced additional buses on a busy route after reviewing passenger demand during peak hours.",
    "The food safety authority recalled one production batch after laboratory testing found that it did not meet the stated standard.",
    "The research team released satellite observations showing seasonal changes in vegetation. The dataset and method are available for review.",
    "The company confirmed a software update that fixes two documented security vulnerabilities. Users were advised to install the latest version.",
    "The sports federation announced the national tournament schedule and published the venue list on its official website.",
    "The power utility scheduled maintenance in three neighbourhoods and gave the expected interruption times in a customer notice.",
    "Police issued a traffic advisory for the marathon and listed temporary road closures, diversion routes and emergency access points.",
    "The hospital inaugurated a new diagnostic unit after receiving regulatory approval and completing equipment calibration.",
]

FAKE_ARTICLES = [
    "SHOCKING secret cure doctors refuse to reveal can remove every disease overnight. Share this message before powerful companies delete it.",
    "Breaking: the government will deposit a huge cash reward into every citizen's account tonight. Click immediately and forward to ten groups.",
    "Scientists prove that drinking a single homemade mixture makes people immune to all viruses forever, but the report has been hidden from the public.",
    "Urgent warning: every mobile phone will stop working at midnight because of a mysterious satellite signal. Switch off your device and share now.",
    "A famous actor has been appointed as the new central bank governor, according to an anonymous viral post with no official announcement.",
    "Miracle seeds produce crops in one day without water, soil or sunlight. Farmers are told to send money to reserve the secret product.",
    "Leaked message claims all schools will remain closed for six months, although no education department notice or signed order is provided.",
    "Doctors confirm that eating one fruit permanently changes blood type within minutes. The article gives no hospital, study or named researcher.",
    "Viral post says a rare planet will pass close enough to Earth to be seen during daytime and will cause gravity to disappear for an hour.",
    "Exclusive report claims the national examination has been cancelled permanently. It cites unnamed insiders and provides no official link.",
    "A forwarded message says bank accounts will be frozen unless users send their password to a verification number within thirty minutes.",
    "Secret technology allows a car to run forever on plain water. The inventor allegedly vanished before demonstrating the machine to independent engineers.",
    "Breaking news says the city has banned all internet access from tomorrow, but the article has no date, authority or public notification.",
    "A viral video caption claims a ten year old image shows today's disaster. The post provides no location, photographer or original source.",
    "Experts guarantee that one investment will double money every week without risk. Readers are pressured to transfer funds immediately.",
    "Anonymous post claims a new law makes weekends illegal and says police will arrest anyone who stays home on Sunday.",
    "Sensational article says an ordinary kitchen spice can replace every prescribed medicine and tells patients to stop treatment immediately.",
    "Forwarded alert says a supermarket is giving free laptops to everyone who completes an unknown survey and shares the link with friends.",
    "Unverified message claims a celebrity donated an impossible amount of money and asks readers to pay a fee to receive a gift.",
    "The moon will turn green for the first time in five hundred years tonight, claims a post that gives no astronomical source or observation time.",
    "A fabricated interview claims a public official resigned during a secret meeting, while official records show no such announcement.",
    "Clickbait story announces that scientists found a city under the ocean populated by immortal humans, but offers no expedition records or evidence.",
    "Message claims every user who types a code will receive unlimited mobile data for life. The link leads to an unrelated form requesting personal details.",
    "Rumour says all currency notes become invalid tomorrow morning and urges people to buy gift cards immediately instead of checking official sources.",
]

def build_demo_corpus() -> Tuple[List[str], List[str]]:
    # Keep every demonstration article unique. This avoids near-duplicate
    # examples appearing in both the training and testing partitions.
    texts = list(REAL_ARTICLES) + list(FAKE_ARTICLES)
    labels = ["REAL"] * len(REAL_ARTICLES) + ["FAKE"] * len(FAKE_ARTICLES)
    return texts, labels


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


@dataclass
class EnsembleBundle:
    vectorizer: TfidfVectorizer
    models: Dict[str, object]
    metrics: Dict[str, object]


def _make_models() -> Dict[str, object]:
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1200, class_weight="balanced", random_state=42
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(
            n_neighbors=5, metric="cosine", algorithm="brute", weights="distance"
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=18, min_samples_leaf=2, class_weight="balanced", random_state=42
        ),
    }


def majority_vote(votes: Iterable[str]) -> str:
    vote_list = list(votes)
    return "REAL" if vote_list.count("REAL") >= 2 else "FAKE"


@lru_cache(maxsize=1)
def get_bundle() -> EnsembleBundle:
    texts, labels = build_demo_corpus()
    x_train, x_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.20, random_state=42, stratify=labels
    )

    vectorizer = TfidfVectorizer(
        preprocessor=clean_text,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=6000,
        sublinear_tf=True,
    )
    train_matrix = vectorizer.fit_transform(x_train)
    test_matrix = vectorizer.transform(x_test)

    models = _make_models()
    model_predictions: Dict[str, np.ndarray] = {}
    model_accuracy: Dict[str, float] = {}
    for name, model in models.items():
        model.fit(train_matrix, y_train)
        predictions = model.predict(test_matrix)
        model_predictions[name] = predictions
        model_accuracy[name] = float(accuracy_score(y_test, predictions))

    ensemble_predictions = np.array(
        [
            majority_vote(
                model_predictions[name][index] for name in model_predictions
            )
            for index in range(len(y_test))
        ]
    )
    ensemble_accuracy = float(accuracy_score(y_test, ensemble_predictions))
    matrix = confusion_matrix(y_test, ensemble_predictions, labels=["FAKE", "REAL"])
    report = classification_report(
        y_test, ensemble_predictions, labels=["FAKE", "REAL"], output_dict=True, zero_division=0
    )

    # Refit every model on all bundled examples for the deployed analyzer.
    full_matrix = vectorizer.fit_transform(texts)
    for model in models.values():
        model.fit(full_matrix, labels)

    return EnsembleBundle(
        vectorizer=vectorizer,
        models=models,
        metrics={
            "samples": len(texts),
            "train_samples": len(x_train),
            "test_samples": len(x_test),
            "model_accuracy": model_accuracy,
            "ensemble_accuracy": ensemble_accuracy,
            "confusion_matrix": matrix.tolist(),
            "classification_report": report,
            "dataset": "Bundled balanced classroom demonstration corpus",
        },
    )


def _language_signals(text: str) -> List[Dict[str, str]]:
    lower = text.lower()
    signals: List[Dict[str, str]] = []
    clickbait = [
        "shocking", "secret", "miracle", "urgent", "share now", "breaking",
        "doctors hate", "guaranteed", "forward", "hidden", "anonymous",
    ]
    verification = [
        "official", "published", "report", "according to", "statement",
        "department", "court", "study", "data", "notification",
    ]
    found_clickbait = [word for word in clickbait if word in lower]
    found_verification = [word for word in verification if word in lower]

    if found_clickbait:
        signals.append({"type": "risk", "text": "Sensational language: " + ", ".join(found_clickbait[:3])})
    if text.count("!") >= 2:
        signals.append({"type": "risk", "text": "Repeated exclamation marks"})
    words = re.findall(r"\b[A-Za-z]{3,}\b", text)
    upper_ratio = sum(word.isupper() for word in words) / max(len(words), 1)
    if upper_ratio > 0.18:
        signals.append({"type": "risk", "text": "High proportion of capitalized words"})
    if found_verification:
        signals.append({"type": "support", "text": "Attribution language: " + ", ".join(found_verification[:3])})
    if re.search(r"\b(19|20)\d{2}\b", text):
        signals.append({"type": "neutral", "text": "Contains a specific year"})
    if not signals:
        signals.append({"type": "neutral", "text": "No strong surface-language signal detected"})
    return signals


def analyze_text(text: str) -> Dict[str, object]:
    normalized = " ".join(text.split())
    if len(normalized) < 40:
        raise ValueError("Enter at least 40 characters so the model has enough text to analyse.")
    if len(normalized) > 20000:
        raise ValueError("The article is too long. Use no more than 20,000 characters.")

    bundle = get_bundle()
    vector = bundle.vectorizer.transform([normalized])
    votes = {
        name: str(model.predict(vector)[0])
        for name, model in bundle.models.items()
    }
    label = majority_vote(votes.values())
    agreement = sum(value == label for value in votes.values()) / len(votes)
    tokens = re.findall(r"\b\w+\b", normalized)
    sentences = [s for s in re.split(r"[.!?]+", normalized) if s.strip()]

    return {
        "label": label,
        "agreement": round(agreement * 100, 1),
        "votes": votes,
        "stats": {
            "words": len(tokens),
            "sentences": len(sentences),
            "characters": len(normalized),
            "average_word_length": round(sum(len(t) for t in tokens) / max(len(tokens), 1), 1),
        },
        "signals": _language_signals(normalized),
        "explanation": (
            "Most classifiers found language patterns closer to the REAL examples in the training corpus."
            if label == "REAL"
            else "Most classifiers found language patterns closer to the FAKE examples in the training corpus."
        ),
        "disclaimer": "This is a text-pattern classification result, not independent fact verification. Confirm important claims with reliable primary sources.",
        "model": "TF-IDF with Logistic Regression, KNN and Decision Tree hard voting",
    }


def evaluation_metrics() -> Dict[str, object]:
    return get_bundle().metrics
