"""Train and evaluate spam classifiers on the UCI SMS Spam Collection dataset."""
from __future__ import annotations
import io, json, zipfile
from pathlib import Path
from urllib.request import urlopen

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
import joblib

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
DATASET_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"

def load_dataset() -> pd.DataFrame:
    """Load local dataset if present; otherwise download it from UCI."""
    DATA_DIR.mkdir(exist_ok=True)
    candidates = [DATA_DIR / "SMSSpamCollection", DATA_DIR / "SMSSpamCollection.txt"]
    for path in candidates:
        if path.exists():
            df = pd.read_csv(path, sep="\t", header=None, names=["label", "text"], encoding="utf-8")
            return df.dropna()
    print("Downloading the UCI SMS Spam Collection dataset...")
    with urlopen(DATASET_URL, timeout=60) as response:
        content = response.read()
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        name = next(n for n in archive.namelist() if n.endswith("SMSSpamCollection"))
        raw = archive.read(name)
    (DATA_DIR / "SMSSpamCollection").write_bytes(raw)
    df = pd.read_csv(io.BytesIO(raw), sep="\t", header=None, names=["label", "text"], encoding="utf-8")
    return df.dropna()

def train_and_save():
    MODEL_DIR.mkdir(exist_ok=True)
    df = load_dataset()
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["label"].isin(["ham", "spam"])]
    df = df[df["text"].str.len() > 0].drop_duplicates(subset=["text"])
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.20, random_state=42, stratify=df["label"]
    )
    models = {
        "Multinomial Naive Bayes": Pipeline([
            ("tfidf", TfidfVectorizer(lowercase=True, strip_accents="unicode",
                                      ngram_range=(1, 2), min_df=2, max_features=50000,
                                      sublinear_tf=True)),
            ("classifier", MultinomialNB(alpha=0.35)),
        ]),
        "Logistic Regression": Pipeline([
            ("tfidf", TfidfVectorizer(lowercase=True, strip_accents="unicode",
                                      ngram_range=(1, 2), min_df=2, max_features=50000,
                                      sublinear_tf=True)),
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]),
        "Linear SVM": Pipeline([
            ("tfidf", TfidfVectorizer(lowercase=True, strip_accents="unicode",
                                      ngram_range=(1, 2), min_df=2, max_features=50000,
                                      sublinear_tf=True)),
            ("classifier", LinearSVC(class_weight="balanced")),
        ]),
    }
    rows, fitted = [], {}
    for name, pipeline in models.items():
        print(f"Training {name}...")
        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_test)
        rows.append({
            "model": name,
            "accuracy": float(accuracy_score(y_test, pred)),
            "precision_spam": float(precision_score(y_test, pred, pos_label="spam", zero_division=0)),
            "recall_spam": float(recall_score(y_test, pred, pos_label="spam", zero_division=0)),
            "f1_spam": float(f1_score(y_test, pred, pos_label="spam", zero_division=0)),
        })
        fitted[name] = pipeline
    results = pd.DataFrame(rows).sort_values(["f1_spam", "precision_spam"], ascending=False)
    best_name = str(results.iloc[0]["model"])
    best_model = fitted[best_name]
    predictions = best_model.predict(X_test)
    joblib.dump(best_model, MODEL_DIR / "spam_classifier.joblib")
    results.to_csv(MODEL_DIR / "model_comparison.csv", index=False)
    (MODEL_DIR / "metadata.json").write_text(json.dumps({
        "best_model": best_name,
        "dataset": "UCI SMS Spam Collection",
        "dataset_rows": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "label_counts": {str(k): int(v) for k, v in df["label"].value_counts().items()},
        "classification_report": classification_report(y_test, predictions, output_dict=True, zero_division=0),
    }, indent=2), encoding="utf-8")
    cm = confusion_matrix(y_test, predictions, labels=["ham", "spam"])
    plt.figure(figsize=(6, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"])
    plt.xlabel("Predicted label")
    plt.ylabel("Actual label")
    plt.title(f"Confusion Matrix — {best_name}")
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "confusion_matrix.png", dpi=160)
    plt.close()
    print("\nModel comparison:")
    print(results.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved best model: {best_name}")
    print(f"Artifacts saved to: {MODEL_DIR}")
    return best_model, results

if __name__ == "__main__":
    train_and_save()
