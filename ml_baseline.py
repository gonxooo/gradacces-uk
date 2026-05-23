"""ml_baseline.py - Logistic Regression and Naive Bayes baselines on TF-IDF."""

import json
import logging
from pathlib import Path
from collections import Counter

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.model_selection import (
        StratifiedKFold, cross_val_score, cross_val_predict
    )
    from sklearn.metrics import classification_report, confusion_matrix
    import numpy as np
except ImportError:
    print("ERROR: scikit-learn not installed.")
    print("Run: pip install scikit-learn")
    exit(1)

SCORED_FILE = Path("data/scored.json")
OUTPUT_FILE = Path("data/ml_comparison.json")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ml")

MIN_PER_CLASS = 5


def main():
    if not SCORED_FILE.exists():
        log.error("%s not found. Run scoring.py first.", SCORED_FILE)
        return

    with SCORED_FILE.open("r", encoding="utf-8") as f:
        data = json.load(f)

    texts = [
        f"{p.get('title','')} . {p.get('description_clean','')}"
        for p in data
    ]
    labels = [p["category"] for p in data]

    log.info("Corpus size: %d postings", len(texts))
    log.info("Label distribution: %s", dict(Counter(labels)))

    counter = Counter(labels)
    valid_labels = {k for k, v in counter.items() if v >= MIN_PER_CLASS}

    X_raw, y = [], []
    for text, label in zip(texts, labels):
        if label in valid_labels:
            X_raw.append(text)
            y.append(label)

    log.info("After filtering rare classes (<%d): %d samples", MIN_PER_CLASS, len(X_raw))
    log.info("Valid classes: %s", sorted(valid_labels))

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        stop_words="english",
        max_features=5000,
    )
    X = vectorizer.fit_transform(X_raw)
    log.info("TF-IDF matrix: %s (%d non-zero)", X.shape, X.nnz)

    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=42
        ),
        "MultinomialNB": MultinomialNB(),
    }

    results = {}
    folds = min(5, min(Counter(y).values()))

    for name, classifier in models.items():
        log.info("%s", name)
        kfold = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)

        accuracy_scores = cross_val_score(classifier, X, y, cv=kfold, scoring="accuracy")
        f1_scores = cross_val_score(classifier, X, y, cv=kfold, scoring="f1_macro")

        log.info("  Accuracy (mean):  %.3f (std: %.3f)", accuracy_scores.mean(), accuracy_scores.std())
        log.info("  Macro F1 (mean):  %.3f (std: %.3f)", f1_scores.mean(), f1_scores.std())
        log.info("  Folds:            %d", folds)

        predictions = cross_val_predict(classifier, X, y, cv=kfold)
        report = classification_report(y, predictions, output_dict=True, zero_division=0)

        labels_sorted = sorted(valid_labels)
        matrix = confusion_matrix(y, predictions, labels=labels_sorted)

        results[name] = {
            "folds": folds,
            "accuracy_mean": round(float(accuracy_scores.mean()), 3),
            "accuracy_std": round(float(accuracy_scores.std()), 3),
            "f1_macro_mean": round(float(f1_scores.mean()), 3),
            "f1_macro_std": round(float(f1_scores.std()), 3),
            "classification_report": report,
            "confusion_matrix": matrix.tolist(),
            "labels": labels_sorted,
        }

        log.info("  Per-class F1:")
        for label in labels_sorted:
            if label in report:
                f1_value = report[label]["f1-score"]
                support = report[label]["support"]
                log.info("    %-20s f1=%.3f (n=%d)", label, f1_value, int(support))

    log.info("Top discriminative features per class (Logistic Regression):")

    lr = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)
    lr.fit(X, y)
    feature_names = vectorizer.get_feature_names_out()

    top_features = {}
    for index, cls in enumerate(lr.classes_):
        top_indices = np.argsort(lr.coef_[index])[-10:][::-1]
        top_terms = [feature_names[j] for j in top_indices]
        top_features[cls] = top_terms
        log.info("  %s: %s", cls, ", ".join(top_terms[:6]))

    output = {
        "corpus_size": len(X_raw),
        "classes_evaluated": sorted(valid_labels),
        "models": results,
        "top_features_per_class": top_features,
        "summary": {
            "best_model": max(results, key=lambda m: results[m]["f1_macro_mean"]),
            "best_accuracy": max(r["accuracy_mean"] for r in results.values()),
            "best_f1_macro": max(r["f1_macro_mean"] for r in results.values()),
        },
    }

    OUTPUT_FILE.parent.mkdir(exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    log.info("Best model:    %s", output["summary"]["best_model"])
    log.info("Macro F1:      %.3f", output["summary"]["best_f1_macro"])
    log.info("Accuracy:      %.3f", output["summary"]["best_accuracy"])
    log.info("Folds:         %d-fold stratified CV", folds)
    log.info("Corpus:        %d postings across %d classes", len(X_raw), len(valid_labels))
    log.info("Saved -> %s", OUTPUT_FILE)


if __name__ == "__main__":
    main()
