import json
from pathlib import Path
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score, accuracy_score

<<<<<<< HEAD
RESULTS_DIR = Path("experiments/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

=======

RESULTS_DIR = Path("experiments/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


>>>>>>> 657dc2e0 (Add Design of Experiment with baseline results)
LABEL_MAP = {
    0: "pants-fire",
    1: "false",
    2: "barely-true",
    3: "half-true",
    4: "mostly-true",
    5: "true",
}

<<<<<<< HEAD
def main():
=======

def main():
    # Load data
>>>>>>> 657dc2e0 (Add Design of Experiment with baseline results)
    train = pd.read_csv("data/processed/liar_train.csv").dropna()
    test = pd.read_csv("data/processed/liar_test.csv").dropna()

    X_train, y_train = train["statement"].astype(str), train["label"].astype(int)
    X_test, y_test = test["statement"].astype(str), test["label"].astype(int)

<<<<<<< HEAD
    clf = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=30000, ngram_range=(1,2))),
        ("lr", LogisticRegression(max_iter=3000, n_jobs=None))
    ])

    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)

=======
    # TF-IDF + Logistic Regression pipeline
    clf = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=30000, ngram_range=(1, 2))),
        ("lr", LogisticRegression(max_iter=3000)),
    ])

    # Train
    clf.fit(X_train, y_train)

    # Predict
    pred = clf.predict(X_test)

    # Metrics
>>>>>>> 657dc2e0 (Add Design of Experiment with baseline results)
    acc = float(accuracy_score(y_test, pred))
    f1m = float(f1_score(y_test, pred, average="macro"))

    report = classification_report(
<<<<<<< HEAD
        y_test, pred,
        target_names=[LABEL_MAP[i] for i in sorted(LABEL_MAP.keys())],
        digits=4
    )

=======
        y_test,
        pred,
        target_names=[LABEL_MAP[i] for i in sorted(LABEL_MAP.keys())],
        digits=4,
    )

    # Save results
>>>>>>> 657dc2e0 (Add Design of Experiment with baseline results)
    out = {
        "model": "tfidf_lr",
        "dataset": "liar",
        "split": "test",
        "accuracy": acc,
        "macro_f1": f1m,
    }

    (RESULTS_DIR / "baseline_tfidf_lr.json").write_text(json.dumps(out, indent=2))
    (RESULTS_DIR / "baseline_tfidf_lr_report.txt").write_text(report)

<<<<<<< HEAD
=======
    # Print summary
>>>>>>> 657dc2e0 (Add Design of Experiment with baseline results)
    print("Saved:", RESULTS_DIR / "baseline_tfidf_lr.json")
    print(out)
    print("\nClassification report:\n")
    print(report)

<<<<<<< HEAD
=======

>>>>>>> 657dc2e0 (Add Design of Experiment with baseline results)
if __name__ == "__main__":
    main()
