import json
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score

RESULTS_DIR = Path("experiments/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
N = 300  # number of test samples to evaluate (keeps it fast); increase later if you want
P_DROP = 0.12  # probability to drop a token
P_SWAP = 0.08  # probability to swap a token with a simple synonym
P_PUNCT = 0.25 # probability to add punctuation noise

SYNONYMS = {
    "said": "stated",
    "says": "states",
    "claim": "assert",
    "claims": "asserts",
    "bad": "poor",
    "good": "great",
    "big": "large",
    "small": "little",
    "many": "numerous",
    "most": "majority",
    "few": "some",
    "increase": "rise",
    "increased": "rose",
    "decrease": "drop",
    "decreased": "dropped",
    "people": "citizens",
    "money": "funds",
    "tax": "levy",
    "taxes": "levies",
    "job": "role",
    "jobs": "roles",
}

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)

def tokenize(text: str):
    # keep simple word tokens
    return re.findall(r"[A-Za-z0-9']+|[^\w\s]", text)

def detokenize(tokens):
    # simple detokenization
    out = []
    for t in tokens:
        if out and re.match(r"[A-Za-z0-9']+", t) and re.match(r"[A-Za-z0-9']+", out[-1]):
            out.append(" ")
        out.append(t)
    return "".join(out)

def perturb(text: str) -> str:
    toks = tokenize(text)
    new = []
    for t in toks:
        # skip pure punctuation from drop/swap logic
        if re.match(r"[A-Za-z0-9']+", t):
            # drop token
            if random.random() < P_DROP:
                continue
            # synonym swap (lowercased key)
            key = t.lower()
            if random.random() < P_SWAP and key in SYNONYMS:
                repl = SYNONYMS[key]
                # preserve capitalization roughly
                if t[0].isupper():
                    repl = repl.capitalize()
                t = repl
        new.append(t)

    if not new:
        new = toks[:]  # fallback

    # punctuation noise
    if random.random() < P_PUNCT:
        noise = random.choice(["!", "?", "...", "!!"])
        new.append(noise)

    return detokenize(new)

def train_lr():
    train = pd.read_csv("data/processed/liar_train.csv").dropna()
    X_train = train["statement"].astype(str)
    y_train = train["label"].astype(int)

    clf = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=30000, ngram_range=(1, 2))),
        ("lr", LogisticRegression(max_iter=3000)),
    ])
    clf.fit(X_train, y_train)
    return clf

def score(clf, X, y):
    pred = clf.predict(X)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
    }

def main():
    set_seed(SEED)

    test = pd.read_csv("data/processed/liar_test.csv").dropna()
    if N < len(test):
        test = test.sample(n=N, random_state=SEED).reset_index(drop=True)

    X_orig = test["statement"].astype(str).tolist()
    y = test["label"].astype(int).tolist()
    X_pert = [perturb(x) for x in X_orig]

    clf = train_lr()

    s_orig = score(clf, X_orig, y)
    s_pert = score(clf, X_pert, y)

    out = {
        "model": "tfidf_lr",
        "dataset": "liar",
        "n": int(len(y)),
        "perturbation": {
            "type": "rule_based_paraphrase_like",
            "seed": SEED,
            "p_drop": P_DROP,
            "p_swap": P_SWAP,
            "p_punct": P_PUNCT,
        },
        "original": s_orig,
        "perturbed": s_pert,
        "drop": {
            "accuracy": s_orig["accuracy"] - s_pert["accuracy"],
            "macro_f1": s_orig["macro_f1"] - s_pert["macro_f1"],
        },
    }

    out_path = RESULTS_DIR / "robustness_simple_lr.json"
    out_path.write_text(json.dumps(out, indent=2))
    print("Saved:", out_path)
    print(out)

if __name__ == "__main__":
    main()
