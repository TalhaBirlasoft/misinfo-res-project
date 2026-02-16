import json
from pathlib import Path

import numpy as np
import pandas as pd
import evaluate
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

RESULTS_DIR = Path("experiments/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "distilbert-base-uncased"
NUM_LABELS = 6


def main():
    # Load CSVs
    train_df = pd.read_csv("data/processed/liar_train.csv").dropna()
    val_df = pd.read_csv("data/processed/liar_validation.csv").dropna()
    test_df = pd.read_csv("data/processed/liar_test.csv").dropna()

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def to_ds(df):
        return Dataset.from_pandas(
            df[["statement", "label"]].rename(columns={"statement": "text"})
        )

    train_ds = to_ds(train_df)
    val_ds = to_ds(val_df)
    test_ds = to_ds(test_df)

    # Tokenization
    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=128,
        )

    train_ds = train_ds.map(tokenize, batched=True)
    val_ds = val_ds.map(tokenize, batched=True)
    test_ds = test_ds.map(tokenize, batched=True)

    # Rename label column for Trainer
    train_ds = train_ds.rename_column("label", "labels")
    val_ds = val_ds.rename_column("label", "labels")
    test_ds = test_ds.rename_column("label", "labels")

    cols = ["input_ids", "attention_mask", "labels"]
    train_ds.set_format(type="torch", columns=cols)
    val_ds.set_format(type="torch", columns=cols)
    test_ds.set_format(type="torch", columns=cols)

    # Model
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_LABELS
    )

    # Metrics
    acc_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        acc = acc_metric.compute(predictions=preds, references=labels)["accuracy"]
        macro_f1 = f1_metric.compute(
            predictions=preds, references=labels, average="macro"
        )["f1"]
        return {"accuracy": acc, "macro_f1": macro_f1}

    # Training arguments (UPDATED for new transformers)
    args = TrainingArguments(
        output_dir="experiments/bert_out",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=2,
        weight_decay=0.01,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        report_to="none",
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # Train
    trainer.train()

    # Evaluate on test
    metrics = trainer.evaluate(test_ds)

    out = {
        "model": MODEL_NAME,
        "dataset": "liar",
        "split": "test",
        "accuracy": float(metrics["eval_accuracy"]),
        "macro_f1": float(metrics["eval_macro_f1"]),
        "epochs": 2,
    }

    (RESULTS_DIR / "baseline_distilbert.json").write_text(json.dumps(out, indent=2))

    print("Saved:", RESULTS_DIR / "baseline_distilbert.json")
    print(out)


if __name__ == "__main__":
    main()
