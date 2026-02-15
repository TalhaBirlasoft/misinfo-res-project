from datasets import load_dataset
import pandas as pd
from pathlib import Path

OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    # LIAR dataset: short political statements labeled by truthfulness
    ds = load_dataset("liar")

    # Flatten to pandas and save train/valid/test
    for split in ["train", "validation", "test"]:
        df = pd.DataFrame(ds[split])
        # We'll use "statement" as text and "label" as target
        df_out = df[["statement", "label"]].copy()
        df_out.to_csv(OUT_DIR / f"liar_{split}.csv", index=False)
        print(f"Saved {split}: {len(df_out)} rows -> {OUT_DIR}/liar_{split}.csv")

if __name__ == "__main__":
    main()
