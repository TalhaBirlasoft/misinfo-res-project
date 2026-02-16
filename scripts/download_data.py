from datasets import load_dataset
import pandas as pd
from pathlib import Path

OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
<<<<<<< HEAD
    # LIAR dataset: short political statements labeled by truthfulness
    ds = load_dataset("liar", trust_remote_code=True)

    # Flatten to pandas and save train/valid/test
    for split in ["train", "validation", "test"]:
        df = pd.DataFrame(ds[split])
        # We'll use "statement" as text and "label" as target
        df_out = df[["statement", "label"]].copy()
        df_out.to_csv(OUT_DIR / f"liar_{split}.csv", index=False)
        print(f"Saved {split}: {len(df_out)} rows -> {OUT_DIR}/liar_{split}.csv")
=======
    ds = load_dataset("liar",trust_remote_code=True )

    for split in ["train", "validation", "test"]:
        df = pd.DataFrame(ds[split])
        df_out = df[["statement", "label"]].copy()
        df_out.to_csv(OUT_DIR / f"liar_{split}.csv", index=False)
        print(f"Saved {split}: {len(df_out)} rows")
>>>>>>> 657dc2e0 (Add Design of Experiment with baseline results)

if __name__ == "__main__":
    main()
