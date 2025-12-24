from pathlib import Path
import pandas as pd

PAIRS = Path("data/train_processed/pairs.csv")
OUT_DIR = Path("data/splits")
OUT_DIR.mkdir(parents=True, exist_ok=True)

VAL_FRAC = 0.2
SEED = 42

df = pd.read_csv(PAIRS).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
n_val = int(len(df) * VAL_FRAC)

val = df.iloc[:n_val]
train = df.iloc[n_val:]

train.to_csv(OUT_DIR / "train.csv", index=False)
val.to_csv(OUT_DIR / "val.csv", index=False)

print("train:", len(train), "val:", len(val))