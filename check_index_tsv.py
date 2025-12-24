import pandas as pd
from pathlib import Path

# поменяй путь на тот, который у тебя реально нашёлся
meta_path = Path("data/train_raw/magick/magick_index.tsv")

df = pd.read_csv(meta_path, sep="\t")
print(df.head())
print("\nColumns:", df.columns.tolist())
print("\nRows:", len(df))
