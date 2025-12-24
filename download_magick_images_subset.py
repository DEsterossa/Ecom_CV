from __future__ import annotations

from pathlib import Path
import pandas as pd
from huggingface_hub import hf_hub_download
from tqdm import tqdm

REPO_ID = "OneOverZero/MAGICK"
REPO_TYPE = "dataset"

INDEX_PATH = Path("data/train_raw/magick/magick_index.tsv")
LOCAL_DIR = Path("data/train_raw/magick")

N_IMAGES = 2000          # <- сколько скачать для старта
SEED = 42                # воспроизводимость
FILTER_NSFW = True       # фильтруем nsfw==0
ONLY_PICKED = None       # например "hand" или None чтобы не фильтровать

def main() -> None:
    df = pd.read_csv(INDEX_PATH, sep="\t")
    print("rows total:", len(df))
    assert "page_id" in df.columns, "Нет колонки page_id"

    if FILTER_NSFW and "nsfw" in df.columns:
        df = df[df["nsfw"] == 0]

    if ONLY_PICKED is not None and "picked" in df.columns:
        df = df[df["picked"] == ONLY_PICKED]

    # перемешаем и возьмём N
    df = df.sample(frac=1.0, random_state=SEED).head(N_IMAGES).reset_index(drop=True)
    print("rows selected:", len(df))

    # Скачиваем
    for page_id in tqdm(df["page_id"].tolist(), desc="Downloading MAGICK images"):
        page_id = str(page_id)
        sub = page_id[:2]
        filename = f"images/{sub}/{page_id}.png"

        # если уже скачано — пропускаем
        out_path = LOCAL_DIR / filename
        if out_path.exists():
            continue

        hf_hub_download(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            filename=filename,
            local_dir=str(LOCAL_DIR),
            local_dir_use_symlinks=False,
        )

    print("Done. Example folder:", (LOCAL_DIR / "images").resolve())

if __name__ == "__main__":
    main()
