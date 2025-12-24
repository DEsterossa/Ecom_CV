from __future__ import annotations

from pathlib import Path
import random

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


# ---- настройки ----
MAGICK_IMAGES_ROOT = Path("data/train_raw/magick/images")   # где лежат скачанные png
OUT_ROOT = Path("data/train_processed")
OUT_IMG_DIR = OUT_ROOT / "images"
OUT_A_DIR = OUT_ROOT / "alpha"

N_SAMPLES = 2000        # сколько обработать (должно совпадать с тем, что скачал)
SEED = 42
OUT_SIZE = (1024, 1024) # оставляем как есть

# типы фонов: ближе к e-commerce (белый/серый/легкий градиент)
BG_MODES = ["white", "lightgray", "gradient"]


def make_background(size: tuple[int, int], mode: str) -> Image.Image:
    w, h = size
    if mode == "white":
        return Image.new("RGB", (w, h), (255, 255, 255))
    if mode == "lightgray":
        v = random.randint(235, 250)
        return Image.new("RGB", (w, h), (v, v, v))
    if mode == "gradient":
        # лёгкий вертикальный градиент около белого
        top = random.randint(245, 255)
        bottom = random.randint(235, 250)
        col = np.linspace(top, bottom, h, dtype=np.uint8)[:, None]
        grad = np.repeat(col, w, axis=1)
        rgb = np.stack([grad, grad, grad], axis=-1)  # HxWx3
        return Image.fromarray(rgb, mode="RGB")
    raise ValueError(f"Unknown mode: {mode}")


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_A_DIR.mkdir(parents=True, exist_ok=True)

    paths = sorted(MAGICK_IMAGES_ROOT.rglob("*.png"))
    if len(paths) == 0:
        raise RuntimeError(f"No png found in {MAGICK_IMAGES_ROOT}")

    # ограничим N_SAMPLES (на случай, если скачал больше)
    paths = paths[:N_SAMPLES]
    print("Found images:", len(paths))

    rows = []
    for i, p in enumerate(tqdm(paths, desc="Preparing processed dataset")):
        with Image.open(p) as im:
            if im.mode != "RGBA":
                im = im.convert("RGBA")

            if im.size != OUT_SIZE:
                # на всякий — но обычно MAGICK уже 1024x1024
                im = im.resize(OUT_SIZE, resample=Image.BILINEAR)

            rgba = im
            rgb = rgba.convert("RGB")
            alpha = rgba.getchannel("A")  # 'L', 0..255

            # композитим RGB на выбранный фон (используя альфу как маску)
            bg_mode = random.choice(BG_MODES)
            bg = make_background(OUT_SIZE, bg_mode)

            # Image.composite берёт пиксели из первого изображения там, где mask!=0
            comp = Image.composite(rgb, bg, alpha)

            # сохраняем
            out_id = f"{i:08d}"
            img_out = OUT_IMG_DIR / f"{out_id}.png"
            a_out = OUT_A_DIR / f"{out_id}.png"

            comp.save(img_out, format="PNG")
            alpha.save(a_out, format="PNG")

            rows.append({"image_path": str(img_out), "alpha_path": str(a_out)})

    df = pd.DataFrame(rows)
    print("Saved processed pairs:", len(df))
    df.to_csv(OUT_ROOT / "pairs.csv", index=False)
    print("Wrote:", OUT_ROOT / "pairs.csv")


if __name__ == "__main__":
    main()
