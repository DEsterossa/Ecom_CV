from pathlib import Path
from PIL import Image
import random

img_dir = Path("data/train_processed/images")
a_dir = Path("data/train_processed/alpha")

ids = sorted([p.stem for p in img_dir.glob("*.png")])
for _ in range(5):
    sid = random.choice(ids)
    img = Image.open(img_dir / f"{sid}.png").convert("RGB")
    a = Image.open(a_dir / f"{sid}.png").convert("L")

    print(sid, "img", img.size, "alpha", a.size, "alpha min/max", min(a.getdata()), max(a.getdata()))