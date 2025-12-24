from pathlib import Path
from PIL import Image

root = Path("data/test_dataset/orig_1024")
paths = sorted(root.glob("*.png"))

print("num_images:", len(paths))
assert len(paths) > 0, "Папка пустая"

# Проверим первые 5 файлов
for p in paths[:5]:
    with Image.open(p) as im:
        print(p.name, im.mode, im.size)

# Проверим, что все 1024x1024 и RGB
bad = []
for p in paths:
    with Image.open(p) as im:
        if im.size != (1024, 1024) or im.mode not in ("RGB", "RGBA"):
            bad.append((p.name, im.mode, im.size))

print("bad:", len(bad))
if bad:
    print("examples:", bad[:10])
