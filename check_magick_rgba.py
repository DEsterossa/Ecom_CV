from pathlib import Path
from PIL import Image
import numpy as np

root = Path("data/train_raw/magick/images")
p = next(root.rglob("*.png"))
im = Image.open(p)

print("file:", p)
print("mode:", im.mode, "size:", im.size)

arr = np.array(im)
print("array shape:", arr.shape, "dtype:", arr.dtype)

if im.mode == "RGBA":
    a = np.array(im.getchannel("A"))
    print("alpha min/max:", int(a.min()), int(a.max()))
else:
    print("WARNING: not RGBA")
