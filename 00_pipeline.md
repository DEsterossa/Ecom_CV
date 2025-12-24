# Pipeline для обучения U2NETP на задаче сегментации

## 0. Заголовок и цель

**Название проекта:** Сегментация объектов с предсказанием alpha-канала

**Задача:** Предсказание alpha (0..255), метрика MSE

**Данные:** MAGICK (train), Kaggle test (orig_1024)

## 1. Импорты и глобальные настройки

### Импорты, настройки PIL, базовые зависимости

```python
import os
import io
import sys
import base64
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm.auto import tqdm
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

Image.MAX_IMAGE_PIXELS = None
```

### Все параметры меняются здесь

```python
Image.MAX_IMAGE_PIXELS = None

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Пути к данным
TRAIN_CSV = Path("data/splits/train.csv")
VAL_CSV   = Path("data/splits/val.csv")
TEST_ROOT = Path("data/test_dataset/orig_1024")
MODEL_DIR = Path("outputs/experiments/exp_001")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Размеры и батчи: train на 512, val на 1024 (оптимально для 3060 Ti)
TRAIN_SIZE = (512, 512)
VAL_SIZE   = (1024, 1024)
TRAIN_BATCH = 8
VAL_BATCH   = 1

LIMIT_DATA = 10
   
NUM_EPOCHS = 5
LR = 1e-4
RUN_TRAINING = True
```

## 2. Датасеты и аугментации

```python
class CsvAlphaDataset(Dataset):
    """
    Ожидает CSV с колонками:
      - image_path
      - alpha_path
    Возвращает dict как в baseline:
      {"img": tensor, "mask": tensor}
    """
    def __init__(self, csv_path: Path, size=(1024, 1024), normalize=True, augment=None, limit=LIMIT_DATA):
        self.csv_path = Path(csv_path)
        self.df = pd.read_csv(self.csv_path)

        if limit is not None:
            self.df = self.df.head(int(limit)).reset_index(drop=True)
            
        self.size = size
        self.augment = augment

        # Трансформы как в baseline
        if normalize:
            self.image_transforms = transforms.Compose([
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # RGB -> [-1, 1]
            ])
        else:
            self.image_transforms = transforms.Compose([
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),  # [0,1]
            ])

        # Маска/альфа: нам важно сохранить значения 0..1 (без Normalize)
        self.mask_transforms = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),  # L -> [0,1], shape [1,H,W]
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img_path = Path(row["image_path"])
        mask_path = Path(row["alpha_path"])

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # 0..255

        if self.augment is not None:
            img, mask = self.augment(img, mask)

        img_t = self.image_transforms(img)
        mask_t = self.mask_transforms(mask)

        return {"img": img_t, "mask": mask_t}


class TestImageDataset(Dataset):
    def __init__(self, root: Path, size=(1024, 1024)):
        self.root = Path(root)
        self.images = sorted([p for p in self.root.iterdir() if p.is_file()])
        self.transform = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        path = self.images[idx]
        img = Image.open(path).convert("RGB")
        return {"path": path.name, "img": self.transform(img)}


# Старая версия get_dataloaders (закомментирована)
# def get_dataloaders(train_csv: Path, val_csv: Path, size=(1024, 1024), batch_size=4, num_workers=0):
#     train_ds = CsvAlphaDataset(train_csv, size=size)
#     val_ds   = CsvAlphaDataset(val_csv, size=size)

#     train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
#     val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
#     return train_loader, val_loader

# train_loader, val_loader = get_dataloaders(TRAIN_CSV, VAL_CSV, size=IMG_SIZE, batch_size=BATCH_SIZE)
```

### Старая версия аугментаций (закомментирована)

```python
# class JointAugment:
#     def __init__(self, p_flip=0.5, p_rotate=0.3, max_rotate=10, p_color=0.7):
#         self.p_flip = p_flip
#         self.p_rotate = p_rotate
#         self.max_rotate = max_rotate
#         self.p_color = p_color

#     def __call__(self, img_pil, mask_pil):
#         # --- Геометрия: одинаково для img и mask ---
#         if random.random() < self.p_flip:
#             img_pil = TF.hflip(img_pil)
#             mask_pil = TF.hflip(mask_pil)

#         if random.random() < self.p_rotate:
#             angle = random.uniform(-self.max_rotate, self.max_rotate)
#             img_pil = TF.rotate(img_pil, angle, interpolation=InterpolationMode.BILINEAR)
#             mask_pil = TF.rotate(mask_pil, angle, interpolation=InterpolationMode.BILINEAR)

#         # --- Фотометрия: только img ---
#         if random.random() < self.p_color:
#             # яркость/контраст/насыщенность/тон
#             # (тон иногда может ухудшать "товар на белом фоне", если будет слишком агрессивно)
#             b = random.uniform(0.85, 1.15)
#             c = random.uniform(0.85, 1.15)
#             s = random.uniform(0.85, 1.15)
#             img_pil = TF.adjust_brightness(img_pil, b)
#             img_pil = TF.adjust_contrast(img_pil, c)
#             img_pil = TF.adjust_saturation(img_pil, s)

#         return img_pil, mask_pil
```

### Класс аугментаций с random crop

```python
class JointAugment:
    def __init__(self, size=512, p_flip=0.5, p_rotate=0.0, max_rotate=10, p_color=0.7):
        self.crop_size = size
        self.p_flip = p_flip
        self.p_rotate = p_rotate
        self.max_rotate = max_rotate
        self.p_color = p_color

    def __call__(self, img_pil, mask_pil):
        # random crop (одинаково для img и mask)
        i, j, h, w = transforms.RandomCrop.get_params(img_pil, output_size=(self.crop_size, self.crop_size))
        img_pil = TF.crop(img_pil, i, j, h, w)
        mask_pil = TF.crop(mask_pil, i, j, h, w)

        # flip
        if random.random() < self.p_flip:
            img_pil = TF.hflip(img_pil)
            mask_pil = TF.hflip(mask_pil)

        # (rotate можно пока выключить)
        if self.p_rotate and random.random() < self.p_rotate:
            angle = random.uniform(-self.max_rotate, self.max_rotate)
            img_pil = TF.rotate(img_pil, angle, interpolation=InterpolationMode.BILINEAR)
            mask_pil = TF.rotate(mask_pil, angle, interpolation=InterpolationMode.BILINEAR)

        # color jitter only for img
        if random.random() < self.p_color:
            b = random.uniform(0.85, 1.15)
            c = random.uniform(0.85, 1.15)
            s = random.uniform(0.85, 1.15)
            img_pil = TF.adjust_brightness(img_pil, b)
            img_pil = TF.adjust_contrast(img_pil, c)
            img_pil = TF.adjust_saturation(img_pil, s)

        return img_pil, mask_pil
```

**Примечание:** Геометрия синхронно img+mask, фотометрия только img

### Создадим Dataloaders для аугментации train данных

```python
train_aug = JointAugment(size=512, p_flip=0.5, p_rotate=0.3, max_rotate=10, p_color=0.7)

train_ds = CsvAlphaDataset(TRAIN_CSV, size=TRAIN_SIZE, augment=train_aug, limit=LIMIT_DATA)
val_ds   = CsvAlphaDataset(VAL_CSV, size=VAL_SIZE, augment=None, limit=LIMIT_DATA)

train_loader = DataLoader(train_ds, batch_size=TRAIN_BATCH, shuffle=True, num_workers=0, pin_memory=True)
val_loader   = DataLoader(val_ds, batch_size=VAL_BATCH, shuffle=False, num_workers=0, pin_memory=True)
```

### Проверяем, что пайплайн данных верный

```python
train_batch = next(iter(train_loader))
print(train_batch["img"].shape, train_batch["img"].min().item(), train_batch["img"].max().item())
print(train_batch["mask"].shape, train_batch["mask"].min().item(), train_batch["mask"].max().item())
val_batch = next(iter(val_loader))
print(val_batch["img"].shape, val_batch["img"].min().item(), val_batch["img"].max().item())
```

**Вывод:**
```
torch.Size([8, 3, 512, 512]) -1.0 1.0
torch.Size([8, 1, 512, 512]) 0.0 1.0
torch.Size([1, 3, 1024, 1024]) -1.0 1.0
```

```python
batch = next(iter(train_loader))
imgs = batch["img"].to(device)
masks = batch["mask"].to(device)

print("imgs finite:", torch.isfinite(imgs).all().item(),
      "min/max:", imgs.min().item(), imgs.max().item(), imgs.dtype)
print("masks finite:", torch.isfinite(masks).all().item(),
      "min/max:", masks.min().item(), masks.max().item(), masks.dtype)
```

**Вывод:**
```
imgs finite: True min/max: -1.0 1.0 torch.float32
masks finite: True min/max: 0.0 1.0 torch.float32
```

## 3. Модель U2NETP + загрузка pretrained

### Инициализация модели и pretrained весов

```python
ckpt = torch.load("U-2-Net/saved_models/u2netp/u2netp.pth", map_location="cpu")
state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
state = {k.replace("module.", ""): v for k, v in state.items()}

bad_ckpt = [k for k,v in state.items() if not torch.isfinite(v).all()]
print("Non-finite tensors in checkpoint:", len(bad_ckpt))
print(bad_ckpt[:20])
```

**Вывод:**
```
Non-finite tensors in checkpoint: 0
[]
```

```python
u2net_path = Path("U-2-Net")
if str(u2net_path.absolute()) not in sys.path:
    sys.path.insert(0, str(u2net_path.absolute()))

from model.u2net import U2NETP  
import torch

model = U2NETP(in_ch=3, out_ch=1).to(device)

# Путь к чекпойнту
ckpt_path = Path("U-2-Net/saved_models/u2netp/u2netp.pth")
ckpt = torch.load(ckpt_path, map_location="cpu")
# иногда в чекпойнте просто state_dict, иногда dict со state_dict
state = ckpt if isinstance(ckpt, dict) and "state_dict" not in ckpt else ckpt.get("state_dict", ckpt)
model.load_state_dict(state, strict=True)
```

```python
bad = []
for k, v in model.state_dict().items():
    if not torch.isfinite(v).all():
        bad.append(k)

print("Non-finite tensors in model.state_dict:", len(bad))
print(bad[:20])
```

**Вывод:**
```
Non-finite tensors in model.state_dict: 0
[]
```

## 4. Loss / Metric для U2NETP

```python
def u2net_outputs_to_list(outputs):
    if isinstance(outputs, (list, tuple)):
        return list(outputs)
    return [outputs]

def u2net_mse_loss(outputs, target):
    outs = u2net_outputs_to_list(outputs)
    loss = 0.0
    for out in outs:
        loss += F.mse_loss(torch.sigmoid(out), target)
    return loss / len(outs)

@torch.no_grad()
def u2net_mse_metric(outputs, target):
    # метрику считаем по главному выходу (обычно первый, d0)
    out0 = u2net_outputs_to_list(outputs)[0]
    probs = torch.sigmoid(out0)
    return F.mse_loss(probs, target)
```

**Примечание:** Лосс — MSE по alpha, метрика — MSE по главному выходу

## 5. Train / Eval loops (с AMP)

**Примечание:** AMP экономит VRAM и ускоряет обучение

```python
scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

def train_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0.0
    total_mse = 0.0

    for batch in tqdm(loader, desc="train", leave=False):
        imgs = batch["img"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            outputs = model(imgs)
            loss = u2net_mse_loss(outputs, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        bs = imgs.size(0)
        total_loss += loss.item() * bs
        total_mse += u2net_mse_metric(outputs, masks).item() * bs

    n = len(loader.dataset)
    return total_loss / n, total_mse / n


@torch.no_grad()
def eval_epoch(model, loader):
    model.eval()
    total_loss = 0.0
    total_mse = 0.0

    for batch in tqdm(loader, desc="val", leave=False):
        imgs = batch["img"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            outputs = model(imgs)
            loss = u2net_mse_loss(outputs, masks)

        bs = imgs.size(0)
        total_loss += loss.item() * bs
        total_mse += u2net_mse_metric(outputs, masks).item() * bs

    n = len(loader.dataset)
    return total_loss / n, total_mse / n
```

## 6. Training run + сохранение лучшей модели

```python
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

best_val_mse = float("inf")
best_path = MODEL_DIR / "u2netp_best.pth"

if RUN_TRAINING:
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_mse = train_epoch(model, train_loader, optimizer)
        val_loss, val_mse = eval_epoch(model, val_loader)
        print(f"Epoch {epoch}: train_mse={train_mse:.6f} val_mse={val_mse:.6f}")

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            torch.save({"model_state": model.state_dict()}, best_path)
else:
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device)["model_state"])
```

**Предупреждение:**
```
c:\Users\79104\anaconda3\Lib\site-packages\torch\nn\functional.py:3809: UserWarning: nn.functional.upsample is deprecated. Use nn.functional.interpolate instead.
  warnings.warn("nn.functional.upsample is deprecated. Use nn.functional.interpolate instead.")
```

### Дополнительные проверки данных

```python
batch = next(iter(train_loader))
imgs = batch["img"]
masks = batch["mask"]

print("imgs:", imgs.shape, imgs.dtype, imgs.min().item(), imgs.max().item())
print("masks:", masks.shape, masks.dtype, masks.min().item(), masks.max().item())

print("imgs finite:", torch.isfinite(imgs).all().item())
print("masks finite:", torch.isfinite(masks).all().item())
print("masks unique approx:", len(torch.unique((masks*255).to(torch.uint8))))
```

**Вывод:**
```
imgs: torch.Size([8, 3, 512, 512]) torch.float32 -1.0 1.0
masks: torch.Size([8, 1, 512, 512]) torch.float32 0.0 1.0
imgs finite: True
masks finite: True
masks unique approx: 256
```

```python
model.eval()
batch = next(iter(train_loader))
imgs = batch["img"].to(device)
masks = batch["mask"].to(device)

with torch.no_grad():
    outputs = model(imgs)
    out0 = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
    probs = torch.sigmoid(out0)
    loss = F.mse_loss(probs, masks)

print("out0 finite:", torch.isfinite(out0).all().item())
print("probs finite:", torch.isfinite(probs).all().item())
print("loss:", loss.item())
```

**Вывод:**
```
out0 finite: False
probs finite: False
loss: nan
```

## 7. Быстрая визуализация

*(Раздел пуст)*

## 8. Инференс на Kaggle test + submission.csv

```python
model.eval()

test_dataset = TestImageDataset(TEST_ROOT, size=(1024, 1024))
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

rows = []
with torch.no_grad():
    for batch in tqdm(test_loader, desc="test", leave=False):
        imgs = batch["img"].to(device)
        names = batch["path"]
        outputs = model(imgs)
        out0 = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
        probs = torch.sigmoid(out0)
        mask = (probs[0,0].cpu().numpy() * 255).clip(0,255).astype(np.uint8)

        print(mask.shape)
        pil_mask = Image.fromarray(mask, mode="L")
        buf = io.BytesIO()
        pil_mask.save(buf, format="PNG")
        image_utf = base64.b64encode(buf.getvalue()).decode("utf-8")
        rows.append({"filename": names[0].split(".")[0], "image_utf": image_utf})

submission = pd.DataFrame(rows)
submission_path = MODEL_DIR / "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Saved submission to {submission_path}")
```

**Предупреждение:**
```
C:\Users\79104\AppData\Local\Temp\ipykernel_27984\1860429419.py:14: RuntimeWarning: invalid value encountered in cast
  mask = (probs[0,0].cpu().numpy() * 255).clip(0,255).astype(np.uint8)
```

**Вывод:** (множество строк `(1024, 1024)`)
```
(1024, 1024)
(1024, 1024)
...
Saved submission to outputs\experiments\exp_001\submission.csv
```
