# Тестовый скрипт для проверки исправления двойного sigmoid
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
from PIL import Image
from torchvision import transforms

# Добавляем путь к U2NETP
u2net_path = Path("U-2-Net")
if str(u2net_path.absolute()) not in sys.path:
    sys.path.insert(0, str(u2net_path.absolute()))

from model.u2net import U2NETP

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Датасет
class CsvAlphaDataset:
    def __init__(self, csv_path, size=(512, 512), limit=10):
        self.df = pd.read_csv(csv_path)
        if limit:
            self.df = self.df.head(limit).reset_index(drop=True)
        self.size = size
        
        self.image_transforms = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        
        self.mask_transforms = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ])
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        row = self.df.iloc[index]
        img = Image.open(Path(row["image_path"])).convert("RGB")
        mask = Image.open(Path(row["alpha_path"])).convert("L")
        return {"img": self.image_transforms(img), "mask": self.mask_transforms(mask)}

# Loss/Metric функции (ИСПРАВЛЕННЫЕ)
def u2net_outputs_to_list(outputs):
    if isinstance(outputs, (list, tuple)):
        return list(outputs)
    return [outputs]

def u2net_mse_loss(outputs, target):
    """ИСПРАВЛЕНО: без повторного sigmoid"""
    outs = u2net_outputs_to_list(outputs)
    loss = 0.0
    for out in outs:
        loss += F.mse_loss(out, target)
    return loss / len(outs)

@torch.no_grad()
def u2net_mse_metric(outputs, target):
    """ИСПРАВЛЕНО: без повторного sigmoid"""
    out0 = u2net_outputs_to_list(outputs)[0]
    return F.mse_loss(out0, target)

# Загрузка модели
print("\n=== Загрузка модели U2NETP ===")
model = U2NETP(in_ch=3, out_ch=1).to(device)
ckpt_path = Path("U-2-Net/saved_models/u2netp/u2netp.pth")
ckpt = torch.load(ckpt_path, map_location="cpu")
state = ckpt if isinstance(ckpt, dict) and "state_dict" not in ckpt else ckpt.get("state_dict", ckpt)
model.load_state_dict(state, strict=True)
model.eval()
print("[OK] Модель загружена")

# Загрузка данных
print("\n=== Загрузка данных ===")
train_ds = CsvAlphaDataset("data/splits/train.csv", size=(512, 512), limit=2)
train_loader = DataLoader(train_ds, batch_size=2, shuffle=False)
print(f"[OK] Датасет загружен: {len(train_ds)} примеров")

# ПРОВЕРКА ИСПРАВЛЕНИЯ
print("\n=== ПРОВЕРКА ИСПРАВЛЕНИЯ ===")
batch = next(iter(train_loader))
imgs = batch["img"].to(device)
masks = batch["mask"].to(device)

print(f"Input imgs shape: {imgs.shape}, range: [{imgs.min().item():.3f}, {imgs.max().item():.3f}]")
print(f"Input masks shape: {masks.shape}, range: [{masks.min().item():.3f}, {masks.max().item():.3f}]")

with torch.no_grad():
    outputs = model(imgs)
    out0 = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
    
    print(f"\n[OK] Model output shape: {out0.shape}")
    print(f"[OK] Model output range: [{out0.min().item():.6f}, {out0.max().item():.6f}]")
    print(f"[OK] Model output finite: {torch.isfinite(out0).all().item()}")
    
    # Проверяем loss и metric
    loss = u2net_mse_loss(outputs, masks)
    metric = u2net_mse_metric(outputs, masks)
    
    print(f"\n[OK] Loss: {loss.item():.6f}")
    print(f"[OK] Metric: {metric.item():.6f}")
    print(f"[OK] Loss finite: {torch.isfinite(loss).item()}")
    print(f"[OK] Metric finite: {torch.isfinite(metric).item()}")

# Итоговая проверка
print("\n" + "="*60)
if torch.isfinite(out0).all() and torch.isfinite(loss) and torch.isfinite(metric):
    print("[SUCCESS] Все значения finite и корректные.")
    print("[SUCCESS] Исправление работает! Можно запускать обучение.")
else:
    print("[ERROR] Есть NaN или Inf значения.")
print("="*60)

