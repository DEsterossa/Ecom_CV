from pathlib import Path

root = Path("data/train_raw/magick")
print("Root exists:", root.exists())

# покажем верхний уровень
print("\nTop-level files:")
for p in sorted(root.iterdir()):
    print(" -", p.name)

# найдём типичные индекс/метадата файлы
patterns = ["*.tsv", "*.csv", "*.json", "*.parquet", "README*", "dataset*"]
found = []
for pat in patterns:
    found += list(root.rglob(pat))

print("\nFound metadata-like files:")
for p in sorted(set(found))[:50]:
    print(" -", p.as_posix())
