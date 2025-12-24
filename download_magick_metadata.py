from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="OneOverZero/MAGICK",
    repo_type="dataset",
    local_dir="data/train_raw/magick",
    local_dir_use_symlinks=False,
    ignore_patterns=["images/**"],   # не скачиваем картинки
    resume_download=True,            # докачиваем при обрыве
    max_workers=1,                   # меньше параллелизма для стабильности
)
print("OK: metadata downloaded to data/train_raw/magick")
