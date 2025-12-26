# Ecom CV Project

Проект для работы с компьютерным зрением в e-commerce.

## Структура проекта

- `00_pipeline.ipynb` - Основной pipeline обработки
- `01_pipeline.ipynb` - Дополнительный pipeline
- `data/` - Данные проекта
  - `baseline.ipynb` - Базовый эксперимент
  - `create_submission.ipynb` - Создание submission файлов
  - `splits/` - Разделение данных на train/val
  - `train_raw/` - Исходные данные
  - `train_processed/` - Обработанные данные
- `U-2-Net/` - Модель U-2-Net для сегментации
- `outputs/` - Результаты экспериментов

## Установка

```bash
# Клонирование репозитория
git clone <repository-url>
cd ecom_CV

# Установка зависимостей (если есть requirements.txt)
pip install -r requirements.txt
```

## Использование

Откройте нужный `.ipynb` файл в Jupyter Notebook или JupyterLab.

## Git

Репозиторий настроен для работы с `.ipynb` файлами. Большие файлы данных и модели исключены через `.gitignore`.




