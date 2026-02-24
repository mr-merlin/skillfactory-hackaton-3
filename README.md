# Помощник выбора на полке (Shelf Helper)

Прототип рекомендаций по данным «Парфюмерного органа»: сессия или рецепт → Top-N SKU. Методы: `cosine`, `nn`, `gbm`.

---

## Разделы

- [Помощник выбора на полке (Shelf Helper)](#помощник-выбора-на-полке-shelf-helper)
  - [Разделы](#разделы)
  - [Требования и данные](#требования-и-данные)
  - [Как запустить](#как-запустить)
  - [CLI](#cli)
  - [API](#api)
  - [Обучение моделей](#обучение-моделей)
  - [Оценка и аналитика](#оценка-и-аналитика)
  - [Docker](#docker)
  - [Структура проекта](#структура-проекта)

---

## Требования и данные

- Python 3.10+
- Данные: папка `data/` (каталог `perfumes.parquet`, `perfume_notes.parquet`; орган — `organ_sessions.parquet`, `organ_recipe_components.parquet`, `organ_aroma_notes_map.parquet` и др.). Пути: `DATA_PERFUME_DIR`, `DATA_ORGAN_DIR` (по умолчанию `./data`).

---

## Как запустить

```bash
cd backend
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Дальше — CLI или API (ниже).

---

## CLI

```bash
cd backend

# по рецепту (cosine по умолчанию)
.venv/bin/python -m app.cli recommend --recipe "0:49,1:80,2:50,3:40,4:63,5:50" --top-n 10

# по сессии, метод gbm
.venv/bin/python -m app.cli recommend --session-id 1 --method gbm --top-n 10

# вывод JSON
.venv/bin/python -m app.cli recommend --recipe "0:49,1:80,2:50,3:40,4:63,5:50" --method nn --json
```

`--method`: `cosine` | `nn` | `gbm`.

---

## API

```bash
cd backend
.venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000
```

Документация: http://localhost:8000/docs

Пример запроса:
```bash
curl -X POST http://localhost:8000/api/recommend \
  -H "Content-Type: application/json" \
  -d '{"recipe": "0:49,1:80,2:50,3:40,4:63,5:50", "top_n": 10, "method": "gbm"}'
```

---

## Обучение моделей

Модели в репозитории не лежат (см. `.gitignore`). Чтобы получить их:

```bash
cd backend
.venv/bin/python train_gbm.py      # ~10 сек
.venv/bin/python train_two_tower.py # ~1 мин на CPU
```

Чекпоинты сохраняются в `backend/models/`.

---

## Оценка и аналитика

```bash
cd backend
.venv/bin/python run_evaluate.py   # метрики Hit@5/10, MRR, NDCG → evaluation_results.json
.venv/bin/python run_analytics.py  # топ SKU, ноты, связь каналов с ароматами
```

---

## Docker

```bash
docker-compose up --build
# API: http://localhost:8000
```

Папка `data/` монтируется в контейнер; положите туда Lite-датасеты.

---

## Структура проекта

```
backend/
  app/           — API, CLI, сервис рекомендаций, ranking (data, profile, scoring, baseline, evaluation, nn, gbm)
  models/        — обученные модели (.pt, .pkl), не в git
  train_*.py     — обучение GBM и two-tower
  run_*.py       — оценка, аналитика
docs/            — PRESENTATION.md, модель и ограничения
```

Подробнее по коду — см. `backend/app/ranking/` и `backend/app/services/recommend.py`.
