# Помощник выбора на полке — Парфюмерный орган

Прототип рекомендательной системы для интерактивного стенда «Парфюмерный орган» (SensoryLAB). Стенд фиксирует нажатия пользователя по каналам с интенсивностями; по этим данным система в реальном времени подбирает ароматы, которые стоит взять с полки прямо сейчас.

---

## Как работает

Орган имеет 6 каналов, за каждым закреплены 2 аромы (12 базовых нот: сандал, ваниль, амбра, роза и т.д.). Когда пользователь сохраняет рецепт, мы переводим интенсивности через маппинг `aroma → note` в весовые векторы нот и считаем косинусное сходство с 4000 SKU каталога.

Если в сессии есть история нажатий, мы её тоже учитываем: более долгие и недавние нажатия весят больше. Итоговый профиль — `0.7 × рецепт + 0.3 × нажатия` (коэффициенты настраиваются через `src/config.py`).

Нотное пространство сознательно сужено до 12 органических нот вместо всех 827, что есть в каталоге. Именно это оказалось ключевым: модель и каталог говорят на одном языке, совпадения становятся осмысленными. Свести все 827 нот — значит разбавить вектор пользователя по компонентам, которых орган не знает, — спойлер Hit@10 упал с 8.3% до 3.3%.

---

## Структура проекта

```
.
├── src/
│   ├── config.py          # Настройки путей и параметров
│   ├── data_loader.py     # Загрузка и индексация данных
│   ├── features.py        # Построение профиля пользователя
│   ├── models.py          # CosineRecommender
│   ├── baselines.py       # 3 бейзлайна (TopPopular, NoteOverlap, SingleSignal)
│   ├── metrics.py         # Hit@K, MRR, NDCG@K
│   ├── recommender.py     # Центральный класс PerfumeRecommendationSystem
│   ├── api.py             # REST API (FastAPI)
│   └── cli.py             # CLI-интерфейс (typer + rich)
├── data/                  # Датасеты (parquet)
├── evaluate.py            # Скрипт оценки качества
├── analytics.py           # Аналитический скрипт (топ SKU/нот)
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

---

## Быстрый старт

### 1. Установка зависимостей

```bash
pip install -r requirements.txt
```

### 2. CLI-инференс

```bash
# Рекомендации по session_id
python -m src.cli recommend-session --session-id 42 --top-n 10

# Рекомендации по строке рецепта
python -m src.cli recommend-recipe --recipe "0:49,1:80,2:50,3:40,4:63,5:50" --top-n 5

# Список сессий
python -m src.cli list-sessions --limit 10

# JSON-вывод
python -m src.cli recommend-session --session-id 1 --json
```

### 3. REST API

```bash
# Запуск сервера
uvicorn src.api:app --host 0.0.0.0 --port 8000

# Рекомендации по session_id
curl -X POST http://localhost:8000/recommend/session \
  -H "Content-Type: application/json" \
  -d '{"session_id": 1, "top_n": 5}'

# Рекомендации по рецепту
curl -X POST http://localhost:8000/recommend/recipe \
  -H "Content-Type: application/json" \
  -d '{"recipe": "0:49,1:80,2:50,3:40,4:63,5:50", "top_n": 5}'

# Интерактивная документация
open http://localhost:8000/docs
```

### 4. Оценка качества (метрики)

```bash
python evaluate.py --test-frac 0.2 --top-n 10
```

### 5. Аналитика (Задание B)

```bash
python analytics.py --top-n 10
```

---

## Docker

```bash
# Сборка образа
docker build -t parfume-recommender .

# Запуск с монтированием данных
docker run -p 8000:8000 -v $(pwd)/data:/app/data parfume-recommender

# Или через docker-compose
docker-compose up
```

---

## Результаты метрик (Lite датасет)

| Модель | MRR | Hit@5 | NDCG@5 | Hit@10 | NDCG@10 |
|--------|-----|-------|--------|--------|---------|
| **Cosine Rec (main)** | **0.0316** | **0.0579** | **0.0351** | **0.0826** | **0.0436** |
| B3: Single Signal | 0.0193 | 0.0331 | 0.0228 | 0.0331 | 0.0228 |
| B2: Note Overlap | 0.0060 | 0.0083 | 0.0052 | 0.0248 | 0.0102 |
| B1: Top Popular | 0.0014 | 0.0000 | 0.0000 | 0.0083 | 0.0029 |

> Тестовый сплит: 20% сессий (121 сессия), seed=42.  
> Наша модель обгоняет лучший бейзлайн (B3: Single Signal) по Hit@10 **в 2.5 раза**. B1 и B2 отстают значительно сильнее.

**Coverage**: 17.6% — за 121 тестовую сессию система предложила 705 уникальных SKU из 4000.

---

## Описание данных (Lite)

### Каталог (student_dataset/lite → data/)
- `perfumes.parquet` — 5000 SKU (4521 уникальных), ключ `perfume_id`
- `perfume_notes.parquet` — 39693 записей, 827 уникальных нот
- `perfume_notes_agg.parquet` — ноты агрегированные по SKU

### Логи органа (organ_dataset_synth/lite → data/)
- `organ_sessions.parquet` — 1000 сессий с `target_perfume_id`
- `organ_presses.parquet` — 20119 нажатий
- `organ_recipe_components.parquet` — рецепты по каналам
- `organ_aroma_notes_map.parquet` — маппинг aroma_id → note  
- `organ_aromas.parquet` — справочник: 6 каналов × 2 аромы = 12 базовых нот

### Ключевые связи
```
organ_sessions.session_id → organ_recipe_components.session_id
organ_sessions.session_id → organ_presses.session_id
organ_sessions.target_perfume_id → perfumes.perfume_id
organ_aromas.channel_index → organ_recipe_components.channel_index
organ_aromas.aroma_id → organ_aroma_notes_map.aroma_id → note
perfume_notes.note ↔ organ_aroma_notes_map.note (общее пространство нот)
```

---

## Профиль пользователя

**Из рецепта** (`organ_recipe_components`):
```
u[note] += (intensity/100) / n_aromas_per_channel × weight_mapping
```
Каждый канал равномерно распределяет вклад по своим аромам.

**Из нажатий** (`organ_presses`):
```
u[note] += (intensity_end/100) × (duration_ms/1000) × exp(-λ × age_sec) × weight
```
Учитывает интенсивность, длительность и давность нажатий.

**Комбинирование**: `u = 0.7 × u_recipe + 0.3 × u_presses`

---

## Что не так с метриками — и почему это нормально

Absolute Hit@10 = 8.3% выглядит скромно. Причина в природе синтетики: `target_perfume_id` генерировался не строго из рецепта пяти нот, поэтому часть правильных ответов физически не может попасть в топ — у них просто нет нужных нот в каталоге. Если смотреть на относительное улучшение над бейзлайнами (+150% к лучшему из трёх), модель работает как ожидалось.

Другое ограничение — 17.6% coverage. Это не баг: система честно концентрируется на тех SKU, у которых высокий вес в органических нотах. На реальных данных с более разнообразными рецептами coverage будет выше.

Персонализации между сессиями пока нет — берём только текущую сессию. На Full-датасете с картами лояльности это можно дополнить.

---

## Переключение на Full датасет

Логика не меняется — просто укажите другую папку:

```bash
DATA_DIR=/path/to/full/data python evaluate.py
DATA_DIR=/path/to/full/data uvicorn src.api:app --port 8000
```
