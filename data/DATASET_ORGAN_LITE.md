# DATASET_ORGAN_LITE

## Датасет: Логи «Парфюмерного органа» (lite, синтетика)

### Назначение
Датасет имитирует поведение пользователя у «Парфюмерного органа»:
- сессии,
- нажатия по каналам/аромам (длительность, интенсивность),
- рецепт (смесь интенсивностей по каналам),
- ground-truth выбранный SKU (`target_perfume_id`) для воспроизводимых метрик рекомендаций.

### Папка
`./organ_dataset_synth\lite`

### Файлы
- `organ_sessions.parquet` — сессии (включая `target_perfume_id`)
- `organ_presses.parquet` — события нажатий
- `organ_recipes.parquet` — рецепты строкой
- `organ_recipe_components.parquet` — рецепт, разложенный по каналам
- `organ_aroma_notes_map.parquet` — маппинг `aroma_id -> note`
- `organ_aromas.parquet` — справочник аром/каналов (человекочитаемо)
- `organ_feedback.parquet` — синтетическая обратная связь (view/purchased)
- `meta.json` — параметры генерации (seed и т.п.)

---

## Ключи и связи
- `session_id`: `organ_sessions` → `organ_presses` / `organ_recipes` / `organ_recipe_components`
- `target_perfume_id`: `organ_sessions.target_perfume_id` → **каталог SKU** `student_dataset/lite/perfumes.perfume_id`
- `aroma_id`: `organ_presses.aroma_id` → `organ_aroma_notes_map.aroma_id` → `note`

---

## Единицы измерения
- `*_ms`: UTC миллисекунды Unix epoch
- интенсивность: `0..100`
- `duration_ms`: миллисекунды

---

## Таблица: organ_sessions
Гранулярность: **1 строка = 1 сессия**

| column | type | описание |
|---|---|---|
| `session_id` | int64 | ID сессии органа |
| `user_id` | int64 | синтетический пользователь (для персонализации/повторов) |
| `preset_id` | int64 | ID пресета (в синтетике обычно 1) |
| `channels_count` | int64 | число каналов (например, 6) |
| `started_ms` | int64 | старт сессии (UTC, ms epoch) |
| `ended_ms` | int64 | конец сессии (UTC, ms epoch) |
| `target_perfume_id` | int64 | истинный выбранный SKU (для метрик рекомендаций) |

---

## Таблица: organ_presses
Гранулярность: **1 строка = 1 событие нажатия**

| column | type | описание |
|---|---|---|
| `session_id` | int64 | ID сессии |
| `channel_index` | int64 | индекс канала (0..channels_count-1) |
| `aroma_id` | int64 | ID аромы (маппится в ноту через organ_aroma_notes_map) |
| `started_ms` | int64 | время события (UTC, ms epoch) |
| `duration_ms` | int64 | длительность события, ms |
| `intensity_end` | int64 | интенсивность (0..100) на конце события |

---

## Таблица: organ_recipes
Гранулярность: **1 строка = 1 сохранённый рецепт**

Формат `recipe`:
`0:49,1:80,2:50,3:40,4:63,5:50`
где слева `channel_index`, справа `intensity`.

| column | type | описание |
|---|---|---|
| `recipe_id` | int64 | ID рецепта |
| `session_id` | int64 | ID сессии |
| `preset_id` | int64 | ID пресета |
| `recipe` | object | строка рецепта: `0:49,1:80,...` (интенсивности по каналам) |
| `created_ms` | int64 | время сохранения, ms epoch |

---

## Таблица: organ_recipe_components
Гранулярность: **1 строка = 1 канал в рецепте**

| column | type | описание |
|---|---|---|
| `recipe_id` | int64 | ID рецепта |
| `session_id` | int64 | ID сессии |
| `channel_index` | int64 | канал |
| `intensity` | int64 | интенсивность 0..100 |

---

## Таблица: organ_aroma_notes_map
Гранулярность: **маппинг арома → ноты**

| column | type | описание |
|---|---|---|
| `aroma_id` | int64 | ID аромы |
| `note` | object | нота (в том же словаре, что и perfume_notes.note) |
| `weight` | float64 | вес соответствия (в синтетике обычно 1.0) |

---

## Рекомендуемый скоринг SKU по смеси (рецепту)
1) Строим профиль пользователя по нотам из `organ_recipe_components` и `organ_aroma_notes_map`
2) Профиль товара берём из `student_dataset/lite/perfume_notes` (например, вес ноты = `log1p(votes)`)
3) Score: cosine(u, v) или нормализованный dot-product
4) Метрики: Hit@K / MRR / NDCG@K по `organ_sessions.target_perfume_id`

---

## Размеры (факт генерации)
- sessions: 1000
- presses: 20119
- channels: 6
См. `meta.json`.
