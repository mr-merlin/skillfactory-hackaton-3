# DATASET_PERFUME_LITE

## Датасет: Каталог SKU/ароматов (lite)

### Назначение
Каталог используется как **пространство товаров (SKU)**, по которому строится рекомендация.
Каждый товар описан метаданными и **нотами** с весами (`votes`).

### Папка
`./student_dataset\lite`

### Файлы
- `perfumes.parquet` / `perfumes.csv` — основная таблица товаров
- `perfume_notes.parquet` / `perfume_notes.csv` — ноты товаров (длинная таблица)
- `perfume_notes_agg.parquet` — ноты, агрегированные в список по товару
- `comments_agg.parquet` — количество комментариев на товар (без текста)
- `meta.json` — служебная информация об экспорте

### Ключи и связи (важно)
**Основной идентификатор товара для всех джойнов: `perfume_id` (int).**

- `perfumes.perfume_id` — ID товара
- `perfume_notes.perfume_id` → `perfumes.perfume_id`
- `perfume_notes_agg.perfume_id` → `perfumes.perfume_id`

`parf_id` — внутренний ID источника (обычно не нужен в моделях).

---

## Таблица: perfumes
Гранулярность: **1 строка = 1 товар** (SKU-кандидат)

Схема:

| column | type | описание |
|---|---|---|
| `parf_id` | int64 | внутренний ID исходной БД (для отладки) |
| `perfume_id` | int32 | основной ID товара (использовать для всех джойнов) |
| `brand` | object | бренд |
| `name` | object | название товара/аромата |
| `clslove` | int64 | агрегированные голоса love |
| `clslike` | int64 | агрегированные голоса like |
| `clsdislike` | int64 | агрегированные голоса dislike |
| `clswinter` | int64 | сезонность (winter), агрегированные голоса |
| `clsspring` | int64 | сезонность (spring), агрегированные голоса |
| `clssummer` | int64 | сезонность (summer), агрегированные голоса |
| `clsautumn` | int64 | сезонность (autumn), агрегированные голоса |
| `clsday` | int64 | day, агрегированные голоса |
| `clsnight` | int64 | night, агрегированные голоса |
| `i_have_it` | int64 | агрегат статуса 'есть' |
| `i_had_it` | int64 | агрегат статуса 'был' |
| `i_want_it` | int64 | агрегат статуса 'хочу' |
| `allVotes` | int64 | суммарные голоса (удобно как популярность) |
| `longs1` | int64 | longevity bucket 1 |
| `longs2` | int64 | longevity bucket 2 |
| `longs3` | int64 | longevity bucket 3 |
| `longs4` | int64 | longevity bucket 4 |
| `longs5` | int64 | longevity bucket 5 |
| `sil1` | int64 | sillage bucket 1 |
| `sil2` | int64 | sillage bucket 2 |
| `sil3` | int64 | sillage bucket 3 |
| `sil4` | int64 | sillage bucket 4 |
| `upd` | datetime64[ns] | время обновления записи |
| `url` | object | ссылка на страницу (если есть) |
| `status` | int64 | статус записи (семантика зависит от источника) |

---

## Таблица: perfume_notes
Гранулярность: **1 строка = 1 нота у 1 товара**

Схема:

| column | type | описание |
|---|---|---|
| `perfume_id` | int32 | ID товара (join -> perfumes.perfume_id) |
| `note` | object | название ноты (нормализовано: lower-case, пробелы) |
| `votes` | int64 | вес/популярность ноты у товара |
| `upd` | datetime64[ns] | время обновления записи |

Замечания:
- `votes` можно использовать как вес ноты (например, `log1p(votes)`).
- Ноты — “мешок слов” (bag-of-notes), порядок не задан.

---

## Таблица: perfume_notes_agg
Гранулярность: **1 строка = 1 товар**

Колонки:
- `perfume_id` (int)
- `notes` (list[{note, votes}]) — список объектов `{"note": str, "votes": int}` (обычно отсортирован по votes desc)

---

## Быстрые бейзлайны
1) **Top Popular**: сортировать по `allVotes` (или `clslove + clslike`)
2) **Note overlap**: пересечение нот пользователя и SKU (с весами)

---

## Размеры (факт экспорта)
- perfumes: 5000
- notes: 39693
- comments_mode: agg
- notes_min_votes: 0

Служебная информация: см. `meta.json`.
