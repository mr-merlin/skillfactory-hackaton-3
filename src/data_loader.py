"""
Загрузка и предобработка всех датасетов.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATA_DIR

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Внутренние хелперы
# ──────────────────────────────────────────────────────────────────────────────

def _load(filename: str, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    path = data_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")
    logger.debug("Загружаю %s", path)
    return pd.read_parquet(path)


# ──────────────────────────────────────────────────────────────────────────────
# Публичный API
# ──────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def load_all(data_dir: str | None = None) -> dict[str, pd.DataFrame]:
    """
    Загружает все таблицы датасета и возвращает словарь DataFrames.
    Результат кэшируется — повторные вызовы возвращают те же объекты.
    """
    d = Path(data_dir) if data_dir else DATA_DIR
    tables: dict[str, pd.DataFrame] = {}

    # ── Каталог парфюмерии ─────────────────────────────────────────────────────
    perfumes = _load("perfumes.parquet", d)
    # Дедупликация по perfume_id — оставляем строку с максимальным allVotes
    perfumes = (
        perfumes
        .sort_values("allVotes", ascending=False)
        .drop_duplicates(subset=["perfume_id"], keep="first")
        .reset_index(drop=True)
    )
    tables["perfumes"] = perfumes
    tables["perfume_notes"] = _load("perfume_notes.parquet", d)
    tables["perfume_notes_agg"] = _load("perfume_notes_agg.parquet", d)

    # ── Логи органа ────────────────────────────────────────────────────────────
    tables["sessions"] = _load("organ_sessions.parquet", d)
    tables["presses"] = _load("organ_presses.parquet", d)
    tables["recipes"] = _load("organ_recipes.parquet", d)
    tables["recipe_components"] = _load("organ_recipe_components.parquet", d)
    tables["aroma_notes_map"] = _load("organ_aroma_notes_map.parquet", d)
    tables["aromas"] = _load("organ_aromas.parquet", d)
    tables["feedback"] = _load("organ_feedback.parquet", d)

    logger.info("Загружено %d таблиц из %s", len(tables), d)
    return tables


def build_note_index(
    perfume_notes: pd.DataFrame,
    organ_notes: list[str] | None = None,
) -> tuple[dict[str, int], np.ndarray, np.ndarray]:
    """
    Строит индекс нот и матрицу SKU x нот (взвешенная log1p).
    
    Parameters
    ----------
    perfume_notes : DataFrame с колонками [perfume_id, note, votes]
    organ_notes   : если задан, ограничивает пространство нот этими значениями
                    (улучшает качество при работе с Парфюмерным органом)
    
    Returns
    -------
    note2idx   : dict[str, int]
    sku_matrix : np.ndarray  shape (n_sku, n_notes), строки = log1p(votes)
    perfume_ids: np.ndarray  shape (n_sku,)
    """
    if organ_notes is not None:
        notes_all = sorted(organ_notes)
        pn_filtered = perfume_notes[perfume_notes["note"].isin(organ_notes)]
    else:
        notes_all = sorted(perfume_notes["note"].unique())
        pn_filtered = perfume_notes

    note2idx = {n: i for i, n in enumerate(notes_all)}

    # Все SKU из каталога (даже если без нот в данном словаре)
    perfume_ids = perfume_notes["perfume_id"].unique()
    pid2row = {pid: i for i, pid in enumerate(perfume_ids)}

    matrix = np.zeros((len(perfume_ids), len(notes_all)), dtype=np.float32)
    for row in pn_filtered.itertuples(index=False):
        if row.perfume_id not in pid2row:
            continue
        r = pid2row[row.perfume_id]
        c = note2idx.get(row.note)
        if c is not None:
            matrix[r, c] = np.log1p(row.votes)

    return note2idx, matrix, np.array(perfume_ids)


def parse_recipe_string(recipe_str: str) -> dict[int, int]:
    """
    Парсит строку вида '0:49,1:80,2:50' в {channel_index: intensity}.
    """
    result: dict[int, int] = {}
    for part in recipe_str.split(","):
        part = part.strip()
        if ":" not in part:
            continue
        ch, val = part.split(":", 1)
        result[int(ch)] = int(val)
    return result
