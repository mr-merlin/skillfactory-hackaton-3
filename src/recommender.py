"""
Центральный класс системы рекомендаций — объединяет загрузку данных,
построение индексов, профилей и скоринг.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATA_DIR, TOP_N_DEFAULT, ALPHA_RECIPE
from .data_loader import load_all, build_note_index
from .features import build_user_profile, profile_from_recipe_string
from .models import CosineRecommender
from .baselines import TopPopularBaseline, NoteOverlapBaseline, SingleSignalBaseline

logger = logging.getLogger(__name__)


class PerfumeRecommendationSystem:
    """
    Единая точка входа для рекомендательной системы.

    Usage
    -----
    >>> prs = PerfumeRecommendationSystem()
    >>> prs.load()
    >>> recs = prs.recommend_by_session(42, top_n=5)
    >>> recs = prs.recommend_by_recipe("0:49,1:80,2:50,3:40,4:63,5:50", top_n=5)
    """

    def __init__(self, data_dir: str | None = None):
        self.data_dir = str(data_dir) if data_dir else str(DATA_DIR)
        self._loaded = False

    # ──────────────────────────────────────────────────────────────────────────
    # Загрузка / инициализация
    # ──────────────────────────────────────────────────────────────────────────

    def load(self) -> "PerfumeRecommendationSystem":
        """Загружает данные и строит индексы. Вызвать один раз перед использованием."""
        t0 = time.time()
        tables = load_all(self.data_dir)

        self.perfumes: pd.DataFrame = tables["perfumes"]
        self.perfume_notes: pd.DataFrame = tables["perfume_notes"]
        self.sessions: pd.DataFrame = tables["sessions"]
        self.presses: pd.DataFrame = tables["presses"]
        self.recipe_components: pd.DataFrame = tables["recipe_components"]
        self.aroma_notes_map: pd.DataFrame = tables["aroma_notes_map"]
        self.aromas: pd.DataFrame = tables["aromas"]
        self.feedback: pd.DataFrame = tables["feedback"]

        # Строим индекс нот — ТОЛЬКО по нотам органа для точного сопоставления
        organ_notes = sorted(self.aroma_notes_map["note"].unique().tolist())
        logger.info("Ноты органа (%d): %s", len(organ_notes), organ_notes)
        self.note2idx, self.sku_matrix, self.perfume_id_arr = build_note_index(
            self.perfume_notes, organ_notes=organ_notes
        )
        logger.info("Индекс нот: %d уникальных нот, %d SKU", len(self.note2idx), len(self.perfume_id_arr))

        # Основная модель
        self.model = CosineRecommender(self.sku_matrix, self.perfume_id_arr, self.perfumes)

        # Бейзлайны
        self.b1_popular = TopPopularBaseline(self.perfumes)
        self.b2_overlap = NoteOverlapBaseline(self.sku_matrix, self.perfume_id_arr, self.perfumes)
        self.b3_single = SingleSignalBaseline(self.sku_matrix, self.perfume_id_arr, self.perfumes)

        self._loaded = True
        logger.info("Система инициализирована за %.2f сек.", time.time() - t0)
        return self

    def _check_loaded(self):
        if not self._loaded:
            raise RuntimeError("Вызовите .load() перед использованием системы.")

    # ──────────────────────────────────────────────────────────────────────────
    # Основные методы инференса
    # ──────────────────────────────────────────────────────────────────────────

    def recommend_by_session(
        self,
        session_id: int,
        top_n: int = TOP_N_DEFAULT,
        alpha: float = ALPHA_RECIPE,
        use_presses: bool = True,
        explain: bool = True,
    ) -> list[dict]:
        """Рекомендации по session_id (из базы органа)."""
        self._check_loaded()
        t0 = time.perf_counter()

        user_vec = build_user_profile(
            session_id=session_id,
            recipe_components=self.recipe_components,
            presses=self.presses,
            aroma_notes_map=self.aroma_notes_map,
            aromas=self.aromas,
            note2idx=self.note2idx,
            alpha=alpha,
            use_presses=use_presses,
        )

        if explain:
            recs = self.model.recommend_with_explanation(user_vec, self.note2idx, top_n)
        else:
            recs = self.model.recommend(user_vec, top_n)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        for r in recs:
            r["latency_ms"] = round(elapsed_ms, 2)
        return recs

    def recommend_by_recipe(
        self,
        recipe_str: str,
        top_n: int = TOP_N_DEFAULT,
        explain: bool = True,
    ) -> list[dict]:
        """Рекомендации из строки рецепта, например '0:49,1:80,2:50,3:40,4:63,5:50'."""
        self._check_loaded()
        t0 = time.perf_counter()

        user_vec = profile_from_recipe_string(
            recipe_str, self.aroma_notes_map, self.aromas, self.note2idx
        )

        if explain:
            recs = self.model.recommend_with_explanation(user_vec, self.note2idx, top_n)
        else:
            recs = self.model.recommend(user_vec, top_n)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        for r in recs:
            r["latency_ms"] = round(elapsed_ms, 2)
        return recs

    # ──────────────────────────────────────────────────────────────────────────
    # Геттеры для оценки (для evaluate.py)
    # ──────────────────────────────────────────────────────────────────────────

    def get_session_ids_list(self, session_id: int) -> list[int]:
        """Возвращает список perfume_id из рекомендаций основной модели."""
        recs = self.recommend_by_session(session_id, explain=False)
        return [r["perfume_id"] for r in recs]

    def get_b1_list(self, session_id: int | None = None) -> list[int]:
        recs = self.b1_popular.recommend()
        return [r["perfume_id"] for r in recs]

    def get_b2_list(self, session_id: int) -> list[int]:
        user_vec = build_user_profile(
            session_id=session_id,
            recipe_components=self.recipe_components,
            presses=self.presses,
            aroma_notes_map=self.aroma_notes_map,
            aromas=self.aromas,
            note2idx=self.note2idx,
        )
        recs = self.b2_overlap.recommend(user_vec)
        return [r["perfume_id"] for r in recs]

    def get_b3_list(self, session_id: int) -> list[int]:
        user_vec = build_user_profile(
            session_id=session_id,
            recipe_components=self.recipe_components,
            presses=self.presses,
            aroma_notes_map=self.aroma_notes_map,
            aromas=self.aromas,
            note2idx=self.note2idx,
        )
        recs = self.b3_single.recommend(user_vec)
        return [r["perfume_id"] for r in recs]
