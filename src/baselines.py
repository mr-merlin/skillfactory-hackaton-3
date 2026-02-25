"""
Три бейзлайна для сравнения с основной моделью.

B1: Top Popular  — сортировка по allVotes
B2: Note Overlap — взвешенное пересечение нот пользователя и SKU (без нормировки)
B3: Single Signal — по самой интенсивной ноте пользователя
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import TOP_N_DEFAULT


# ──────────────────────────────────────────────────────────────────────────────
# B1: Top Popular
# ──────────────────────────────────────────────────────────────────────────────

class TopPopularBaseline:
    """Рекомендует Top-N самых популярных SKU по allVotes (независимо от пользователя)."""

    def __init__(self, perfumes_df: pd.DataFrame):
        self._ranked = (
            perfumes_df[["perfume_id", "brand", "name", "allVotes"]]
            .sort_values("allVotes", ascending=False)
            .reset_index(drop=True)
        )

    def recommend(self, user_vector: np.ndarray | None = None, top_n: int = TOP_N_DEFAULT) -> list[dict]:
        rows = self._ranked.head(top_n)
        return [
            {
                "perfume_id": int(r["perfume_id"]),
                "brand": str(r["brand"]),
                "name": str(r["name"]),
                "score": float(r["allVotes"]),
            }
            for _, r in rows.iterrows()
        ]


# ──────────────────────────────────────────────────────────────────────────────
# B2: Note Overlap (взвешенное пересечение)
# ──────────────────────────────────────────────────────────────────────────────

class NoteOverlapBaseline:
    """
    Score(u, sku) = sum_{note in u ∩ sku} u[note] * sku[note]
    (ненормированный dot-product)
    """

    def __init__(
        self,
        sku_matrix: np.ndarray,
        perfume_ids: np.ndarray,
        perfumes_df: pd.DataFrame,
    ):
        self.sku_matrix = sku_matrix.astype(np.float32)
        self.perfume_ids = perfume_ids
        self.perfumes_df = perfumes_df.set_index("perfume_id")

    def recommend(self, user_vector: np.ndarray, top_n: int = TOP_N_DEFAULT) -> list[dict]:
        scores = self.sku_matrix @ user_vector.astype(np.float32)
        top_idx = np.argpartition(scores, -top_n)[-top_n:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

        results = []
        for idx in top_idx:
            pid = int(self.perfume_ids[idx])
            rec: dict = {"perfume_id": pid, "score": float(scores[idx])}
            if pid in self.perfumes_df.index:
                row = self.perfumes_df.loc[pid]
                rec["brand"] = str(row.get("brand", ""))
                rec["name"] = str(row.get("name", ""))
            results.append(rec)
        return results


# ──────────────────────────────────────────────────────────────────────────────
# B3: Single Signal — только самая интенсивная нота
# ──────────────────────────────────────────────────────────────────────────────

class SingleSignalBaseline:
    """
    Берёт одну самую «сильную» ноту в профиле пользователя,
    затем ранжирует SKU по весу этой ноты в каталоге.
    """

    def __init__(
        self,
        sku_matrix: np.ndarray,
        perfume_ids: np.ndarray,
        perfumes_df: pd.DataFrame,
    ):
        self.sku_matrix = sku_matrix.astype(np.float32)
        self.perfume_ids = perfume_ids
        self.perfumes_df = perfumes_df.set_index("perfume_id")

    def recommend(self, user_vector: np.ndarray, top_n: int = TOP_N_DEFAULT) -> list[dict]:
        best_note_idx = int(np.argmax(user_vector))
        scores = self.sku_matrix[:, best_note_idx]
        top_idx = np.argpartition(scores, -top_n)[-top_n:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

        results = []
        for idx in top_idx:
            pid = int(self.perfume_ids[idx])
            rec: dict = {"perfume_id": pid, "score": float(scores[idx])}
            if pid in self.perfumes_df.index:
                row = self.perfumes_df.loc[pid]
                rec["brand"] = str(row.get("brand", ""))
                rec["name"] = str(row.get("name", ""))
            results.append(rec)
        return results
