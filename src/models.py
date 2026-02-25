"""
Основная модель: косинусное сходство между профилем пользователя и SKU в пространстве нот.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from .config import TOP_N_DEFAULT


def _l2_norm(v: np.ndarray) -> np.ndarray:
    """Нормирует вектор(ы) по L2; избегает деления на ноль."""
    if v.ndim == 1:
        n = np.linalg.norm(v)
        return v / n if n > 0 else v
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return v / norms


class CosineRecommender:
    """
    Рекомендатель на основе косинусного сходства:
      score(u, sku) = cosine(u_vector, sku_vector)

    Оба вектора задаются в пространстве нот.
    """

    def __init__(
        self,
        sku_matrix: np.ndarray,
        perfume_ids: np.ndarray,
        perfumes_df: pd.DataFrame,
    ):
        """
        Parameters
        ----------
        sku_matrix   : np.ndarray (n_sku, n_notes) — взвешенные ноты SKU
        perfume_ids  : np.ndarray (n_sku,)  — perfume_id в том же порядке
        perfumes_df  : pd.DataFrame с колонками [perfume_id, brand, name, allVotes]
        """
        self.perfume_ids = perfume_ids
        self.sku_matrix_raw = sku_matrix
        self.sku_matrix_norm = _l2_norm(sku_matrix.astype(np.float32))

        # Быстрый lookup: perfume_id -> строка в perfumes
        self.perfumes_df = perfumes_df.set_index("perfume_id")

    def recommend(
        self,
        user_vector: np.ndarray,
        top_n: int = TOP_N_DEFAULT,
        explain: bool = True,
        exclude_ids: list[int] | None = None,
    ) -> list[dict]:
        """
        Возвращает список словарей:
          {perfume_id, brand, name, score, explanation: {note: contribution}}
        """
        u_norm = _l2_norm(user_vector)
        scores = self.sku_matrix_norm @ u_norm  # (n_sku,)

        if exclude_ids:
            mask = np.isin(self.perfume_ids, exclude_ids)
            scores[mask] = -999.0

        top_idx = np.argpartition(scores, -top_n)[-top_n:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

        results = []
        for idx in top_idx:
            pid = int(self.perfume_ids[idx])
            score = float(scores[idx])

            rec: dict = {"perfume_id": pid, "score": round(score, 4)}

            # Мета из каталога
            if pid in self.perfumes_df.index:
                row = self.perfumes_df.loc[pid]
                rec["brand"] = str(row.get("brand", ""))
                rec["name"] = str(row.get("name", ""))
                rec["all_votes"] = int(row.get("allVotes", 0))
            else:
                rec["brand"] = ""
                rec["name"] = ""
                rec["all_votes"] = 0

            results.append(rec)

        return results

    def recommend_with_explanation(
        self,
        user_vector: np.ndarray,
        note2idx: dict[str, int],
        top_n: int = TOP_N_DEFAULT,
    ) -> list[dict]:
        """Рекомендации с вкладом каждой ноты в итоговый скор."""
        recs = self.recommend(user_vector, top_n=top_n, explain=False)
        idx2note = {v: k for k, v in note2idx.items()}

        u_norm = _l2_norm(user_vector)

        for rec in recs:
            pid = rec["perfume_id"]
            row_idx = np.where(self.perfume_ids == pid)[0]
            if len(row_idx) == 0:
                rec["explanation"] = {}
                continue

            sku_vec = self.sku_matrix_norm[row_idx[0]]  # нормированный
            contributions = u_norm * sku_vec             # поэлементное произведение
            nonzero = np.where(contributions > 1e-6)[0]

            # Топ-5 нот по вкладу
            top_note_idx = nonzero[np.argsort(contributions[nonzero])[::-1][:5]]
            explanation = {
                idx2note[i]: round(float(contributions[i]), 4)
                for i in top_note_idx
                if i in idx2note
            }
            rec["explanation"] = explanation

        return recs
