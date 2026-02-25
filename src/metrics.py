"""
Метрики качества рекомендаций: Hit@K, MRR, NDCG@K.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def hit_at_k(recommended_ids: Sequence[int], relevant_id: int, k: int) -> float:
    """1 если relevant_id входит в первые k рекомендаций."""
    return float(relevant_id in recommended_ids[:k])


def reciprocal_rank(recommended_ids: Sequence[int], relevant_id: int) -> float:
    """1 / (rank+1) если relevant_id найден, иначе 0."""
    try:
        rank = list(recommended_ids).index(relevant_id)
        return 1.0 / (rank + 1)
    except ValueError:
        return 0.0


def ndcg_at_k(recommended_ids: Sequence[int], relevant_id: int, k: int) -> float:
    """
    NDCG@K для одного релевантного документа (бинарная релевантность).
    DCG = 1/log2(rank+2) если найден, иначе 0.
    IDCG = 1/log2(2) = 1.0 (идеальная позиция = 1).
    """
    ids = list(recommended_ids[:k])
    if relevant_id not in ids:
        return 0.0
    rank = ids.index(relevant_id)
    dcg = 1.0 / math.log2(rank + 2)
    idcg = 1.0 / math.log2(2)  # = 1.0
    return dcg / idcg


def evaluate(
    recommend_fn,
    sessions: "pd.DataFrame",
    top_n: int = 10,
    k_values: list[int] | None = None,
) -> dict[str, float]:
    """
    Вычисляет Hit@K, MRR, NDCG@K по всем сессиям с `target_perfume_id`.

    Parameters
    ----------
    recommend_fn : callable(session_id) -> list[int]  (perfume_ids в порядке рейтинга)
    sessions     : DataFrame с колонками [session_id, target_perfume_id]
    top_n        : глубина выдачи
    k_values     : список K для Hit@K и NDCG@K

    Returns
    -------
    dict с метриками
    """
    if k_values is None:
        k_values = [5, 10]

    hits = {k: [] for k in k_values}
    ndcgs = {k: [] for k in k_values}
    rrs = []

    for session_id, target_id in zip(sessions["session_id"], sessions["target_perfume_id"]):
        try:
            recs = recommend_fn(int(session_id))
        except Exception:
            recs = []

        for k in k_values:
            hits[k].append(hit_at_k(recs, int(target_id), k))
            ndcgs[k].append(ndcg_at_k(recs, int(target_id), k))

        rrs.append(reciprocal_rank(recs, int(target_id)))

    result: dict[str, float] = {"MRR": float(np.mean(rrs))}
    for k in k_values:
        result[f"Hit@{k}"] = float(np.mean(hits[k]))
        result[f"NDCG@{k}"] = float(np.mean(ndcgs[k]))

    return result


def diversity_at_k(
    recommended_ids: list[int],
    sku_matrix_norm: "np.ndarray",
    pid2row: dict[int, int],
    k: int,
) -> float:
    """
    Intra-List Diversity@K: среднее попарное косинусное расстояние (1 - cosine_sim)
    между SKU в выдаче. Чем выше — тем разнообразнее выдача.
    """
    ids = [pid for pid in recommended_ids[:k] if pid in pid2row]
    if len(ids) < 2:
        return 0.0
    vectors = np.array([sku_matrix_norm[pid2row[pid]] for pid in ids])
    # попарные косинусные сходства
    sims = vectors @ vectors.T
    n = len(ids)
    # среднее расстояние по всем парам (i < j)
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += 1.0 - float(sims[i, j])
            count += 1
    return total / count if count > 0 else 0.0


def stability_score(
    recommend_fn_a,
    recommend_fn_b,
    session_ids: list[int],
    k: int = 10,
) -> float:
    """
    Стабильность выдачи: средняя доля совпадающих SKU в Top-K
    между двумя вариантами рекомендаций (например, слегка изменённый рецепт).
    Диапазон: 0 (полное несовпадение) .. 1 (идентичные списки).
    """
    scores = []
    for sid in session_ids:
        recs_a = set(recommend_fn_a(sid)[:k])
        recs_b = set(recommend_fn_b(sid)[:k])
        if not recs_a and not recs_b:
            scores.append(1.0)
        elif not recs_a or not recs_b:
            scores.append(0.0)
        else:
            overlap = len(recs_a & recs_b) / k
            scores.append(overlap)
    return float(np.mean(scores)) if scores else 0.0


def print_metrics_table(results: dict[str, dict[str, float]]) -> None:
    """Красиво выводит таблицу сравнения моделей."""
    import pandas as pd
    df = pd.DataFrame(results).T.round(4)
    print("\n" + "=" * 60)
    print("СРАВНЕНИЕ МЕТРИК КАЧЕСТВА РЕКОМЕНДАЦИЙ")
    print("=" * 60)
    print(df.to_string())
    print("=" * 60 + "\n")
