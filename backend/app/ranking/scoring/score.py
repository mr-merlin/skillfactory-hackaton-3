from typing import Optional

import numpy as np
import pandas as pd


def build_sku_vectors(
    perfume_notes: pd.DataFrame,
    normalize: str = "log1p",
) -> tuple[dict[int, np.ndarray], dict[str, int], list[str]]:
    notes_ser = perfume_notes["note"].astype(str).str.strip().str.lower()
    perfume_notes = perfume_notes.assign(note_norm=notes_ser)
    agg = perfume_notes.groupby(["perfume_id", "note_norm"], as_index=False)["votes"].sum()
    pivot = agg.pivot_table(index="perfume_id", columns="note_norm", values="votes", fill_value=0, aggfunc="sum")

    idx_to_note = sorted(pivot.columns.tolist())
    note_to_idx = {n: i for i, n in enumerate(idx_to_note)}
    pivot = pivot.reindex(columns=idx_to_note, fill_value=0)

    vals = pivot.values.copy()
    if normalize == "log1p":
        vals = np.log1p(np.maximum(vals, 0))
    elif normalize == "l2":
        norms = np.linalg.norm(vals, axis=1, keepdims=True)
        vals = np.where(norms > 0, vals / norms, vals)

    perfume_vectors = {int(pid): vals[i].astype(np.float64) for i, pid in enumerate(pivot.index)}
    return perfume_vectors, note_to_idx, idx_to_note


def build_sku_matrix(
    perfume_vectors: dict[int, np.ndarray],
    perfume_ids: list[int] | None = None,
) -> tuple[np.ndarray, list[int]]:
    if perfume_ids is None:
        perfume_ids = sorted(perfume_vectors.keys())
    D = len(perfume_vectors[perfume_ids[0]])
    M = np.zeros((len(perfume_ids), D), dtype=np.float64)
    for i, pid in enumerate(perfume_ids):
        M[i] = perfume_vectors[pid]
    return M, perfume_ids


def _prepare_user_vec(user_vec: dict[str, float], note_to_idx: dict[str, int], cosine: bool):
    size = len(note_to_idx)
    u = np.zeros(size, dtype=np.float64)
    for note, val in user_vec.items():
        idx = note_to_idx.get(note.strip().lower())
        if idx is not None:
            u[idx] = val
    if np.all(u == 0):
        return None, None
    norm = np.linalg.norm(u)
    if cosine:
        if norm <= 0:
            return None, None
        u = u / norm
    return u, norm


def _compute_scores(u, M, perfume_ids, cosine: bool):
    if cosine:
        norms = np.linalg.norm(M, axis=1, keepdims=True)
        M_unit = np.where(norms > 0, M / norms, M)
        return M_unit @ u, M_unit
    return M @ u, M


def _rerank_with_popularity(scores_arr, perfume_ids, popularity, retrieval_k, blend_alpha, top_n):
    k = min(retrieval_k, len(perfume_ids))
    top_k = np.argsort(-scores_arr)[:k]
    max_pop = max(popularity.values()) if popularity else 1.0
    reranked = []
    for idx in top_k:
        pid = perfume_ids[idx]
        blended = blend_alpha * float(scores_arr[idx]) + (1 - blend_alpha) * popularity.get(pid, 0.0) / max_pop
        reranked.append((pid, blended, idx))
    reranked.sort(key=lambda x: -x[1])
    return reranked[:top_n]


def score_skus(
    user_vec: dict[str, float],
    perfume_vectors: dict[int, np.ndarray],
    note_to_idx: dict[str, int],
    idx_to_note: list[str],
    top_n: int = 10,
    use_cosine: bool = True,
    sku_matrix: np.ndarray | None = None,
    sku_ids: list[int] | None = None,
    popularity: dict[int, float] | None = None,
    retrieval_k: int = 100,
    blend_alpha: float = 0.5,
) -> list[tuple[int, float]]:
    u, _ = _prepare_user_vec(user_vec, note_to_idx, use_cosine)
    if u is None:
        return []

    if sku_matrix is not None and sku_ids is not None:
        M, pids = sku_matrix, sku_ids
    else:
        M, pids = build_sku_matrix(perfume_vectors)

    scores_arr, _ = _compute_scores(u, M, pids, use_cosine)

    if popularity is None:
        top_indices = np.argsort(-scores_arr)[:top_n]
        return [(pids[i], float(scores_arr[i])) for i in top_indices]

    reranked = _rerank_with_popularity(scores_arr, pids, popularity, retrieval_k, blend_alpha, top_n)
    return [(pid, sc) for pid, sc, _ in reranked]


def score_skus_with_explanation(
    user_vec: dict[str, float],
    perfume_vectors: dict[int, np.ndarray],
    note_to_idx: dict[str, int],
    idx_to_note: list[str],
    top_n: int = 10,
    use_cosine: bool = True,
    explanation_top_k_notes: int = 5,
    sku_matrix: np.ndarray | None = None,
    sku_ids: list[int] | None = None,
    popularity: dict[int, float] | None = None,
    retrieval_k: int = 100,
    blend_alpha: float = 0.5,
) -> list[tuple[int, float, list[dict]]]:
    u, _ = _prepare_user_vec(user_vec, note_to_idx, use_cosine)
    if u is None:
        return []

    if sku_matrix is not None and sku_ids is not None:
        M, pids = sku_matrix, sku_ids
    else:
        M, pids = build_sku_matrix(perfume_vectors)

    scores_arr, M_unit = _compute_scores(u, M, pids, use_cosine)
    pid_to_idx = {pid: i for i, pid in enumerate(pids)}

    if popularity is None:
        top_indices = np.argsort(-scores_arr)[:top_n]
        top_pids = [(pids[idx], float(scores_arr[idx])) for idx in top_indices]
    else:
        reranked = _rerank_with_popularity(scores_arr, pids, popularity, retrieval_k, blend_alpha, top_n)
        top_pids = [(pid, sc) for pid, sc, _ in reranked]

    results = []
    for pid, sc in top_pids:
        idx = pid_to_idx[pid]
        if np.linalg.norm(M[idx]) <= 0:
            results.append((pid, sc, []))
            continue
        contrib = u * M_unit[idx]
        best = np.argsort(-contrib)[:explanation_top_k_notes]
        explanation = [
            {"note": idx_to_note[i], "contribution": round(float(contrib[i]), 4)}
            for i in best if contrib[i] > 0
        ]
        results.append((pid, sc, explanation))
    return results
