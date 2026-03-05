import pickle
from pathlib import Path

import numpy as np
import lightgbm as lgb


def extract_features(user_vec: dict[str, float], sku_vec: np.ndarray,
                     note_to_idx: dict[str, int], idx_to_note: list[str],
                     popularity: float = 0.0,
                     norm_sku_vec: np.ndarray | None = None,
                     norm_note_to_idx: dict[str, int] | None = None,
                     sku_meta: dict | None = None) -> np.ndarray:
    dim = len(note_to_idx)
    u = np.zeros(dim, dtype=np.float64)
    for note, val in user_vec.items():
        i = note_to_idx.get(note.strip().lower())
        if i is not None:
            u[i] = val

    u_norm = np.linalg.norm(u)
    s_norm = np.linalg.norm(sku_vec)

    if u_norm > 0 and s_norm > 0:
        cosine = np.dot(u, sku_vec) / (u_norm * s_norm)
    else:
        cosine = 0.0

    dot = np.dot(u, sku_vec)

    u_nonzero = set(np.where(u > 0)[0])
    s_nonzero = set(np.where(sku_vec > 0)[0])
    overlap = len(u_nonzero & s_nonzero)
    overlap_ratio = overlap / max(len(u_nonzero), 1)

    u_top5 = set(np.argsort(-u)[:5])
    s_top5 = set(np.argsort(-sku_vec)[:5])
    top5_match = len(u_top5 & s_top5)

    u_top1 = np.argmax(u) if u_norm > 0 else -1
    s_top1 = np.argmax(sku_vec) if s_norm > 0 else -1
    top1_match = float(u_top1 == s_top1 and u_top1 >= 0)

    s_active = len(s_nonzero)
    u_active = len(u_nonzero)

    if u_norm > 0 and s_norm > 0:
        u_unit = u / u_norm
        s_log = np.log1p(np.maximum(sku_vec, 0))
        s_log_norm = np.linalg.norm(s_log)
        if s_log_norm > 0:
            weighted_cos = np.dot(u_unit, s_log / s_log_norm)
        else:
            weighted_cos = 0.0
    else:
        weighted_cos = 0.0

    u_entropy = 0.0
    if u_norm > 0:
        p = np.abs(u) / np.sum(np.abs(u))
        p = p[p > 0]
        u_entropy = -np.sum(p * np.log(p))

    cosine_norm = 0.0
    if norm_sku_vec is not None and norm_note_to_idx is not None:
        un = np.zeros(len(norm_note_to_idx), dtype=np.float64)
        for note, val in user_vec.items():
            i = norm_note_to_idx.get(note.strip().lower())
            if i is not None:
                un[i] = val
        un_norm = np.linalg.norm(un)
        sn_norm = np.linalg.norm(norm_sku_vec)
        if un_norm > 0 and sn_norm > 0:
            cosine_norm = np.dot(un, norm_sku_vec) / (un_norm * sn_norm)

    # --- Non-similarity features (SKU properties) ---
    meta = sku_meta or {}

    collision_group_size = meta.get("collision_group_size", 0.0)
    pop_rank_in_group = meta.get("pop_rank_in_group", 0.5)
    note_count = meta.get("note_count", 0.0)
    note_entropy = meta.get("note_entropy", 0.0)
    top3_concentration = meta.get("top3_concentration", 0.0)
    love_ratio = meta.get("love_ratio", 0.5)
    longevity_idx = meta.get("longevity_idx", 0.5)
    sillage_idx = meta.get("sillage_idx", 0.5)

    return np.array([
        cosine,
        dot,
        overlap,
        overlap_ratio,
        top5_match,
        top1_match,
        s_active,
        u_active,
        popularity,
        weighted_cos,
        u_entropy,
        cosine_norm,
        # Non-similarity features
        collision_group_size,
        pop_rank_in_group,
        note_count,
        note_entropy,
        top3_concentration,
        love_ratio,
        longevity_idx,
        sillage_idx,
    ], dtype=np.float64)


FEATURE_NAMES = [
    "cosine", "dot", "overlap", "overlap_ratio", "top5_match",
    "top1_match", "s_active", "u_active", "popularity",
    "weighted_cos", "u_entropy", "cosine_norm",
    # Non-similarity features
    "collision_group_size", "pop_rank_in_group",
    "note_count", "note_entropy", "top3_concentration",
    "love_ratio", "longevity_idx", "sillage_idx",
]


class GBMScorer:
    def __init__(self, model_path, note_to_idx, idx_to_note, perfume_vectors,
                 popularity=None, norm_perfume_vectors=None, norm_note_to_idx=None,
                 sku_meta=None):
        self.note_to_idx = note_to_idx
        self.idx_to_note = idx_to_note
        self.perfume_vectors = perfume_vectors
        self.popularity = popularity or {}
        self.norm_perfume_vectors = norm_perfume_vectors
        self.norm_note_to_idx = norm_note_to_idx
        self.sku_meta = sku_meta or {}

        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

        self.pids = sorted(perfume_vectors.keys())
        self.max_pop = max(self.popularity.values()) if self.popularity else 1.0

    def score(self, user_vec, top_n=10, candidate_pids=None):
        if not user_vec:
            return []

        pids = candidate_pids if candidate_pids is not None else self.pids
        feats = []
        for pid in pids:
            if pid not in self.perfume_vectors:
                continue
            pop = self.popularity.get(pid, 0.0) / self.max_pop
            nv = self.norm_perfume_vectors.get(pid) if self.norm_perfume_vectors else None
            f = extract_features(user_vec, self.perfume_vectors[pid],
                                 self.note_to_idx, self.idx_to_note, pop,
                                 norm_sku_vec=nv, norm_note_to_idx=self.norm_note_to_idx,
                                 sku_meta=self.sku_meta.get(pid))
            feats.append((pid, f))

        if not feats:
            return []

        feat_pids, feat_vecs = zip(*feats)
        X = np.vstack(feat_vecs)
        scores = self.model.predict(X)

        top_indices = np.argsort(-scores)[:top_n]
        return [(feat_pids[i], float(scores[i])) for i in top_indices]
