import pickle
from pathlib import Path

import numpy as np
import lightgbm as lgb


def extract_features(user_vec: dict[str, float], sku_vec: np.ndarray,
                     note_to_idx: dict[str, int], idx_to_note: list[str],
                     popularity: float = 0.0) -> np.ndarray:
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
    ], dtype=np.float64)


FEATURE_NAMES = [
    "cosine", "dot", "overlap", "overlap_ratio", "top5_match",
    "top1_match", "s_active", "u_active", "popularity",
    "weighted_cos", "u_entropy",
]


class GBMScorer:
    def __init__(self, model_path, note_to_idx, idx_to_note, perfume_vectors, popularity=None):
        self.note_to_idx = note_to_idx
        self.idx_to_note = idx_to_note
        self.perfume_vectors = perfume_vectors
        self.popularity = popularity or {}

        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

        self.pids = sorted(perfume_vectors.keys())
        self.max_pop = max(self.popularity.values()) if self.popularity else 1.0

    def score(self, user_vec, top_n=10):
        if not user_vec:
            return []

        feats = []
        for pid in self.pids:
            pop = self.popularity.get(pid, 0.0) / self.max_pop
            f = extract_features(user_vec, self.perfume_vectors[pid],
                                 self.note_to_idx, self.idx_to_note, pop)
            feats.append(f)

        X = np.vstack(feats)
        scores = self.model.predict(X)

        top_indices = np.argsort(-scores)[:top_n]
        return [(self.pids[i], float(scores[i])) for i in top_indices]
