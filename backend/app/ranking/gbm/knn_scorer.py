"""kNN-augmented GBM Scorer for production inference (v2).

Uses kitchen-sink v2 kNN features (multi-K + dual-space, 18 features) on top of
the standard 20 GBM features, achieving 14.0% Hit@10 (up from 9.5%).
"""

import pickle
from pathlib import Path

import numpy as np
import lightgbm as lgb

from .ranker import extract_features, FEATURE_NAMES


KNN_FEATURE_NAMES = [
    "raw_k20_match", "raw_k20_sim", "raw_k20_weighted",
    "raw_k20_maxsim", "raw_k20_popw", "raw_k20_div",
    "raw_k50_match", "raw_k50_sim", "raw_k50_weighted",
    "raw_k50_maxsim", "raw_k50_popw", "raw_k50_div",
    "norm_k50_match", "norm_k50_sim", "norm_k50_weighted",
    "norm_k50_maxsim", "norm_k50_popw", "norm_k50_div",
]


class KnnGBMScorer:
    """GBM scorer augmented with kNN features from training user embeddings."""

    def __init__(self, model_path, knn_data_path,
                 note_to_idx, idx_to_note, perfume_vectors,
                 popularity=None, norm_perfume_vectors=None, norm_note_to_idx=None,
                 sku_meta=None):
        self.note_to_idx = note_to_idx
        self.idx_to_note = idx_to_note
        self.perfume_vectors = perfume_vectors
        self.popularity = popularity or {}
        self.norm_perfume_vectors = norm_perfume_vectors
        self.norm_note_to_idx = norm_note_to_idx
        self.sku_meta = sku_meta or {}

        # Load GBM model
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

        # Load kNN data
        with open(knn_data_path, "rb") as f:
            knn_data = pickle.load(f)

        self.train_U_raw_n = knn_data["train_U_raw_n"]
        self.train_U_norm_n = knn_data["train_U_norm_n"]
        self.train_targets = knn_data["train_targets"]
        self.train_target_pop = knn_data.get("train_target_pop")
        self.knn_note_to_idx = knn_data["note_to_idx"]
        self.knn_norm_nti = knn_data["norm_nti"]

        self.dim_raw = self.train_U_raw_n.shape[1]
        self.dim_norm = self.train_U_norm_n.shape[1]

        self.pids = sorted(perfume_vectors.keys())
        self.max_pop = max(self.popularity.values()) if self.popularity else 1.0

        # Fallback: compute train_target_pop if not in saved data (backward compat)
        if self.train_target_pop is None:
            self.train_target_pop = np.array([
                self.popularity.get(int(t), 0) / self.max_pop
                for t in self.train_targets
            ])

    def score(self, user_vec, top_n=10, candidate_pids=None, explain=True):
        """Score candidates and return top_n (pid, score) or (pid, score, explanation) tuples."""
        if not user_vec:
            return []

        pids = candidate_pids if candidate_pids is not None else self.pids

        # Pre-compute user vectors in both spaces (once per query)
        u_raw = np.zeros(self.dim_raw, dtype=np.float64)
        for note, val in user_vec.items():
            idx = self.knn_note_to_idx.get(note.strip().lower())
            if idx is not None:
                u_raw[idx] = val
        u_raw_n = u_raw / max(np.linalg.norm(u_raw), 1e-9)

        u_norm = np.zeros(self.dim_norm, dtype=np.float64)
        for note, val in user_vec.items():
            idx = self.knn_norm_nti.get(note.strip().lower())
            if idx is not None:
                u_norm[idx] = val
        u_norm_n = u_norm / max(np.linalg.norm(u_norm), 1e-9)

        # Pre-compute all kNN similarities at once for efficiency
        raw_sims = self.train_U_raw_n @ u_raw_n   # (n_train,)
        norm_sims = self.train_U_norm_n @ u_norm_n  # (n_train,)

        # Pre-sort for K=50 (covers K=20 as subset)
        raw_top50 = np.argsort(-raw_sims)[:50]
        norm_top50 = np.argsort(-norm_sims)[:50]
        raw_top20 = raw_top50[:20]

        # Pre-compute shared values
        raw_sims_k20 = raw_sims[raw_top20]
        raw_sims_k50 = raw_sims[raw_top50]
        norm_sims_k50 = norm_sims[norm_top50]
        raw_targets_k20 = self.train_targets[raw_top20]
        raw_targets_k50 = self.train_targets[raw_top50]
        norm_targets_k50 = self.train_targets[norm_top50]
        raw_pop_k20 = self.train_target_pop[raw_top20]
        raw_pop_k50 = self.train_target_pop[raw_top50]
        norm_pop_k50 = self.train_target_pop[norm_top50]
        raw_avg_sim_k20 = float(np.mean(raw_sims_k20))
        raw_avg_sim_k50 = float(np.mean(raw_sims_k50))
        norm_avg_sim_k50 = float(np.mean(norm_sims_k50))
        raw_sum_k20 = float(np.sum(raw_sims_k20))
        raw_sum_k50 = float(np.sum(raw_sims_k50))
        norm_sum_k50 = float(np.sum(norm_sims_k50))

        # Pre-compute diversity (shared across all PIDs)
        raw_k20_diversity = len(set(int(t) for t in raw_targets_k20)) / 20.0
        raw_k50_diversity = len(set(int(t) for t in raw_targets_k50)) / 50.0
        norm_k50_diversity = len(set(int(t) for t in norm_targets_k50)) / 50.0

        feats = []
        valid_pids = []
        for pid in pids:
            if pid not in self.perfume_vectors:
                continue

            pop = self.popularity.get(pid, 0.0) / self.max_pop
            nv = self.norm_perfume_vectors.get(pid) if self.norm_perfume_vectors else None
            base = extract_features(user_vec, self.perfume_vectors[pid],
                                    self.note_to_idx, self.idx_to_note, pop,
                                    norm_sku_vec=nv, norm_note_to_idx=self.norm_note_to_idx,
                                    sku_meta=self.sku_meta.get(pid))

            # --- raw K=20: 6 features ---
            rk20_mask = (raw_targets_k20 == pid)
            rk20_match = float(np.sum(rk20_mask)) / 20.0
            rk20_w = float(np.sum(raw_sims_k20[rk20_mask])) / max(raw_sum_k20, 1e-9)
            rk20_matching_sims = raw_sims_k20[rk20_mask]
            rk20_maxsim = float(np.max(rk20_matching_sims)) if len(rk20_matching_sims) > 0 else 0.0
            rk20_popw = float(np.sum(raw_pop_k20[rk20_mask])) / 20.0

            # --- raw K=50: 6 features ---
            rk50_mask = (raw_targets_k50 == pid)
            rk50_match = float(np.sum(rk50_mask)) / 50.0
            rk50_w = float(np.sum(raw_sims_k50[rk50_mask])) / max(raw_sum_k50, 1e-9)
            rk50_matching_sims = raw_sims_k50[rk50_mask]
            rk50_maxsim = float(np.max(rk50_matching_sims)) if len(rk50_matching_sims) > 0 else 0.0
            rk50_popw = float(np.sum(raw_pop_k50[rk50_mask])) / 50.0

            # --- norm K=50: 6 features ---
            nk50_mask = (norm_targets_k50 == pid)
            nk50_match = float(np.sum(nk50_mask)) / 50.0
            nk50_w = float(np.sum(norm_sims_k50[nk50_mask])) / max(norm_sum_k50, 1e-9)
            nk50_matching_sims = norm_sims_k50[nk50_mask]
            nk50_maxsim = float(np.max(nk50_matching_sims)) if len(nk50_matching_sims) > 0 else 0.0
            nk50_popw = float(np.sum(norm_pop_k50[nk50_mask])) / 50.0

            knn = np.array([
                rk20_match, raw_avg_sim_k20, rk20_w,
                rk20_maxsim, rk20_popw, raw_k20_diversity,
                rk50_match, raw_avg_sim_k50, rk50_w,
                rk50_maxsim, rk50_popw, raw_k50_diversity,
                nk50_match, norm_avg_sim_k50, nk50_w,
                nk50_maxsim, nk50_popw, norm_k50_diversity,
            ], dtype=np.float64)

            feats.append(np.concatenate([base, knn]))
            valid_pids.append(pid)

        if not feats:
            return []

        X = np.vstack(feats)
        scores = self.model.predict(X)

        top_indices = np.argsort(-scores)[:top_n]

        if not explain:
            return [(valid_pids[i], float(scores[i])) for i in top_indices]

        # Build user vector in note space for explanations
        u_exp = np.zeros(len(self.note_to_idx), dtype=np.float64)
        for note, val in user_vec.items():
            idx = self.note_to_idx.get(note.strip().lower())
            if idx is not None:
                u_exp[idx] = val
        u_norm_exp = u_exp / max(np.linalg.norm(u_exp), 1e-9)

        results = []
        for i in top_indices:
            pid = valid_pids[i]
            sku_vec = self.perfume_vectors.get(pid)
            if sku_vec is None or np.linalg.norm(sku_vec) == 0:
                results.append((pid, float(scores[i]), []))
                continue
            sku_n = sku_vec / max(np.linalg.norm(sku_vec), 1e-9)
            contrib = u_norm_exp * sku_n
            best = np.argsort(-contrib)[:5]
            explanation = [
                {"note": self.idx_to_note[j], "contribution": round(float(contrib[j]), 4)}
                for j in best if contrib[j] > 0
            ]
            results.append((pid, float(scores[i]), explanation))
        return results
