"""Hybrid GBM Scorer: GBM with Two-Tower neural embedding features.

Uses 22 features: 20 standard GBM features + 2 neural features (nn_cosine, nn_l2_dist).
The Two-Tower model is trained on synthetic data only (no leakage).
"""

import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .ranker import extract_features, FEATURE_NAMES


class _EnhancedTwoTower(nn.Module):
    """Standalone re-definition to avoid importing train_hybrid_gbm at eval time."""

    def __init__(self, input_dim, hidden=256, mid=256, out=128, dropout=0.3):
        super().__init__()
        self.user_tower = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, mid),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mid, out),
        )
        self.sku_tower = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, mid),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mid, out),
        )

    def forward(self, user_x, sku_x):
        u = F.normalize(self.user_tower(user_x), dim=-1)
        s = F.normalize(self.sku_tower(sku_x), dim=-1)
        return u, s


class HybridGBMScorer:
    def __init__(self, gbm_path, tt_path, note_to_idx, idx_to_note, perfume_vectors,
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
        with open(gbm_path, "rb") as f:
            self.gbm_model = pickle.load(f)

        # Load Two-Tower model
        ckpt = torch.load(tt_path, map_location="cpu", weights_only=False)
        input_dim = len(note_to_idx)
        self.nn_model = _EnhancedTwoTower(input_dim, hidden=256, mid=256, out=128)
        self.nn_model.load_state_dict(ckpt["model_state_dict"])
        self.nn_model.eval()

        # Pre-compute SKU embeddings
        self.pids = sorted(perfume_vectors.keys())
        self.max_pop = max(self.popularity.values()) if self.popularity else 1.0

        N = len(self.pids)
        sku_matrix = np.zeros((N, input_dim), dtype=np.float32)
        for i, pid in enumerate(self.pids):
            sku_matrix[i] = perfume_vectors[pid].astype(np.float32)

        with torch.no_grad():
            self.sku_embs = self.nn_model.sku_tower(torch.from_numpy(sku_matrix)).numpy()
            norms = np.linalg.norm(self.sku_embs, axis=1, keepdims=True)
            self.sku_embs = np.where(norms > 0, self.sku_embs / norms, self.sku_embs)

        self.pid_to_idx = {pid: i for i, pid in enumerate(self.pids)}

    def score(self, user_vec, top_n=10, candidate_pids=None):
        if not user_vec:
            return []

        pids = candidate_pids if candidate_pids is not None else self.pids

        # Compute user embedding once
        dim = len(self.note_to_idx)
        u_np = np.zeros(dim, dtype=np.float32)
        for note, val in user_vec.items():
            idx = self.note_to_idx.get(note.strip().lower())
            if idx is not None:
                u_np[idx] = val

        with torch.no_grad():
            u_emb = self.nn_model.user_tower(torch.from_numpy(u_np).unsqueeze(0))
            u_emb = F.normalize(u_emb, dim=-1).squeeze(0).numpy()

        feats = []
        valid_pids = []
        for pid in pids:
            if pid not in self.perfume_vectors:
                continue

            # 20 standard GBM features
            pop = self.popularity.get(pid, 0.0) / self.max_pop
            nv = self.norm_perfume_vectors.get(pid) if self.norm_perfume_vectors else None
            base = extract_features(user_vec, self.perfume_vectors[pid],
                                    self.note_to_idx, self.idx_to_note, pop,
                                    norm_sku_vec=nv, norm_note_to_idx=self.norm_note_to_idx,
                                    sku_meta=self.sku_meta.get(pid))

            # 2 neural features
            idx = self.pid_to_idx.get(pid)
            if idx is not None:
                s_emb = self.sku_embs[idx]
            else:
                s_emb = np.zeros(128, dtype=np.float32)

            nn_cos = float(np.dot(u_emb, s_emb))
            nn_l2 = float(np.linalg.norm(u_emb - s_emb))

            feats.append(np.append(base, [nn_cos, nn_l2]))
            valid_pids.append(pid)

        if not feats:
            return []

        X = np.vstack(feats)
        scores = self.gbm_model.predict(X)

        top_indices = np.argsort(-scores)[:top_n]
        return [(valid_pids[i], float(scores[i])) for i in top_indices]
