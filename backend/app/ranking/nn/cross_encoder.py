"""Cross-Encoder v2 for perfume ranking.

Key changes from v1 (which scored 2.1% Hit@10):
  - Wider projection: 827→64 (was 12) with SEPARATE user/sku projectors
  - LayerNorm instead of BatchNorm (stable on small groups)
  - Pre-computed scalar features as shortcut: cosine, dot, overlap
  - Deeper MLP with residual connection
  - Pure BCE loss (ListNet removed)

Architecture:
    user_proj: Linear(note_dim, 64)
    sku_proj:  Linear(note_dim, 64)
    Interaction: [u64, s64, u*s, |u-s|, scalars(3), meta(8)] = 267-dim
    MLP: 267 → 256 → ReLU → Drop → 128 → ReLU → Drop → 1
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEncoderModel(nn.Module):
    PROJ_DIM = 64
    META_DIM = 8
    SCALAR_DIM = 3  # cosine, dot_norm, overlap_ratio

    def __init__(self, note_dim: int, proj_dim: int = 64,
                 hidden1: int = 256, hidden2: int = 128,
                 dropout: float = 0.4):
        super().__init__()
        self.note_dim = note_dim
        self.proj_dim = proj_dim

        # Separate projections for user and SKU (different distributions)
        self.user_proj = nn.Sequential(
            nn.Linear(note_dim, proj_dim),
            nn.ReLU(),
        )
        self.sku_proj = nn.Sequential(
            nn.Linear(note_dim, proj_dim),
            nn.ReLU(),
        )

        # Input: u_proj(64) + s_proj(64) + hadamard(64) + diff(64) + scalars(3) + meta(8) = 267
        mlp_input = proj_dim * 4 + self.SCALAR_DIM + self.META_DIM
        self.mlp = nn.Sequential(
            nn.LayerNorm(mlp_input),
            nn.Linear(mlp_input, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, 1),
        )

    def forward(self, user_vec: torch.Tensor, sku_vec: torch.Tensor,
                meta: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
        """
        Args:
            user_vec: (B, note_dim)
            sku_vec:  (B, note_dim)
            meta:     (B, 8) non-similarity SKU features
            scalars:  (B, 3) pre-computed [cosine, dot_norm, overlap_ratio]
        Returns:
            logits:   (B,) relevance score
        """
        u = self.user_proj(user_vec)        # (B, 64)
        s = self.sku_proj(sku_vec)           # (B, 64)

        hadamard = u * s                     # element-wise interaction
        diff = torch.abs(u - s)              # distance signal

        x = torch.cat([u, s, hadamard, diff, scalars, meta], dim=-1)
        return self.mlp(x).squeeze(-1)


def compute_scalars(user_vec_np: np.ndarray, sku_vec_np: np.ndarray) -> np.ndarray:
    """Pre-compute scalar interaction features: cosine, dot_norm, overlap_ratio."""
    u_norm = np.linalg.norm(user_vec_np)
    s_norm = np.linalg.norm(sku_vec_np)

    if u_norm > 0 and s_norm > 0:
        cosine = np.dot(user_vec_np, sku_vec_np) / (u_norm * s_norm)
    else:
        cosine = 0.0

    dot = np.dot(user_vec_np, sku_vec_np)
    # Normalize dot product by geometric mean of norms
    dot_norm = dot / max(u_norm * s_norm, 1e-9)

    u_nz = set(np.where(user_vec_np > 0)[0])
    s_nz = set(np.where(sku_vec_np > 0)[0])
    overlap_ratio = len(u_nz & s_nz) / max(len(u_nz), 1)

    return np.array([cosine, dot_norm, overlap_ratio], dtype=np.float32)


# --- Meta feature keys (must match train_gbm.build_sku_meta output) ---
META_KEYS = [
    "collision_group_size", "pop_rank_in_group",
    "note_count", "note_entropy", "top3_concentration",
    "love_ratio", "longevity_idx", "sillage_idx",
]


def _meta_to_tensor(sku_meta: dict | None) -> np.ndarray:
    """Convert sku_meta dict to fixed-size array."""
    m = sku_meta or {}
    return np.array([m.get(k, 0.0) for k in META_KEYS], dtype=np.float32)


class CrossEncoderScorer:
    """Inference-time scorer matching TwoTowerScorer / GBMScorer interface."""

    def __init__(self, model_path, note_to_idx, perfume_vectors, sku_meta=None):
        self.note_to_idx = note_to_idx
        self.input_dim = len(note_to_idx)
        self.sku_meta = sku_meta or {}

        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        note_dim = ckpt.get("note_dim", self.input_dim)
        self.model = CrossEncoderModel(note_dim)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        # Pre-compute SKU tensors
        pids = sorted(perfume_vectors.keys())
        S = np.zeros((len(pids), self.input_dim), dtype=np.float32)
        M = np.zeros((len(pids), CrossEncoderModel.META_DIM), dtype=np.float32)
        for i, pid in enumerate(pids):
            S[i] = perfume_vectors[pid].astype(np.float32)
            M[i] = _meta_to_tensor(self.sku_meta.get(pid))

        self.sku_matrix = torch.from_numpy(S)    # (N, note_dim)
        self.meta_matrix = torch.from_numpy(M)   # (N, 8)
        self.perfume_ids = pids

    def score(self, user_vec: dict, top_n: int = 10):
        """Score all SKUs against user_vec. Returns list of (pid, score)."""
        u = np.zeros(self.input_dim, dtype=np.float32)
        for note, val in user_vec.items():
            idx = self.note_to_idx.get(note.strip().lower())
            if idx is not None:
                u[idx] = val
        if np.all(u == 0):
            return []

        N = len(self.perfume_ids)
        user_batch = torch.from_numpy(u).unsqueeze(0).expand(N, -1)  # (N, note_dim)

        # Pre-compute scalars for each (user, sku) pair
        scalars = np.zeros((N, CrossEncoderModel.SCALAR_DIM), dtype=np.float32)
        sku_np = self.sku_matrix.numpy()
        for i in range(N):
            scalars[i] = compute_scalars(u, sku_np[i])
        scalars_t = torch.from_numpy(scalars)

        with torch.no_grad():
            scores = self.model(user_batch, self.sku_matrix, self.meta_matrix, scalars_t)
            scores = scores.numpy()

        top_indices = np.argsort(-scores)[:top_n]
        return [(self.perfume_ids[i], float(scores[i])) for i in top_indices]
