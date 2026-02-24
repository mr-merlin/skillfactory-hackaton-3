import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Tower(nn.Module):
    def __init__(self, input_dim, hidden=128, out=64, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


class TwoTowerModel(nn.Module):
    def __init__(self, input_dim, hidden=128, out=64, dropout=0.3, temperature=0.07):
        super().__init__()
        self.user_tower = Tower(input_dim, hidden, out, dropout)
        self.sku_tower = Tower(input_dim, hidden, out, dropout)
        self.temperature = temperature

    def forward(self, user_x, sku_x):
        return self.user_tower(user_x), self.sku_tower(sku_x)

    def info_nce_loss(self, user_emb, sku_emb):
        logits = user_emb @ sku_emb.T / self.temperature
        labels = torch.arange(logits.size(0), device=logits.device)
        return F.cross_entropy(logits, labels)


class TwoTowerScorer:
    def __init__(self, model_path, note_to_idx, perfume_vectors):
        self.note_to_idx = note_to_idx
        self.input_dim = len(note_to_idx)

        self.model = TwoTowerModel(self.input_dim)
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        pids = sorted(perfume_vectors.keys())
        M = np.zeros((len(pids), self.input_dim), dtype=np.float32)
        for i, pid in enumerate(pids):
            M[i] = perfume_vectors[pid].astype(np.float32)

        with torch.no_grad():
            self.sku_embeddings = self.model.sku_tower(torch.from_numpy(M))
        self.perfume_ids = pids

    def score(self, user_vec, top_n=10):
        u = np.zeros(self.input_dim, dtype=np.float32)
        for note, val in user_vec.items():
            idx = self.note_to_idx.get(note.strip().lower())
            if idx is not None:
                u[idx] = val
        if np.all(u == 0):
            return []

        with torch.no_grad():
            u_emb = self.model.user_tower(torch.from_numpy(u).unsqueeze(0))
            scores = (u_emb @ self.sku_embeddings.T).squeeze(0).numpy()

        top_indices = np.argsort(-scores)[:top_n]
        return [(self.perfume_ids[i], float(scores[i])) for i in top_indices]
