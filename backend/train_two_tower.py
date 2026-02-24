import sys
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader as TorchLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, str(Path(__file__).parent))

from app.core.config import get_settings
from app.ranking.data import DataLoader
from app.ranking.profile.build_profile import session_to_user_vector
from app.ranking.scoring.score import build_sku_vectors
from app.ranking.nn.two_tower import TwoTowerModel


class PerfumeDataset(Dataset):
    def __init__(self, user_vecs, sku_vecs, augment=False):
        self.users = user_vecs
        self.skus = sku_vecs
        self.augment = augment

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        u = self.users[idx].copy()
        s = self.skus[idx].copy()
        if self.augment:
            u = augment_user(u)
        return torch.from_numpy(u), torch.from_numpy(s)


def augment_user(vec):
    nonzero = vec != 0
    r = random.random()
    if r < 0.4:
        noise = np.random.normal(0, 0.15, size=vec.shape).astype(np.float32)
        vec = np.maximum(vec + noise * nonzero, 0)
    elif r < 0.7:
        active = np.where(nonzero)[0]
        if len(active) > 1:
            vec[np.random.choice(active)] = 0
    else:
        vec = vec * np.random.uniform(0.7, 1.3)
    return vec


def build_pairs(loader, note_to_idx, perfume_vectors, seed=42):
    sessions = loader.load_organ_sessions()
    shuffled = sessions.sample(frac=1, random_state=seed)
    n_test = max(1, int(len(shuffled) * 0.2))
    train_sessions = shuffled.iloc[:-n_test]

    dim = len(note_to_idx)
    catalog = set(perfume_vectors.keys())
    users, skus = [], []

    for _, row in train_sessions.iterrows():
        sid = int(row["session_id"])
        target = int(row["target_perfume_id"])
        if target not in catalog:
            continue
        uv = session_to_user_vector(sid, loader, use_presses=True, alpha_recipe=0.7)
        if not uv:
            continue

        u = np.zeros(dim, dtype=np.float32)
        for note, val in uv.items():
            i = note_to_idx.get(note.strip().lower())
            if i is not None:
                u[i] = val
        if np.all(u == 0):
            continue

        users.append(u)
        skus.append(perfume_vectors[target].astype(np.float32))

    return users, skus


def train():
    settings = get_settings()
    loader = DataLoader(
        perfume_dir=settings.data_perfume_dir,
        organ_dir=settings.data_organ_dir if settings.data_organ_dir.exists() else None,
    )

    perfume_notes = loader.load_perfume_notes()
    perfume_vectors, note_to_idx, _ = build_sku_vectors(perfume_notes)

    users, skus = build_pairs(loader, note_to_idx, perfume_vectors)
    print(f"Всего пар: {len(users)}")

    n_val = max(1, int(len(users) * 0.2))
    idx = list(range(len(users)))
    random.seed(123)
    random.shuffle(idx)

    val_i, train_i = idx[:n_val], idx[n_val:]
    train_ds = PerfumeDataset([users[i] for i in train_i], [skus[i] for i in train_i], augment=True)
    val_ds = PerfumeDataset([users[i] for i in val_i], [skus[i] for i in val_i])

    train_dl = TorchLoader(train_ds, batch_size=64, shuffle=True, drop_last=True)
    val_dl = TorchLoader(val_ds, batch_size=64, shuffle=False)

    model = TwoTowerModel(len(note_to_idx))
    opt = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    sched = CosineAnnealingLR(opt, T_max=100)

    best_loss = float("inf")
    patience, no_improve = 15, 0
    save_path = Path(__file__).parent / "models" / "two_tower_best.pt"
    save_path.parent.mkdir(exist_ok=True)

    for epoch in range(100):
        model.train()
        total, n = 0.0, 0
        for ub, sb in train_dl:
            u_emb, s_emb = model(ub, sb)
            loss = model.info_nce_loss(u_emb, s_emb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            n += 1
        sched.step()
        t_loss = total / max(n, 1)

        model.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for ub, sb in val_dl:
                u_emb, s_emb = model(ub, sb)
                total += model.info_nce_loss(u_emb, s_emb).item()
                n += 1
        v_loss = total / max(n, 1)

        print(f"Эпоха {epoch + 1:3d}  train={t_loss:.4f}  val={v_loss:.4f}")

        if v_loss < best_loss:
            best_loss = v_loss
            no_improve = 0
            torch.save({"model_state_dict": model.state_dict(), "input_dim": len(note_to_idx)}, save_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Ранняя остановка на эпохе {epoch + 1}")
                break

    print(f"Лучший val_loss: {best_loss:.4f}")
    print(f"Модель сохранена: {save_path}")


if __name__ == "__main__":
    train()
