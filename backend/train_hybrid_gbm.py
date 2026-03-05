"""Hybrid GBM: Two-Tower embeddings as extra GBM features.

Strategy:
  1. Train Enhanced Two-Tower on SYNTHETIC data only (no real data leakage)
  2. Freeze Two-Tower, extract embeddings for each (user, sku) pair
  3. Add 2 neural features (nn_cosine, nn_l2_dist) to existing 20 GBM features
  4. Train GBM LambdaRank on real data with 22 features

This avoids the data scarcity problem:
  - Two-Tower learns general note→embedding patterns from 40K+ synthetic pairs
  - GBM uses the embeddings as features on 483 real pairs
  - No leakage because Two-Tower never sees real pairs
"""

import sys
import random
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).parent))

from app.core.config import get_settings
from app.ranking.data import DataLoader
from app.ranking.profile.build_profile import session_to_user_vector
from app.ranking.normalize import build_synonym_map, ORGAN_NOTES
from app.ranking.scoring.score import build_sku_vectors
from app.ranking.synthetic import generate_noisy_sessions, generate_confused_sessions
from app.ranking.nn.two_tower import TwoTowerModel
from app.ranking.gbm.ranker import extract_features, FEATURE_NAMES
from train_gbm import build_sku_meta


class EnhancedTwoTower(TwoTowerModel):
    """Deeper Two-Tower: 3 layers, 128-dim embeddings."""

    def __init__(self, input_dim, hidden=256, mid=256, out=128, dropout=0.3, temperature=0.07):
        # Skip TwoTowerModel.__init__, call nn.Module directly
        import torch.nn as nn
        import torch.nn.functional as F
        nn.Module.__init__(self)

        self.temperature = temperature
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
        import torch.nn.functional as F
        u = F.normalize(self.user_tower(user_x), dim=-1)
        s = F.normalize(self.sku_tower(sku_x), dim=-1)
        return u, s

    def info_nce_loss(self, user_emb, sku_emb):
        import torch.nn.functional as F
        logits = user_emb @ sku_emb.T / self.temperature
        labels = torch.arange(logits.size(0), device=logits.device)
        return F.cross_entropy(logits, labels)


def train_two_tower_on_synthetic(perfume_vectors, note_to_idx, aroma_map, save_path, epochs=80):
    """Train Enhanced Two-Tower on synthetic data only."""
    print("=== Training Enhanced Two-Tower on synthetic data ===")

    synth_noisy = generate_noisy_sessions(perfume_vectors, note_to_idx, aroma_map,
                                          n_per_sku=10, seed=42)
    synth_confused = generate_confused_sessions(perfume_vectors, note_to_idx, aroma_map,
                                                n_per_group=5, seed=42)
    all_pairs = synth_noisy + synth_confused
    all_pairs = [(uv, t) for uv, t in all_pairs if t in perfume_vectors and uv]
    random.seed(42)
    random.shuffle(all_pairs)
    print(f"Synthetic pairs: {len(all_pairs)}")

    dim = len(note_to_idx)
    model = EnhancedTwoTower(dim, hidden=256, mid=256, out=128, dropout=0.3)
    print(f"Two-Tower params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    batch_size = 128

    for epoch in range(epochs):
        model.train()
        random.shuffle(all_pairs)
        total_loss = 0.0
        n_batches = 0

        for start in range(0, len(all_pairs), batch_size):
            batch = all_pairs[start:start + batch_size]
            if len(batch) < 4:
                continue

            # Build user and sku tensors
            user_np = np.zeros((len(batch), dim), dtype=np.float32)
            sku_np = np.zeros((len(batch), dim), dtype=np.float32)

            for i, (uv, pid) in enumerate(batch):
                for note, val in uv.items():
                    idx = note_to_idx.get(note.strip().lower())
                    if idx is not None:
                        user_np[i, idx] = val
                sku_np[i] = perfume_vectors[pid].astype(np.float32)

            user_t = torch.from_numpy(user_np)
            sku_t = torch.from_numpy(sku_np)

            user_emb, sku_emb = model(user_t, sku_t)
            loss = model.info_nce_loss(user_emb, sku_emb)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1:3d}: loss={avg_loss:.4f}")

    torch.save({
        "model_state_dict": model.state_dict(),
        "input_dim": dim,
        "hidden": 256,
        "out": 128,
    }, save_path)
    print(f"Two-Tower saved: {save_path}")
    return model


def extract_nn_features(model, user_vec, sku_vec, note_to_idx):
    """Extract 2 neural features: nn_cosine, nn_l2_dist."""
    dim = len(note_to_idx)

    u = np.zeros(dim, dtype=np.float32)
    for note, val in user_vec.items():
        idx = note_to_idx.get(note.strip().lower())
        if idx is not None:
            u[idx] = val

    u_t = torch.from_numpy(u).unsqueeze(0)
    s_t = torch.from_numpy(sku_vec.astype(np.float32)).unsqueeze(0)

    with torch.no_grad():
        u_emb, s_emb = model(u_t, s_t)
        u_emb = u_emb.squeeze(0).numpy()
        s_emb = s_emb.squeeze(0).numpy()

    nn_cosine = float(np.dot(u_emb, s_emb))  # Already L2-normalized
    nn_l2 = float(np.linalg.norm(u_emb - s_emb))

    return nn_cosine, nn_l2


def build_hybrid_dataset(pairs, perfume_vectors, note_to_idx, idx_to_note, popularity,
                         norm_pv, norm_nti, sku_meta, nn_model, neg_random=16):
    """Build dataset with 20 GBM features + 2 neural features = 22 features."""
    catalog = sorted(perfume_vectors.keys())
    max_pop = max(popularity.values()) if popularity else 1.0
    rng = random.Random(42)

    # Pre-compute all SKU embeddings
    dim = len(note_to_idx)
    N = len(catalog)
    sku_matrix = np.zeros((N, dim), dtype=np.float32)
    for i, pid in enumerate(catalog):
        sku_matrix[i] = perfume_vectors[pid].astype(np.float32)

    with torch.no_grad():
        sku_embs = nn_model.sku_tower(torch.from_numpy(sku_matrix)).numpy()  # (N, 128)
    pid_to_sku_emb = {pid: sku_embs[i] for i, pid in enumerate(catalog)}

    def _feat(uv, pid):
        # Standard 20 GBM features
        nv = norm_pv.get(pid)
        base = extract_features(uv, perfume_vectors[pid], note_to_idx, idx_to_note,
                                popularity.get(pid, 0) / max_pop,
                                norm_sku_vec=nv, norm_note_to_idx=norm_nti,
                                sku_meta=sku_meta.get(pid) if sku_meta else None)

        # Neural features
        u_np = np.zeros(dim, dtype=np.float32)
        for note, val in uv.items():
            idx = note_to_idx.get(note.strip().lower())
            if idx is not None:
                u_np[idx] = val

        with torch.no_grad():
            u_emb = nn_model.user_tower(torch.from_numpy(u_np).unsqueeze(0))
            u_emb = torch.nn.functional.normalize(u_emb, dim=-1).squeeze(0).numpy()

        s_emb = pid_to_sku_emb[pid]
        nn_cos = float(np.dot(u_emb, s_emb))
        nn_l2 = float(np.linalg.norm(u_emb - s_emb))

        return np.append(base, [nn_cos, nn_l2])

    X, y, groups = [], [], []

    for uv, target in pairs:
        if target not in perfume_vectors or not uv:
            continue

        X.append(_feat(uv, target))
        y.append(1)

        neg_pool = [p for p in catalog if p != target]
        n_rand = min(neg_random, len(neg_pool))
        rand_chosen = rng.sample(neg_pool, n_rand)

        for neg_pid in rand_chosen:
            X.append(_feat(uv, neg_pid))
            y.append(0)

        groups.append(1 + n_rand)

    return np.vstack(X), np.array(y, dtype=np.int32), groups


HYBRID_FEATURE_NAMES = FEATURE_NAMES + ["nn_cosine", "nn_l2_dist"]


def main():
    settings = get_settings()
    loader = DataLoader(
        perfume_dir=settings.data_perfume_dir,
        organ_dir=settings.data_organ_dir if settings.data_organ_dir.exists() else None,
    )

    perfume_notes = loader.load_perfume_notes()
    perfumes = loader.load_perfumes()
    perfume_vectors, note_to_idx, idx_to_note = build_sku_vectors(perfume_notes)

    catalog_notes = perfume_notes["note"].astype(str).str.strip().str.lower().unique().tolist()
    syn_map = build_synonym_map(catalog_notes)
    norm_pv, norm_nti, _ = build_sku_vectors(perfume_notes, synonym_map=syn_map)

    popularity = {}
    if "allVotes" in perfumes.columns:
        popularity = perfumes.set_index("perfume_id")["allVotes"].to_dict()

    sku_meta = build_sku_meta(perfume_vectors, note_to_idx, perfume_notes, perfumes, popularity)
    aroma_map = loader.load_organ_aroma_notes_map()

    save_dir = Path(__file__).parent / "models"
    save_dir.mkdir(exist_ok=True)

    # ================================================================
    # Step 1: Train Two-Tower on synthetic data
    # ================================================================
    tt_path = save_dir / "synthetic_two_tower.pt"
    if tt_path.exists():
        print(f"Loading existing Two-Tower from {tt_path}")
        ckpt = torch.load(tt_path, map_location="cpu", weights_only=False)
        nn_model = EnhancedTwoTower(len(note_to_idx), hidden=256, mid=256, out=128)
        nn_model.load_state_dict(ckpt["model_state_dict"])
        nn_model.eval()
    else:
        nn_model = train_two_tower_on_synthetic(perfume_vectors, note_to_idx, aroma_map, tt_path)
        nn_model.eval()

    # ================================================================
    # Step 2: Build hybrid dataset with neural features
    # ================================================================
    sessions = loader.load_organ_sessions()
    shuffled = sessions.sample(frac=1, random_state=42)
    n_test = max(1, int(len(shuffled) * 0.2))
    train_sessions = shuffled.iloc[:-n_test]

    real_pairs = []
    for _, row in train_sessions.iterrows():
        sid = int(row["session_id"])
        target = int(row["target_perfume_id"])
        uv = session_to_user_vector(sid, loader, use_presses=True, alpha_recipe=0.7)
        if uv and target in perfume_vectors:
            real_pairs.append((uv, target))

    random.seed(42)
    random.shuffle(real_pairs)
    print(f"\nReal pairs: {len(real_pairs)}")

    print("Building hybrid dataset (20 GBM + 2 neural features)...")
    X, y, groups = build_hybrid_dataset(
        real_pairs, perfume_vectors, note_to_idx, idx_to_note, popularity,
        norm_pv, norm_nti, sku_meta, nn_model, neg_random=16,
    )
    print(f"Dataset: {len(y)} samples, {len(groups)} groups, {X.shape[1]} features")

    # ================================================================
    # Step 3: Train GBM with 22 features
    # ================================================================
    n_val_groups = max(1, int(len(groups) * 0.2))
    train_groups = groups[:-n_val_groups]
    val_groups = groups[-n_val_groups:]
    train_end = sum(train_groups)

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:], y[train_end:]

    train_ds = lgb.Dataset(X_train, label=y_train, group=train_groups, feature_name=HYBRID_FEATURE_NAMES)
    val_ds = lgb.Dataset(X_val, label=y_val, group=val_groups, feature_name=HYBRID_FEATURE_NAMES, reference=train_ds)

    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "eval_at": [5, 10],
        "num_leaves": 15,
        "learning_rate": 0.05,
        "min_child_samples": 30,
        "subsample": 0.6,
        "colsample_bytree": 0.7,
        "reg_alpha": 1.0,
        "reg_lambda": 5.0,
        "verbose": -1,
    }

    print("\n=== Training Hybrid GBM (22 features) ===")
    model = lgb.train(
        params, train_ds,
        num_boost_round=500,
        valid_sets=[val_ds],
        valid_names=["val"],
        callbacks=[
            lgb.early_stopping(50),
            lgb.log_evaluation(50),
        ],
    )

    # Save
    gbm_path = save_dir / "hybrid_gbm_ranker.pkl"
    with open(gbm_path, "wb") as f:
        pickle.dump(model, f)

    print(f"\nFeature importance:")
    imp = model.feature_importance(importance_type="gain")
    for name, val in sorted(zip(HYBRID_FEATURE_NAMES, imp), key=lambda x: -x[1]):
        print(f"  {name:20s} {val:.1f}")

    print(f"\nHybrid GBM saved: {gbm_path}")
    print(f"Two-Tower saved: {tt_path}")
    print(f"Iterations: {model.current_iteration()}")

    # ================================================================
    # Quick Hit@10 evaluation
    # ================================================================
    test_sessions = shuffled.iloc[-n_test:]
    catalog = sorted(perfume_vectors.keys())
    max_pop = max(popularity.values()) if popularity else 1.0
    dim = len(note_to_idx)

    # Pre-compute SKU embeddings
    N = len(catalog)
    sku_matrix_np = np.zeros((N, dim), dtype=np.float32)
    for i, pid in enumerate(catalog):
        sku_matrix_np[i] = perfume_vectors[pid].astype(np.float32)

    with torch.no_grad():
        sku_embs = nn_model.sku_tower(torch.from_numpy(sku_matrix_np)).numpy()
    pid_to_sku_emb = {pid: sku_embs[i] for i, pid in enumerate(catalog)}

    hits5, hits10 = 0, 0
    n_eval = 0

    for _, row in test_sessions.iterrows():
        sid = int(row["session_id"])
        target = int(row["target_perfume_id"])
        uv = session_to_user_vector(sid, loader, use_presses=True, alpha_recipe=0.7)
        if not uv or target not in perfume_vectors:
            continue

        # User embedding
        u_np = np.zeros(dim, dtype=np.float32)
        for note, val in uv.items():
            idx = note_to_idx.get(note.strip().lower())
            if idx is not None:
                u_np[idx] = val

        with torch.no_grad():
            u_emb = nn_model.user_tower(torch.from_numpy(u_np).unsqueeze(0))
            u_emb = torch.nn.functional.normalize(u_emb, dim=-1).squeeze(0).numpy()

        feats = []
        for pid in catalog:
            nv = norm_pv.get(pid)
            base = extract_features(uv, perfume_vectors[pid], note_to_idx, idx_to_note,
                                    popularity.get(pid, 0) / max_pop,
                                    norm_sku_vec=nv, norm_note_to_idx=norm_nti,
                                    sku_meta=sku_meta.get(pid))
            s_emb = pid_to_sku_emb[pid]
            nn_cos = float(np.dot(u_emb, s_emb))
            nn_l2 = float(np.linalg.norm(u_emb - s_emb))
            feats.append(np.append(base, [nn_cos, nn_l2]))

        Xp = np.vstack(feats)
        scores = model.predict(Xp)
        top_idx = np.argsort(-scores)[:10]
        top_pids = [catalog[i] for i in top_idx]

        if target in top_pids[:5]:
            hits5 += 1
        if target in top_pids[:10]:
            hits10 += 1
        n_eval += 1

    print(f"\nTest Hit@5 = {hits5/max(n_eval,1):.1%}, Hit@10 = {hits10/max(n_eval,1):.1%} (n={n_eval})")


if __name__ == "__main__":
    main()
