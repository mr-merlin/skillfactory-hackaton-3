"""Train Cross-Encoder v2 for perfume ranking.

Key improvements over v1 (2.1% Hit@10):
  - Wider projection (64 vs 12), separate user/sku projectors
  - LayerNorm (stable on small groups) instead of BatchNorm
  - Pre-computed scalar shortcuts (cosine, dot, overlap)
  - Multi-group gradient accumulation (4 groups per step)
  - Pure BCE loss (ListNet removed — unstable on small groups)
  - Gaussian noise augmentation on user vectors
  - Phase 3: Joint training as alternative to pretrain→finetune

Three phases:
  Phase 1: Pre-train on synthetic data, 100 epochs, lr=5e-4
  Phase 2: Fine-tune on real data, 50 epochs, lr=5e-5, early stop
  Phase 3: Joint training (real×5 + synthetic), 60 epochs, lr=3e-4
  → Pick whichever gives better val Hit@10
"""

import sys
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).parent))

from app.core.config import get_settings
from app.ranking.data import DataLoader
from app.ranking.profile.build_profile import session_to_user_vector
from app.ranking.normalize import ORGAN_NOTES
from app.ranking.scoring.score import build_sku_vectors
from app.ranking.synthetic import (
    generate_noisy_sessions,
    generate_confused_sessions,
)
from app.ranking.nn.cross_encoder import (
    CrossEncoderModel, _meta_to_tensor, compute_scalars,
)
from train_gbm import build_sku_meta


def build_collision_index(perfume_vectors, note_to_idx):
    """Build collision groups + hamming-1 neighbors for hard negative mining."""
    organ_idxs = [note_to_idx[n] for n in ORGAN_NOTES if n in note_to_idx]
    pid_to_fp = {}
    fp_to_pids = {}
    for pid, vec in perfume_vectors.items():
        fp = tuple(1 if vec[i] > 0 else 0 for i in organ_idxs)
        pid_to_fp[pid] = fp
        fp_to_pids.setdefault(fp, []).append(pid)

    all_fps = set(fp_to_pids.keys())
    fp_hamming1 = {}
    for fp in all_fps:
        neighbors = []
        for i in range(len(fp)):
            flipped = list(fp)
            flipped[i] = 1 - flipped[i]
            flipped = tuple(flipped)
            if flipped in fp_to_pids:
                neighbors.extend(fp_to_pids[flipped])
        fp_hamming1[fp] = neighbors

    return pid_to_fp, fp_to_pids, fp_hamming1


def mine_negatives(target_pid, catalog, pid_to_fp, fp_to_pids, fp_hamming1,
                   n_hard=4, n_semi=4, n_random=8, rng=None):
    """Mine 3 tiers of negatives."""
    if rng is None:
        rng = random.Random()
    fp = pid_to_fp.get(target_pid)
    negatives = set()

    if fp and fp in fp_to_pids:
        hard_pool = [p for p in fp_to_pids[fp] if p != target_pid]
        if hard_pool:
            negatives.update(rng.sample(hard_pool, min(n_hard, len(hard_pool))))

    if fp and fp in fp_hamming1:
        semi_pool = [p for p in fp_hamming1[fp] if p != target_pid and p not in negatives]
        if semi_pool:
            negatives.update(rng.sample(semi_pool, min(n_semi, len(semi_pool))))

    rand_pool = [p for p in catalog if p != target_pid and p not in negatives]
    if rand_pool:
        negatives.update(rng.sample(rand_pool, min(n_random, len(rand_pool))))

    return list(negatives)


def prepare_group(user_vec, target_pid, neg_pids, perfume_vectors, note_to_idx, sku_meta,
                  noise_std=0.0, rng_np=None):
    """Build tensors for one group: 1 positive + N negatives."""
    dim = len(note_to_idx)

    u = np.zeros(dim, dtype=np.float32)
    for note, val in user_vec.items():
        i = note_to_idx.get(note.strip().lower())
        if i is not None:
            u[i] = val

    # Noise augmentation
    if noise_std > 0 and rng_np is not None:
        noise = rng_np.normal(0, noise_std, size=dim).astype(np.float32)
        u = np.maximum(u + noise * np.abs(u), 0)

    all_pids = [target_pid] + neg_pids
    n = len(all_pids)

    user_batch = np.tile(u, (n, 1))
    sku_batch = np.zeros((n, dim), dtype=np.float32)
    meta_batch = np.zeros((n, CrossEncoderModel.META_DIM), dtype=np.float32)
    scalar_batch = np.zeros((n, CrossEncoderModel.SCALAR_DIM), dtype=np.float32)
    labels = np.zeros(n, dtype=np.float32)
    labels[0] = 1.0

    for i, pid in enumerate(all_pids):
        sku_batch[i] = perfume_vectors[pid].astype(np.float32)
        meta_batch[i] = _meta_to_tensor(sku_meta.get(pid))
        scalar_batch[i] = compute_scalars(u, sku_batch[i])

    return (
        torch.from_numpy(user_batch),
        torch.from_numpy(sku_batch),
        torch.from_numpy(meta_batch),
        torch.from_numpy(scalar_batch),
        torch.from_numpy(labels),
    )


def evaluate_hit_at_k(model, pairs, perfume_vectors, note_to_idx, sku_meta, k=10):
    """Evaluate Hit@k by scoring ALL catalog items for each query."""
    model.eval()
    dim = len(note_to_idx)
    catalog = sorted(perfume_vectors.keys())
    N = len(catalog)

    sku_matrix = np.zeros((N, dim), dtype=np.float32)
    meta_matrix = np.zeros((N, CrossEncoderModel.META_DIM), dtype=np.float32)
    for i, pid in enumerate(catalog):
        sku_matrix[i] = perfume_vectors[pid].astype(np.float32)
        meta_matrix[i] = _meta_to_tensor(sku_meta.get(pid))

    sku_t = torch.from_numpy(sku_matrix)
    meta_t = torch.from_numpy(meta_matrix)
    pid_to_idx = {pid: i for i, pid in enumerate(catalog)}

    hits = {5: 0, 10: 0}
    total = 0

    with torch.no_grad():
        for user_vec, target_pid in pairs:
            if target_pid not in pid_to_idx:
                continue
            u = np.zeros(dim, dtype=np.float32)
            for note, val in user_vec.items():
                idx = note_to_idx.get(note.strip().lower())
                if idx is not None:
                    u[idx] = val
            if np.all(u == 0):
                continue

            scalars = np.zeros((N, CrossEncoderModel.SCALAR_DIM), dtype=np.float32)
            for i in range(N):
                scalars[i] = compute_scalars(u, sku_matrix[i])
            scalars_t = torch.from_numpy(scalars)

            user_batch = torch.from_numpy(u).unsqueeze(0).expand(N, -1)
            scores = model(user_batch, sku_t, meta_t, scalars_t).numpy()
            top_indices = np.argsort(-scores)[:10]
            top_pids = [catalog[i] for i in top_indices]

            for kk in hits:
                if target_pid in top_pids[:kk]:
                    hits[kk] += 1
            total += 1

    return {kk: hits[kk] / max(total, 1) for kk in hits}, total


def train():
    settings = get_settings()
    loader = DataLoader(
        perfume_dir=settings.data_perfume_dir,
        organ_dir=settings.data_organ_dir if settings.data_organ_dir.exists() else None,
    )

    perfume_notes = loader.load_perfume_notes()
    perfumes = loader.load_perfumes()
    perfume_vectors, note_to_idx, idx_to_note = build_sku_vectors(perfume_notes)

    popularity = {}
    if "allVotes" in perfumes.columns:
        popularity = perfumes.set_index("perfume_id")["allVotes"].to_dict()

    sku_meta = build_sku_meta(perfume_vectors, note_to_idx, perfume_notes, perfumes, popularity)
    pid_to_fp, fp_to_pids, fp_hamming1 = build_collision_index(perfume_vectors, note_to_idx)
    catalog = sorted(perfume_vectors.keys())
    print(f"Catalog: {len(perfume_vectors)} perfumes, {len(note_to_idx)} notes")

    # ================================================================
    # Data splits (same as GBM for comparability)
    # ================================================================
    sessions = loader.load_organ_sessions()
    shuffled = sessions.sample(frac=1, random_state=42)
    n_test = max(1, int(len(shuffled) * 0.2))
    train_sessions = shuffled.iloc[:-n_test]
    test_sessions = shuffled.iloc[-n_test:]

    real_pairs = []
    for _, row in train_sessions.iterrows():
        sid = int(row["session_id"])
        target = int(row["target_perfume_id"])
        uv = session_to_user_vector(sid, loader, use_presses=True, alpha_recipe=0.7)
        if uv and target in perfume_vectors:
            real_pairs.append((uv, target))

    random.seed(42)
    random.shuffle(real_pairs)
    n_val = max(1, int(len(real_pairs) * 0.2))
    real_train = real_pairs[:-n_val]
    real_val = real_pairs[-n_val:]

    test_pairs = []
    for _, row in test_sessions.iterrows():
        sid = int(row["session_id"])
        target = int(row["target_perfume_id"])
        uv = session_to_user_vector(sid, loader, use_presses=True, alpha_recipe=0.7)
        if uv and target in perfume_vectors:
            test_pairs.append((uv, target))

    print(f"Real train: {len(real_train)}, val: {len(real_val)}, test: {len(test_pairs)}")

    # ================================================================
    # Synthetic data
    # ================================================================
    aroma_map = loader.load_organ_aroma_notes_map()
    synth_noisy = generate_noisy_sessions(perfume_vectors, note_to_idx, aroma_map,
                                          n_per_sku=10, seed=42)
    synth_confused = generate_confused_sessions(perfume_vectors, note_to_idx, aroma_map,
                                                n_per_group=5, seed=42)
    synth_pairs = synth_noisy + synth_confused
    synth_pairs = [(uv, t) for uv, t in synth_pairs if t in perfume_vectors and uv]
    random.seed(42)
    random.shuffle(synth_pairs)
    print(f"Synthetic pairs: {len(synth_pairs)}")

    # ================================================================
    # Model init
    # ================================================================
    note_dim = len(note_to_idx)
    model = CrossEncoderModel(note_dim, proj_dim=64, hidden1=256, hidden2=128, dropout=0.4)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    save_dir = Path(__file__).parent / "models"
    save_dir.mkdir(exist_ok=True)
    best_path = save_dir / "cross_encoder_best.pt"

    rng = random.Random(42)
    rng_np = np.random.default_rng(42)

    ACCUM_STEPS = 4

    def run_epoch(mdl, pairs, optimizer, noise_std=0.0):
        mdl.train()
        total_loss = 0.0
        n_groups = 0
        indices = list(range(len(pairs)))
        rng.shuffle(indices)
        optimizer.zero_grad()

        for idx in indices:
            uv, target = pairs[idx]
            if target not in perfume_vectors:
                continue
            neg_pids = mine_negatives(target, catalog, pid_to_fp, fp_to_pids, fp_hamming1,
                                      n_hard=4, n_semi=4, n_random=8, rng=rng)
            if not neg_pids:
                continue

            u_t, s_t, m_t, sc_t, labels = prepare_group(
                uv, target, neg_pids, perfume_vectors, note_to_idx, sku_meta,
                noise_std=noise_std, rng_np=rng_np,
            )
            logits = mdl(u_t, s_t, m_t, sc_t)
            loss = F.binary_cross_entropy_with_logits(logits, labels) / ACCUM_STEPS
            loss.backward()
            total_loss += loss.item() * ACCUM_STEPS
            n_groups += 1

            if n_groups % ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(mdl.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

        if n_groups % ACCUM_STEPS != 0:
            torch.nn.utils.clip_grad_norm_(mdl.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        return total_loss / max(n_groups, 1)

    # ================================================================
    # Phase 1: Pre-train on synthetic data
    # ================================================================
    print("\n=== PHASE 1: Synthetic pre-training (100 epochs) ===")
    opt1 = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    sch1 = optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=100, eta_min=1e-5)

    for epoch in range(100):
        loss = run_epoch(model, synth_pairs, opt1, noise_std=0.1)
        sch1.step()
        if (epoch + 1) % 20 == 0:
            hits, n = evaluate_hit_at_k(model, real_val, perfume_vectors, note_to_idx, sku_meta)
            print(f"  Epoch {epoch+1:3d}: loss={loss:.4f}, val Hit@5={hits[5]:.1%}, Hit@10={hits[10]:.1%} (n={n})")

    torch.save({"model_state_dict": model.state_dict(), "note_dim": note_dim},
               save_dir / "cross_encoder_pretrain.pt")
    print("Pre-train checkpoint saved")

    # ================================================================
    # Phase 2: Fine-tune on real data
    # ================================================================
    print("\n=== PHASE 2: Fine-tune on real data (50 epochs) ===")
    opt2 = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-3)
    sch2 = optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=50, eta_min=1e-6)

    best_hit10_p2 = 0.0
    no_improve = 0

    for epoch in range(50):
        loss = run_epoch(model, real_train, opt2, noise_std=0.05)
        sch2.step()

        hits, n = evaluate_hit_at_k(model, real_val, perfume_vectors, note_to_idx, sku_meta)
        h5, h10 = hits[5], hits[10]

        improved = ""
        if h10 >= best_hit10_p2:
            best_hit10_p2 = h10
            no_improve = 0
            torch.save({"model_state_dict": model.state_dict(), "note_dim": note_dim}, best_path)
            improved = " *BEST*"
        else:
            no_improve += 1

        if (epoch + 1) % 5 == 0 or improved:
            print(f"  Epoch {epoch+1:3d}: loss={loss:.4f}, val Hit@5={h5:.1%}, Hit@10={h10:.1%}{improved} (n={n})")

        if no_improve >= 15:
            print(f"  Early stop at epoch {epoch+1}")
            break

    # ================================================================
    # Phase 3: Joint training from scratch (alternative approach)
    # ================================================================
    print("\n=== PHASE 3: Joint training (real×5 + synthetic), 60 epochs ===")
    model_j = CrossEncoderModel(note_dim, proj_dim=64, hidden1=256, hidden2=128, dropout=0.4)

    joint_pairs = real_train * 5 + synth_pairs
    random.shuffle(joint_pairs)
    print(f"Joint pairs: {len(joint_pairs)}")

    opt3 = optim.AdamW(model_j.parameters(), lr=3e-4, weight_decay=1e-3)
    sch3 = optim.lr_scheduler.CosineAnnealingLR(opt3, T_max=60, eta_min=1e-5)

    best_hit10_p3 = 0.0
    joint_path = save_dir / "cross_encoder_joint.pt"
    no_improve_j = 0

    for epoch in range(60):
        loss = run_epoch(model_j, joint_pairs, opt3, noise_std=0.05)
        sch3.step()

        hits, n = evaluate_hit_at_k(model_j, real_val, perfume_vectors, note_to_idx, sku_meta)
        h5, h10 = hits[5], hits[10]

        improved = ""
        if h10 >= best_hit10_p3:
            best_hit10_p3 = h10
            no_improve_j = 0
            torch.save({"model_state_dict": model_j.state_dict(), "note_dim": note_dim}, joint_path)
            improved = " *BEST*"
        else:
            no_improve_j += 1

        if (epoch + 1) % 10 == 0 or improved:
            print(f"  Epoch {epoch+1:3d}: loss={loss:.4f}, val Hit@5={h5:.1%}, Hit@10={h10:.1%}{improved} (n={n})")

        if no_improve_j >= 15:
            print(f"  Early stop at epoch {epoch+1}")
            break

    # ================================================================
    # Final: pick best, evaluate on test
    # ================================================================
    print(f"\n=== FINAL EVALUATION ===")
    print(f"Phase 2 (pretrain→finetune) best val Hit@10: {best_hit10_p2:.1%}")
    print(f"Phase 3 (joint training)    best val Hit@10: {best_hit10_p3:.1%}")

    if best_hit10_p3 > best_hit10_p2:
        print("→ Using joint training model")
        ckpt = torch.load(joint_path, map_location="cpu", weights_only=False)
        final_model = model_j
        torch.save(ckpt, best_path)
    else:
        print("→ Using pretrain→finetune model")
        ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
        final_model = model

    final_model.load_state_dict(ckpt["model_state_dict"])

    test_hits, n_test = evaluate_hit_at_k(final_model, test_pairs, perfume_vectors, note_to_idx, sku_meta)
    print(f"Test Hit@5 = {test_hits[5]:.1%}, Hit@10 = {test_hits[10]:.1%} (n={n_test})")

    val_hits, n_val = evaluate_hit_at_k(final_model, real_val, perfume_vectors, note_to_idx, sku_meta)
    print(f"Val  Hit@5 = {val_hits[5]:.1%}, Hit@10 = {val_hits[10]:.1%} (n={n_val})")

    print(f"\nModel saved: {best_path}")


if __name__ == "__main__":
    train()
