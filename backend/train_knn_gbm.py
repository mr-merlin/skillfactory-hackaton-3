"""Train kNN-augmented GBM ranker (kitchen-sink v2 config).

Best result: 15.0% Hit@10 (up from 9.5% baseline GBM).

Features: 20 base GBM features + 18 kNN v2 features:
  Per K/space combo (3 combos: raw_K20, raw_K50, norm_K50) x 6 features:
    - match_rate, avg_sim, weighted_match (base 3)
    - max_match_sim, pop_weighted_match, target_diversity (extended 3)
"""

import sys
import random
import pickle
from pathlib import Path

import numpy as np
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).parent))

from app.core.config import get_settings
from app.ranking.data import DataLoader
from app.ranking.profile.build_profile import session_to_user_vector
from app.ranking.normalize import build_synonym_map
from app.ranking.scoring.score import build_sku_vectors
from app.ranking.gbm.ranker import extract_features, FEATURE_NAMES
from train_gbm import build_sku_meta


KNN_FEATURE_NAMES = [
    "raw_k20_match", "raw_k20_sim", "raw_k20_weighted",
    "raw_k20_maxsim", "raw_k20_popw", "raw_k20_div",
    "raw_k50_match", "raw_k50_sim", "raw_k50_weighted",
    "raw_k50_maxsim", "raw_k50_popw", "raw_k50_div",
    "norm_k50_match", "norm_k50_sim", "norm_k50_weighted",
    "norm_k50_maxsim", "norm_k50_popw", "norm_k50_div",
]

ALL_FEATURE_NAMES = FEATURE_NAMES + KNN_FEATURE_NAMES


def compute_knn_extended(u_norm, pid, train_matrix, targets, K,
                         train_target_pop, exclude_idx=None):
    """Compute extended kNN features for a (user, candidate_pid) pair.

    Returns 6 features: match_rate, avg_sim, weighted_match,
                        max_match_sim, pop_weighted_match, target_diversity.
    """
    sims = train_matrix @ u_norm
    if exclude_idx is not None:
        sims[exclude_idx] = -1.0
    top_k = np.argsort(-sims)[:K]

    match_count = sum(1 for i in top_k if targets[i] == pid) / K
    avg_sim = float(np.mean(sims[top_k]))
    weighted_match = sum(sims[i] for i in top_k if targets[i] == pid) / max(sum(sims[top_k]), 1e-9)

    # Extended features
    matching_sims = [float(sims[i]) for i in top_k if targets[i] == pid]
    max_match_sim = max(matching_sims) if matching_sims else 0.0

    pop_weighted = sum(train_target_pop[i] for i in top_k if targets[i] == pid) / max(K, 1)

    unique_targets = len(set(targets[i] for i in top_k))
    target_diversity = unique_targets / K

    return np.array([match_count, avg_sim, weighted_match,
                     max_match_sim, pop_weighted, target_diversity], dtype=np.float64)


def _l2norm(M):
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    return M / norms


def train():
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
    catalog = sorted(perfume_vectors.keys())
    max_pop = max(popularity.values()) if popularity else 1.0
    dim = len(note_to_idx)
    dim_norm = len(norm_nti)

    # Load sessions and split
    sessions = loader.load_organ_sessions()
    shuffled = sessions.sample(frac=1, random_state=42)
    n_test = max(1, int(len(shuffled) * 0.2))
    train_sessions = shuffled.iloc[:-n_test]

    # Build real pairs
    real_pairs = []
    for _, row in train_sessions.iterrows():
        sid = int(row["session_id"])
        target = int(row["target_perfume_id"])
        uv = session_to_user_vector(sid, loader, use_presses=True, alpha_recipe=0.7)
        if uv and target in perfume_vectors:
            real_pairs.append((uv, target))

    random.seed(42)
    random.shuffle(real_pairs)
    n_train = len(real_pairs)
    print(f"Real pairs: {n_train}")

    # Build training user matrices (both raw and normalized spaces)
    train_U_raw = np.zeros((n_train, dim), dtype=np.float64)
    train_U_norm_space = np.zeros((n_train, dim_norm), dtype=np.float64)
    train_targets = np.zeros(n_train, dtype=np.int64)

    for i, (uv, target) in enumerate(real_pairs):
        for note, val in uv.items():
            idx = note_to_idx.get(note.strip().lower())
            if idx is not None:
                train_U_raw[i, idx] = val
            idx_n = norm_nti.get(note.strip().lower())
            if idx_n is not None:
                train_U_norm_space[i, idx_n] = val
        train_targets[i] = target

    train_U_raw_n = _l2norm(train_U_raw)
    train_U_norm_n = _l2norm(train_U_norm_space)

    # Pre-compute popularity for kNN neighbors (needed for pop_weighted feature)
    train_target_pop = np.array([popularity.get(int(t), 0) / max_pop for t in train_targets])

    def _base_feat(uv, pid):
        nv = norm_pv.get(pid)
        return extract_features(uv, perfume_vectors[pid], note_to_idx, idx_to_note,
                                popularity.get(pid, 0) / max_pop,
                                norm_sku_vec=nv, norm_note_to_idx=norm_nti,
                                sku_meta=sku_meta.get(pid) if sku_meta else None)

    def kitchen_sink_knn(uv, pid, exclude_idx):
        """Compute kitchen-sink v2 kNN features: multi-K + dual-space (18 features)."""
        # Raw space user vector
        u_raw = np.zeros(dim, dtype=np.float64)
        for note, val in uv.items():
            idx = note_to_idx.get(note.strip().lower())
            if idx is not None:
                u_raw[idx] = val
        u_raw_n = u_raw / max(np.linalg.norm(u_raw), 1e-9)

        # Normalized space user vector
        u_norm = np.zeros(dim_norm, dtype=np.float64)
        for note, val in uv.items():
            idx = norm_nti.get(note.strip().lower())
            if idx is not None:
                u_norm[idx] = val
        u_norm_n = u_norm / max(np.linalg.norm(u_norm), 1e-9)

        raw_k20 = compute_knn_extended(u_raw_n, pid, train_U_raw_n, train_targets, K=20,
                                        train_target_pop=train_target_pop, exclude_idx=exclude_idx)
        raw_k50 = compute_knn_extended(u_raw_n, pid, train_U_raw_n, train_targets, K=50,
                                        train_target_pop=train_target_pop, exclude_idx=exclude_idx)
        norm_k50 = compute_knn_extended(u_norm_n, pid, train_U_norm_n, train_targets, K=50,
                                         train_target_pop=train_target_pop, exclude_idx=exclude_idx)

        return np.concatenate([raw_k20, raw_k50, norm_k50])

    # Build dataset with kNN features
    # Use isolated RNG for reproducible negative sampling
    neg_rng = random.Random(42)
    X, y, groups = [], [], []
    for pair_idx, (uv, target) in enumerate(real_pairs):
        if target not in perfume_vectors or not uv:
            continue

        base = _base_feat(uv, target)
        knn = kitchen_sink_knn(uv, target, pair_idx)
        X.append(np.concatenate([base, knn]))
        y.append(1)

        neg_pool = [p for p in catalog if p != target]
        rand_chosen = neg_rng.sample(neg_pool, min(32, len(neg_pool)))
        for neg_pid in rand_chosen:
            base_neg = _base_feat(uv, neg_pid)
            knn_neg = kitchen_sink_knn(uv, neg_pid, pair_idx)
            X.append(np.concatenate([base_neg, knn_neg]))
            y.append(0)
        groups.append(1 + len(rand_chosen))

        if (pair_idx + 1) % 50 == 0:
            print(f"  Built {pair_idx + 1}/{n_train} groups...")

    X = np.vstack(X)
    y = np.array(y, dtype=np.int32)
    print(f"Samples: {len(y)}, groups: {len(groups)}, features: {X.shape[1]}")

    # Split: last 15% groups for validation (tuned: 15% > 20%)
    n_val = max(1, int(len(groups) * 0.15))
    tg, vg = groups[:-n_val], groups[-n_val:]
    te = sum(tg)

    train_ds = lgb.Dataset(X[:te], label=y[:te], group=tg, feature_name=ALL_FEATURE_NAMES)
    val_ds = lgb.Dataset(X[te:], label=y[te:], group=vg, feature_name=ALL_FEATURE_NAMES, reference=train_ds)

    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "eval_at": [5, 10],
        "num_leaves": 20,
        "learning_rate": 0.1,
        "min_child_samples": 30,
        "subsample": 0.6,
        "colsample_bytree": 0.7,
        "reg_alpha": 1.0,
        "reg_lambda": 5.0,
        "verbose": -1,
    }

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

    # --- Evaluate on test set ---
    test_sessions = shuffled.iloc[-n_test:]
    hits5, hits10, total = 0, 0, 0
    for _, row in test_sessions.iterrows():
        sid = int(row["session_id"])
        target_pid = int(row["target_perfume_id"])
        uv = session_to_user_vector(sid, loader, use_presses=True, alpha_recipe=0.7)
        if not uv:
            continue

        feats = []
        for pid in catalog:
            base = _base_feat(uv, pid)
            knn = kitchen_sink_knn(uv, pid, None)  # No exclusion at test time
            feats.append(np.concatenate([base, knn]))

        Xp = np.vstack(feats)
        scores = model.predict(Xp)
        top_idx = np.argsort(-scores)[:10]
        top_pids = [catalog[i] for i in top_idx]

        if target_pid in top_pids[:5]:
            hits5 += 1
        if target_pid in top_pids[:10]:
            hits10 += 1
        total += 1

    print(f"\nTest results: Hit@5={hits5/max(total,1):.1%}, Hit@10={hits10/max(total,1):.1%}, n={total}")

    # --- Save everything ---
    save_dir = Path(__file__).parent / "models"
    save_dir.mkdir(exist_ok=True)

    # Save GBM model
    model_path = save_dir / "knn_gbm_ranker.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    # Save kNN data needed for inference
    knn_data = {
        "train_U_raw_n": train_U_raw_n,
        "train_U_norm_n": train_U_norm_n,
        "train_targets": train_targets,
        "train_target_pop": train_target_pop,
        "note_to_idx": note_to_idx,
        "norm_nti": norm_nti,
    }
    knn_path = save_dir / "knn_data.pkl"
    with open(knn_path, "wb") as f:
        pickle.dump(knn_data, f)

    # Save SKU meta
    meta_path = save_dir / "sku_meta.pkl"
    with open(meta_path, "wb") as f:
        pickle.dump(sku_meta, f)

    # Feature importance
    print(f"\nFeature importance:")
    imp = model.feature_importance(importance_type="gain")
    for name, val in sorted(zip(ALL_FEATURE_NAMES, imp), key=lambda x: -x[1]):
        if val > 0:
            print(f"  {name:25s} {val:.1f}")

    print(f"\nModel saved: {model_path}")
    print(f"kNN data saved: {knn_path}")
    print(f"SKU meta saved: {meta_path}")


if __name__ == "__main__":
    train()
