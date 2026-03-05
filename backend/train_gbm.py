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
from app.ranking.normalize import build_synonym_map, ORGAN_NOTES
from app.ranking.scoring.score import build_sku_vectors
from app.ranking.gbm.ranker import extract_features, FEATURE_NAMES


def build_sku_meta(perfume_vectors, note_to_idx, perfume_notes_df, perfumes_df, popularity):
    """Precompute non-similarity features for each SKU."""
    # --- Collision groups (organ-note fingerprint) ---
    organ_idxs = [note_to_idx[n] for n in ORGAN_NOTES if n in note_to_idx]
    collision_groups = {}
    pid_to_fp = {}
    for pid, vec in perfume_vectors.items():
        fp = tuple(1 if vec[i] > 0 else 0 for i in organ_idxs)
        pid_to_fp[pid] = fp
        collision_groups.setdefault(fp, []).append(pid)

    # --- Note profile per perfume ---
    pn = perfume_notes_df.copy()
    pn["note"] = pn["note"].astype(str).str.strip().str.lower()
    pn["votes"] = pn["votes"].astype(float)
    note_stats = pn.groupby("perfume_id").agg(
        note_count=("votes", "count"),
        vote_sum=("votes", "sum"),
        vote_max=("votes", "max"),
    )
    top3_conc = pn.groupby("perfume_id")["votes"].apply(
        lambda x: x.nlargest(3).sum() / max(x.sum(), 1e-9)
    )

    def _entropy(votes):
        v = votes.values.astype(float)
        s = v.sum()
        if s <= 0:
            return 0.0
        p = v / s
        p = p[p > 0]
        return float(-np.sum(p * np.log(p)))

    note_ent = pn.groupby("perfume_id")["votes"].apply(_entropy)

    # --- Community ratings + seasons from perfumes_df ---
    perf = perfumes_df.set_index("perfume_id")

    sku_meta = {}
    for pid in perfume_vectors:
        fp = pid_to_fp[pid]
        group = collision_groups[fp]
        group_sorted = sorted(group, key=lambda p: popularity.get(p, 0), reverse=True)
        rank = group_sorted.index(pid)
        group_size = len(group)

        nc = int(note_stats.loc[pid]["note_count"]) if pid in note_stats.index else 0
        tc = float(top3_conc.get(pid, 0.0))
        ne = float(note_ent.get(pid, 0.0))

        meta = {
            "collision_group_size": float(group_size),
            "pop_rank_in_group": rank / max(group_size - 1, 1),
            "note_count": float(nc),
            "note_entropy": ne,
            "top3_concentration": tc,
            "love_ratio": 0.5,
            "longevity_idx": 0.5,
            "sillage_idx": 0.5,
        }

        if pid in perf.index:
            row = perf.loc[pid]
            cl = float(row.get("clslove", 50))
            cd = float(row.get("clsdislike", 50))
            meta["love_ratio"] = cl / max(cl + cd, 1e-9)

            ls = [float(row.get(f"longs{i}", 0)) for i in range(1, 6)]
            ls_sum = sum(ls) or 1e-9
            meta["longevity_idx"] = sum(i * v for i, v in enumerate(ls)) / (4.0 * ls_sum)

            ss = [float(row.get(f"sil{i}", 0)) for i in range(1, 5)]
            ss_sum = sum(ss) or 1e-9
            meta["sillage_idx"] = sum(i * v for i, v in enumerate(ss)) / (3.0 * ss_sum)

        sku_meta[pid] = meta

    return sku_meta


def build_dataset(
    pairs: list[tuple[dict, int]],
    perfume_vectors, note_to_idx, idx_to_note, popularity,
    norm_perfume_vectors, norm_note_to_idx,
    sku_meta=None,
    neg_random=16,
):
    catalog = sorted(perfume_vectors.keys())
    max_pop = max(popularity.values()) if popularity else 1.0

    def _feat(uv, pid):
        nv = norm_perfume_vectors.get(pid)
        return extract_features(uv, perfume_vectors[pid], note_to_idx, idx_to_note,
                                popularity.get(pid, 0) / max_pop,
                                norm_sku_vec=nv, norm_note_to_idx=norm_note_to_idx,
                                sku_meta=sku_meta.get(pid) if sku_meta else None)

    X, y, groups = [], [], []

    for uv, target in pairs:
        if target not in perfume_vectors:
            continue
        if not uv:
            continue

        X.append(_feat(uv, target))
        y.append(1)

        neg_pool = [p for p in catalog if p != target]
        n_rand = min(neg_random, len(neg_pool))
        rand_chosen = random.sample(neg_pool, n_rand)

        for neg_pid in rand_chosen:
            X.append(_feat(uv, neg_pid))
            y.append(0)

        groups.append(1 + n_rand)

    return np.vstack(X), np.array(y, dtype=np.int32), groups


def train():
    settings = get_settings()
    loader = DataLoader(
        perfume_dir=settings.data_perfume_dir,
        organ_dir=settings.data_organ_dir if settings.data_organ_dir.exists() else None,
    )

    perfume_notes = loader.load_perfume_notes()
    perfume_vectors, note_to_idx, idx_to_note = build_sku_vectors(perfume_notes)

    catalog_notes = perfume_notes["note"].astype(str).str.strip().str.lower().unique().tolist()
    syn_map = build_synonym_map(catalog_notes)
    norm_pv, norm_nti, _ = build_sku_vectors(perfume_notes, synonym_map=syn_map)

    perfumes = loader.load_perfumes()
    popularity = {}
    if "allVotes" in perfumes.columns:
        popularity = perfumes.set_index("perfume_id")["allVotes"].to_dict()

    # Real pairs
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
    print(f"Real pairs: {len(real_pairs)}")

    random.seed(42)
    random.shuffle(real_pairs)

    # Precompute non-similarity SKU metadata
    sku_meta = build_sku_meta(perfume_vectors, note_to_idx, perfume_notes, perfumes, popularity)
    print(f"SKU meta computed for {len(sku_meta)} perfumes")

    X, y, groups = build_dataset(
        real_pairs, perfume_vectors, note_to_idx, idx_to_note, popularity,
        norm_pv, norm_nti,
        sku_meta=sku_meta,
        neg_random=16,
    )
    print(f"Samples: {len(y)}, groups: {len(groups)}, label range: [{y.min():.2f}, {y.max():.2f}]")

    # Split: last 20% groups for validation
    n_val_groups = max(1, int(len(groups) * 0.2))
    train_groups = groups[:-n_val_groups]
    val_groups = groups[-n_val_groups:]

    train_end = sum(train_groups)
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:], y[train_end:]

    train_ds = lgb.Dataset(X_train, label=y_train, group=train_groups, feature_name=FEATURE_NAMES)
    val_ds = lgb.Dataset(X_val, label=y_val, group=val_groups, feature_name=FEATURE_NAMES, reference=train_ds)

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

    save_dir = Path(__file__).parent / "models"
    save_dir.mkdir(exist_ok=True)

    save_path = save_dir / "gbm_ranker.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(model, f)

    meta_path = save_dir / "sku_meta.pkl"
    with open(meta_path, "wb") as f:
        pickle.dump(sku_meta, f)

    print(f"\nFeature importance:")
    imp = model.feature_importance(importance_type="gain")
    for name, val in sorted(zip(FEATURE_NAMES, imp), key=lambda x: -x[1]):
        print(f"  {name:20s} {val:.1f}")

    print(f"\nModel saved: {save_path}")
    print(f"SKU meta saved: {meta_path}")


if __name__ == "__main__":
    train()
